import os
import glob
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import stats

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Ścieżka do datasetu i katalogu wyjściowego modeli
DATASET_DIR = "dataset"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# 7 gestów w zdefiniowanej kolejności
GESTURE_NAMES = [
    "circle",
    "go_up",
    "go_down",
    "go_left",
    "go_right",
    "iks",
    "idle"
]
LABEL_TO_IDX = {name: idx for idx, name in enumerate(GESTURE_NAMES)}
IDX_TO_LABEL = {idx: name for idx, name in enumerate(GESTURE_NAMES)}

# Stała długość sekwencji czasowej (T próbek przy 100 Hz = 1.0 sekundy)
TARGET_LEN = 100

# 6 kanałów wejściowych: 3x accel + 3x gyro
FEATURE_COLS = [
    "cal_accel_x", "cal_accel_y", "cal_accel_z",
    "cal_gyro_x",  "cal_gyro_y",  "cal_gyro_z"
]


# =====================================================================
# MODEL 1D-CNN (PyTorch)
# =====================================================================
class Gesture1DCNN(nn.Module):
    def __init__(self, in_channels=6, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            # Blok 1: wejście (B, 6, 100) -> wyjście (B, 32, 50)
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.15),

            # Blok 2: (B, 32, 50) -> (B, 64, 25)
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2),

            # Blok 3: (B, 64, 25) -> (B, 128, 25)
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global Average Pooling -> (B, 128, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (B, 6, 100)
        feat = self.features(x)
        out = self.classifier(feat)
        return out


# =====================================================================
# MODEL CONV-LSTM (1D-CNN + Bidirectional LSTM)
# =====================================================================
class GestureConvLSTM(nn.Module):
    def __init__(self, in_channels=6, num_classes=7):
        super().__init__()
        # Ekstrakcja cech przestrzennych z 6 osi IMU
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # (B, 32, 50)
            nn.Dropout(0.15),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # (B, 64, 25)
            nn.Dropout(0.2)
        )
        # Warstwa rekurencyjna Bi-LSTM modelująca dynamikę czasową i kolejność faz ruchu
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, 6, 100)
        c = self.conv(x)  # (B, 64, 25)
        c = c.transpose(1, 2)  # (B, 25, 64) pod format (Batch, Seq, Features)
        lstm_out, _ = self.lstm(c)  # (B, 25, 128)
        feat = torch.mean(lstm_out, dim=1)  # Temporal pooling -> (B, 128)
        return self.classifier(feat)


# =====================================================================
# 1. WCZYTYWANIE I NORMALIZACJA CZASOWA PRÓBEK
# =====================================================================
def resample_sequence(data_matrix, target_len=TARGET_LEN):
    """
    Interpoluje sekwencję (N_próbek, C_kanałów) do stałej liczby punktów target_len.
    Zapewnia stały wymiar tensora dla sieci 1D-CNN.
    """
    n_samples, n_channels = data_matrix.shape
    if n_samples == target_len:
        return data_matrix

    x_old = np.linspace(0, 1, n_samples)
    x_new = np.linspace(0, 1, target_len)
    resampled = np.zeros((target_len, n_channels), dtype=np.float32)
    for c in range(n_channels):
        f = interp1d(x_old, data_matrix[:, c], kind="linear", fill_value="extrapolate")
        resampled[:, c] = f(x_new)
    return resampled


def extract_rf_features(seq_matrix):
    """
    Ekstrakcja bogatego wektora cech statystyczno-czasowych z 6 kanałów
    oraz norm przyspieszenia i prędkości kątowej dla Random Forest.
    """
    feats = []
    # 6 kanałów podstawowych
    for c in range(seq_matrix.shape[1]):
        sig = seq_matrix[:, c]
        feats.extend([
            np.mean(sig),
            np.std(sig),
            np.min(sig),
            np.max(sig),
            np.max(sig) - np.min(sig),
            np.median(sig),
            float(stats.skew(sig)),
            float(stats.kurtosis(sig)),
            np.mean(sig**2),            # Energia
            np.sqrt(np.mean(sig**2)),   # RMS
            np.percentile(sig, 25),
            np.percentile(sig, 75),
            np.sum(np.diff(np.sign(sig)) != 0) # Zero crossings
        ])

    # Norma przyspieszenia i norma żyroskopu
    acc_norm = np.linalg.norm(seq_matrix[:, 0:3], axis=1)
    gyr_norm = np.linalg.norm(seq_matrix[:, 3:6], axis=1)
    for sig in [acc_norm, gyr_norm]:
        feats.extend([
            np.mean(sig),
            np.std(sig),
            np.min(sig),
            np.max(sig),
            np.max(sig) - np.min(sig),
            np.sqrt(np.mean(sig**2))
        ])

    return np.array(feats, dtype=np.float32)


def load_dataset():
    """Wczytuje wszystkie pliki CSV ze struktury dataset/."""
    sequences = []
    rf_features = []
    labels = []
    file_paths = []

    print("[1/4] Wczytywanie plików CSV z datasetu...")
    for gesture in GESTURE_NAMES:
        folder = os.path.join(DATASET_DIR, gesture)
        csv_files = glob.glob(os.path.join(folder, "*.csv"))
        print(f"  -> Gest '{gesture}': {len(csv_files)} plików")

        for fpath in csv_files:
            try:
                df = pd.read_csv(fpath)
                # Sprawdź dostępność kolumn skalibrowanych lub surowych
                if all(col in df.columns for col in FEATURE_COLS):
                    mat = df[FEATURE_COLS].values.astype(np.float32)
                else:
                    raw_cols = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
                    mat = df[raw_cols].values.astype(np.float32)

                if len(mat) < 5:
                    continue

                resampled = resample_sequence(mat, TARGET_LEN)
                feats = extract_rf_features(resampled)

                sequences.append(resampled)
                rf_features.append(feats)
                labels.append(LABEL_TO_IDX[gesture])
                file_paths.append(fpath)
            except Exception as e:
                print(f"     [!] Błąd wczytywania {fpath}: {e}")

    sequences = np.array(sequences, dtype=np.float32)      # Shape: (N, 100, 6)
    rf_features = np.array(rf_features, dtype=np.float32)  # Shape: (N, D_feats)
    labels = np.array(labels, dtype=np.int64)

    print(f"\nŁącznie wczytano: {len(labels)} próbek z {len(np.unique(labels))} klas.")
    return sequences, rf_features, labels, file_paths


def plot_and_save_confusion_matrix(cm, class_names, title, out_path):
    """Rysuje i zapisuje estetyczną macierz pomyłek w matplotlib."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel='Prawdziwa klasa (True Label)',
        xlabel='Predykcja modelu (Predicted Label)'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f" [OK] Zapisano macierz pomyłek: {out_path}")


# =====================================================================
# 2. TRENING RANDOM FOREST (RF)
# =====================================================================
def train_random_forest(X_feats, y):
    print("\n" + "="*50)
    print("[2/4] TRENING RANDOM FOREST (Stratified 5-Fold CV)")
    print("="*50)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracies = []
    all_true = []
    all_pred = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_feats, y)):
        X_train, X_val = X_feats[train_idx], X_feats[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=2,
            random_state=42 + fold,
            class_weight='balanced'
        )
        clf.fit(X_train, y_train)
        y_val_pred = clf.predict(X_val)

        acc = accuracy_score(y_val, y_val_pred)
        cv_accuracies.append(acc)
        all_true.extend(y_val)
        all_pred.extend(y_val_pred)
        print(f"  Fold {fold+1}: Dokładność (Accuracy) = {acc*100:5.1f}%")

    mean_acc = np.mean(cv_accuracies)
    print(f"\nŚrednia dokładność Random Forest (5-Fold CV): {mean_acc*100:5.2f}%")

    # Wytrenuj finalny model na całym zbiorze danych
    final_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=14,
        min_samples_split=2,
        random_state=42,
        class_weight='balanced'
    )
    final_rf.fit(X_feats, y)

    rf_path = os.path.join(MODELS_DIR, "random_forest_model.joblib")
    joblib.dump(final_rf, rf_path)
    print(f" [OK] Zapisano model Random Forest: {rf_path}")

    # Raport i macierz pomyłek CV
    cm = confusion_matrix(all_true, all_pred)
    plot_and_save_confusion_matrix(
        cm, GESTURE_NAMES,
        f"Random Forest - Confusion Matrix (CV Mean: {mean_acc*100:.1f}%)",
        os.path.join(MODELS_DIR, "rf_confusion_matrix.png")
    )
    print("\nRaport klasyfikacji Random Forest (OOF Predictions):")
    print(classification_report(all_true, all_pred, target_names=GESTURE_NAMES))

    return final_rf, mean_acc


# =====================================================================
# 3. TRENING 1D-CNN (PyTorch)
# =====================================================================
def train_1d_cnn(X_seqs, y):
    print("\n" + "="*50)
    print("[3/4] TRENING 1D-CNN (PyTorch)")
    print("="*50)

    # Funkcja augmentacji próbek (jittering + time scaling)
    def augment_sample(sample_matrix):
        # sample_matrix: (100, 6)
        aug = sample_matrix.copy()
        # 1. Delikatny szum Gaussa
        aug += np.random.normal(0, 0.05, size=aug.shape)
        # 2. Skalowanie amplitudy +/- 8%
        scale = np.random.uniform(0.92, 1.08)
        aug *= scale
        return aug.astype(np.float32)

    # Przygotowanie danych (transpozycja z (N, 100, 6) na (N, 6, 100) pod Conv1D)
    X_perm = np.transpose(X_seqs, (0, 2, 1))  # (N, 6, 100)

    # 5-krotna walidacja krzyżowa (5-Fold CV)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracies = []
    all_true = []
    all_pred = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    EPOCHS = 30
    BATCH_SIZE = 16

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_perm, y)):
        # Augmentacja zbioru treningowego (x3)
        X_tr_list, y_tr_list = [], []
        for idx in train_idx:
            orig = X_seqs[idx]
            lbl = y[idx]
            X_tr_list.append(orig)
            y_tr_list.append(lbl)
            # 2 augmentowane kopie
            X_tr_list.append(augment_sample(orig))
            y_tr_list.append(lbl)
            X_tr_list.append(augment_sample(orig))
            y_tr_list.append(lbl)

        X_train_aug = np.transpose(np.array(X_tr_list, dtype=np.float32), (0, 2, 1))
        y_train_aug = np.array(y_tr_list, dtype=np.int64)

        X_val_fold = X_perm[val_idx]
        y_val_fold = y[val_idx]

        train_ds = TensorDataset(torch.from_numpy(X_train_aug), torch.from_numpy(y_train_aug))
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        model = Gesture1DCNN(in_channels=6, num_classes=len(GESTURE_NAMES)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        # Trening
        model.train()
        for epoch in range(EPOCHS):
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
            scheduler.step()

        # Walidacja
        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(X_val_fold).to(device)
            logits = model(vx)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        acc = accuracy_score(y_val_fold, preds)
        cv_accuracies.append(acc)
        all_true.extend(y_val_fold)
        all_pred.extend(preds)
        print(f"  Fold {fold+1}: Dokładność (Accuracy) = {acc*100:5.1f}%")

    mean_acc = np.mean(cv_accuracies)
    print(f"\nŚrednia dokładność 1D-CNN (5-Fold CV): {mean_acc*100:5.2f}%")

    # Finalny model uczony na całym zbiorze z augmentacją
    print("\nTrenowanie finalnego modelu 1D-CNN na pełnym zbiorze danych...")
    X_full_list, y_full_list = [], []
    for i in range(len(X_seqs)):
        orig = X_seqs[i]
        lbl = y[i]
        X_full_list.append(orig)
        y_full_list.append(lbl)
        X_full_list.append(augment_sample(orig))
        y_full_list.append(lbl)
        X_full_list.append(augment_sample(orig))
        y_full_list.append(lbl)

    X_full_aug = np.transpose(np.array(X_full_list, dtype=np.float32), (0, 2, 1))
    y_full_aug = np.array(y_full_list, dtype=np.int64)

    full_ds = TensorDataset(torch.from_numpy(X_full_aug), torch.from_numpy(y_full_aug))
    full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True)

    final_model = Gesture1DCNN(in_channels=6, num_classes=len(GESTURE_NAMES)).to(device)
    optimizer = optim.AdamW(final_model.parameters(), lr=0.003, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35)
    criterion = nn.CrossEntropyLoss()

    final_model.train()
    for epoch in range(35):
        for bx, by in full_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(final_model(bx), by)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Zapis wag modelu PyTorch
    torch_path = os.path.join(MODELS_DIR, "gesture_1d_cnn.pth")
    torch.save(final_model.state_dict(), torch_path)
    print(f" [OK] Zapisano wagi modelu 1D-CNN: {torch_path}")

    # Raport i macierz pomyłek CV
    cm = confusion_matrix(all_true, all_pred)
    plot_and_save_confusion_matrix(
        cm, GESTURE_NAMES,
        f"1D-CNN - Confusion Matrix (CV Mean: {mean_acc*100:.1f}%)",
        os.path.join(MODELS_DIR, "cnn_confusion_matrix.png")
    )
    print("\nRaport klasyfikacji 1D-CNN (OOF Predictions):")
    print(classification_report(all_true, all_pred, target_names=GESTURE_NAMES))

    return final_model, mean_acc


# =====================================================================
# 3b. TRENING MODELU CONV-LSTM (PyTorch)
# =====================================================================
def train_conv_lstm(sequences, labels):
    print("\n" + "="*50)
    print("TRENING MODELU CONV-LSTM (1D-CNN + BiLSTM w PyTorch)")
    print("="*50)

    X_seqs = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    def augment_sample(seq):
        aug = seq.copy()
        aug += np.random.normal(0, 0.05, size=aug.shape)
        scale = np.random.uniform(0.92, 1.08)
        aug *= scale
        return aug.astype(np.float32)

    X_perm = np.transpose(X_seqs, (0, 2, 1))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracies, all_true, all_pred = [], [], []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 60
    BATCH_SIZE = 16

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_perm, y)):
        X_tr_list, y_tr_list = [], []
        for idx in train_idx:
            orig = X_seqs[idx]
            lbl = y[idx]
            X_tr_list.append(orig)
            y_tr_list.append(lbl)
            X_tr_list.append(augment_sample(orig))
            y_tr_list.append(lbl)
            X_tr_list.append(augment_sample(orig))
            y_tr_list.append(lbl)

        X_train_aug = np.transpose(np.array(X_tr_list, dtype=np.float32), (0, 2, 1))
        y_train_aug = np.array(y_tr_list, dtype=np.int64)
        X_val_fold, y_val_fold = X_perm[val_idx], y[val_idx]

        train_ds = TensorDataset(torch.from_numpy(X_train_aug), torch.from_numpy(y_train_aug))
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        model = GestureConvLSTM(in_channels=6, num_classes=len(GESTURE_NAMES)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        model.train()
        for epoch in range(EPOCHS):
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(torch.from_numpy(X_val_fold).to(device)), dim=1).cpu().numpy()

        acc = accuracy_score(y_val_fold, preds)
        cv_accuracies.append(acc)
        all_true.extend(y_val_fold)
        all_pred.extend(preds)
        print(f"  Fold {fold+1}: Dokładność (Accuracy) = {acc*100:5.1f}%")

    mean_acc = np.mean(cv_accuracies)
    print(f"\nŚrednia dokładność Conv-LSTM (5-Fold CV): {mean_acc*100:5.2f}%")

    # Finalny model na pełnym zbiorze
    X_full_list, y_full_list = [], []
    for i in range(len(X_seqs)):
        orig = X_seqs[i]
        lbl = y[i]
        X_full_list.append(orig)
        y_full_list.append(lbl)
        X_full_list.append(augment_sample(orig))
        y_full_list.append(lbl)
        X_full_list.append(augment_sample(orig))
        y_full_list.append(lbl)

    X_full_aug = np.transpose(np.array(X_full_list, dtype=np.float32), (0, 2, 1))
    y_full_aug = np.array(y_full_list, dtype=np.int64)

    full_ds = TensorDataset(torch.from_numpy(X_full_aug), torch.from_numpy(y_full_aug))
    full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True)

    final_model = GestureConvLSTM(in_channels=6, num_classes=len(GESTURE_NAMES)).to(device)
    optimizer = optim.AdamW(final_model.parameters(), lr=0.003, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35)
    criterion = nn.CrossEntropyLoss()

    final_model.train()
    for epoch in range(35):
        for bx, by in full_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(final_model(bx), by)
            loss.backward()
            optimizer.step()
        scheduler.step()

    torch_path = os.path.join(MODELS_DIR, "gesture_conv_lstm.pth")
    torch.save(final_model.state_dict(), torch_path)
    print(f" [OK] Zapisano wagi modelu Conv-LSTM: {torch_path}")

    return final_model, mean_acc


# =====================================================================
# 4. ZAPIS METADANYCH MODELI (Do Live Demo)
# =====================================================================
def save_metadata(rf_acc, cnn_acc, lstm_acc=None):
    metadata = {
        "gesture_names": GESTURE_NAMES,
        "label_to_idx": LABEL_TO_IDX,
        "idx_to_label": IDX_TO_LABEL,
        "target_len": TARGET_LEN,
        "feature_cols": FEATURE_COLS,
        "in_channels": len(FEATURE_COLS),
        "num_classes": len(GESTURE_NAMES),
        "rf_cv_accuracy": float(rf_acc),
        "cnn_cv_accuracy": float(cnn_acc),
        "lstm_cv_accuracy": float(lstm_acc) if lstm_acc is not None else 0.0
    }
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f" [OK] Zapisano metadane modeli: {meta_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train IMU Gesture Recognition Models")
    parser.add_argument("--lstm", action="store_true", default=False, help="Also train Conv-LSTM model (takes longer)")
    args, _ = parser.parse_known_args()

    sequences, rf_features, labels, file_paths = load_dataset()
    if len(labels) == 0:
        print("[ERROR] No data found in dataset/ folder!")
        return

    # 1. Random Forest (Fast, high accuracy)
    rf_model, rf_acc = train_random_forest(rf_features, labels)

    # 2. 1D-CNN (Fast, deep representation)
    cnn_model, cnn_acc = train_1d_cnn(sequences, labels)

    # 3. Conv-LSTM (CNN + BiLSTM, disabled by default for rapid training)
    lstm_acc = None
    if args.lstm:
        lstm_model, lstm_acc = train_conv_lstm(sequences, labels)
    else:
        # Preserve previously saved lstm accuracy if exists
        meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    old_meta = json.load(f)
                    lstm_acc = old_meta.get("lstm_cv_accuracy", 0.0)
            except Exception:
                pass
        print("\n[*] Conv-LSTM training skipped by default (use --lstm to train Conv-LSTM).")

    # 4. Save metadata
    save_metadata(rf_acc, cnn_acc, lstm_acc)

    print("\n" + "="*50)
    print("TRAINING SUMMARY:")
    print(f" -> Random Forest (5-Fold CV): {rf_acc*100:.2f}%")
    print(f" -> 1D-CNN PyTorch (5-Fold CV): {cnn_acc*100:.2f}%")
    if args.lstm:
        print(f" -> Conv-LSTM      (5-Fold CV): {lstm_acc*100:.2f}%")
    print(f" Models and confusion matrices saved to: {MODELS_DIR}/")
    print("="*50)


if __name__ == "__main__":
    main()
