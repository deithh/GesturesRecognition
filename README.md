# WandSense IMU • Telemetry & Gesture Recognition Platform

Kompletna platforma do akwizycji danych sensorycznych, automatycznego przycinania ciszy (VAD/MAD), kalibracji, treningu modeli uczenia maszynowego (**Conv-LSTM / BiLSTM**, **1D-CNN w PyTorch**, **Random Forest w scikit-learn**), prezentacji **LIVE DEMO** w czasie rzeczywistym oraz gry zręcznościowej **Space Invaders** sterowanej gestami w powietrzu.

Projekt współpracuje z bezprzewodową różdżką/pałeczką IMU opartą o mikrokontroler **ESP32** z czujnikami **LSM6DS3TR-C** (akcelerometr + żyroskop 6-DOF) oraz **MMC5603** (magnetometr 3-DOF), komunikującą się przez Wi-Fi (protokół hybrydowy: sterowanie TCP + telemetria UDP 100 Hz).

---

## Obsługiwane gesty (7 klas)

| ID | Nazwa kodowa | Etykieta | Opis ruchu |
|---|---|---|---|
| 1 | `circle` | Circle | Ruch okrężny w powietrzu |
| 2 | `go_up` | Swipe Up | Szybkie pociągnięcie w górę |
| 3 | `go_down` | Swipe Down | Szybkie pociągnięcie w dół |
| 4 | `go_left` | Swipe Left | Szybkie pociągnięcie w lewo |
| 5 | `go_right` | Swipe Right | Szybkie pociągnięcie w prawo |
| 6 | `iks` | Cross (X) | Gest ukośnego krzyżyka / X |
| 7 | `idle` | Rest (Idle) | Szum spoczynkowy / brak zamierzonego gestu |

---

## Architektura AI i modele

Projekt zawiera 3 silniki wnioskowania z możliwością przełączania w locie:
1. **Conv-LSTM (1D-CNN + BiLSTM w PyTorch):**
   - Warstwy splotowe 1D wyciągające cechy przestrzenno-kinetyczne połączone z dwukierunkową siecią LSTM modelującą zależność czasową.
2. **1D-CNN (PyTorch):**
   - Lekka i szybka głęboka sieć konwolucyjna 1D z normalizacją wsadową (*BatchNorm*) i warstwami *Dropout*.
3. **Random Forest (scikit-learn):**
   - Klasyfikator cech statystyczno-czasowych (wartości średnie, wariancje, energie, percentyle, piki prędkości kątowej i całki ruchu).

> **Uwaga:** Zgodnie z dobrymi praktykami wagi modeli (`*.pth`, `*.joblib`) są wykluczone z repozytorium git (`.gitignore`). Dataset z nagraniami CSV znajduje się w folderze `dataset/`. Aby wytrenować modele od zera, wystarczy uruchomić:
> ```powershell
> python train_models.py
> ```
> lub wybrać opcję `[R]` w menu głównym `recorder_app.py`.

---

## Szybki start

### Wymagania i środowisko Conda
```powershell
conda create -n r_imu python=3.11
conda activate r_imu
pip install -r requirements.txt
```

### 1. Centrum dowodzenia platformą (Claude Code CLI)
```powershell
python recorder_app.py
```

Dostępne moduły:
* **`[1]` INFINITE MODE:** Ciągłe zbieranie danych z automatycznym balanserem klas, podpowiedziami brakujących prędkości (Fast/Slow) i skrótami klawiszowymi `[1]`-`[7]`.
* **`[2]` Paced Series:** Nagrywanie serii prób z odliczaniem i trójwymiarową inspekcją trajektorii 6-DOF w oknie Pygame.
* **`[3]` Manual Trigger:** Ręczne wyzwalanie nagrywania klawiszem `[ENTER]`.
* **`[4]` Duration & Speed Report:** Raport rozkładu czasów trwania gestów z wykrywaniem luk prędkościowych (za mało szybkich / wolnych wykonań).
* **`[5]` Live Gyro Monitor:** Monitor prędkości kątowej `||ω||` z kalibracją progu ciszy na żywo.
* **`[6]` Adjust Threshold:** Strojenie progu bramkowania ciszy (VAD/MAD).
* **`[7]` Sensor Calibration:** Kalibracja biasów żyroskopu i skali grawitacyjnej akcelerometru (zapis do `calibration.json`).
* **`[8]` 3D Visualizer:** Interaktywna wizualizacja orientacji pałki w 3D (AHRS EKF / Madgwick).
* **`[9]` LIVE DEMO:** Rozpoznawanie gestów w czasie rzeczywistym z ciągłym oknem przesuwnym i detektorem stacjonarności.
* **`[G]` WAND SPACE INVADERS:** Retro-arkadowa gra Space Invaders sterowana w 100% gestami różdżki.
* **`[M]` Switch Model:** Błyskawiczne przełączanie aktywnego silnika AI (Conv-LSTM ↔ 1D-CNN ↔ Random Forest).
* **`[B]` Dynamic UDP Batching:** Zmiana rozmiaru batcha UDP (1-32 próbki) w locie bez restartu firmware'u ESP32.
* **`[R]` Retrain Models:** Pełny retraining modeli z walidacją 5-Fold Cross-Validation.

---

### 2. Prezentacja LIVE DEMO (Sliding Window & Stationarity Gating)
```powershell
python live_recognition.py
```
- Trzymaj pałeczkę spokojnie = stan `READY (IDLE)`.
- Wykonaj gest w powietrzu i zatrzymaj dłoń.
- System automatycznie wykrywa początek i koniec gestu, estymuje prawdopodobieństwa klas i wyświetla wynik wraz z dźwiękiem.
- Klawisz `[N]` przełącza model AI w locie, `[SPACE]` / `[R]` taruje orientację 3D.

---

### 3. Gra zręcznościowa: Wand Space Invaders
```powershell
python wand_invaders.py
```
Sterowanie gestami:
- **Swipe Left (`go_left`):** Błyskawiczny unik w lewo (*Dash Left*).
- **Swipe Right (`go_right`):** Błyskawiczny unik w prawo (*Dash Right*).
- **Swipe Up (`go_up`):** Podjazd działkiem w górę (*Advance*).
- **Swipe Down (`go_down`):** Wycofanie działka w dół za bunkry (*Retreat*).
- **Circle (`circle`):** Strzał z podwójnych działek laserowych.
- **X-Mark (`iks`):** **Ładowane Ulti (Orbital Ion Cannon)** – potężny pionowy słup jonowy niszczący całe kolumny obcych.
- **Key `[N]`:** Przełączanie aktywnego modelu AI w trakcie gry.
- *(Zapasowo działa sterowanie klawiaturą: Strzałki/WASD, Space, X, N, ESC).*

---

## Firmware ESP32 (`esp32_firmware/`)

Plik szkicu Arduino/ESP-IDF: [`esp32_firmware/esp32_firmware.ino`](file:///c:/Users/jasiu/Desktop/mgr/sem2/roboty/pały/esp32_firmware/esp32_firmware.ino).
- Częstotliwość próbkowania: **100 Hz** (10 ms).
- Konfigurowalne dynamiczne batchowanie próbek (1–32 próbek w pakiecie UDP) sterowane komendą TCP `B<size>\n`.
- Automatyczne wznawianie połączenia (*reconnect loop*).
- Wsparcie dla mDNS (`acer.local`) oraz fallback na adres IP.
