#include <Wire.h>
#include <Adafruit_LSM6DS3TRC.h>
#include <Adafruit_MMC56x3.h>
#include <WiFi.h>
#include <lwip/sockets.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <lwip/inet.h>
#include <arpa/inet.h>
#include <ESPmDNS.h>

const char* ssid = "Play-163-2.4GHz";      // UWAGA: ESP32 obsługuje tylko Wi-Fi 2.4 GHz!
const char* password = "Zimnawoda";
const char* remote_host = "acer.local";
const char* remote_ip = "192.168.0.15";    // Aktualne IP Twojego laptopa na Play-163-2.4GHz
const char start_streaming = '0';
const char stop_streaming  = '1';
const char prepare_cmd     = '2';          // opcjonalna komenda przygotowania (mrugnięcie)
const int remote_port = 2137;

const int config_pin = 4;
const int bat_pin = 39;
const int status_pin = 32;
const int stream_pin = 33;

const float r_plus = 100000.0;
const float r_gnd = 56000.0;

const int frequency = 100;
const unsigned long stream_interval = 1000 / frequency; // 10 ms (100 Hz)

// Dynamiczne batchowanie próbek (konfigurowalne w locie z recorder_app komendą B<size>\n)
const int MAX_BATCH_CAPACITY = 32;
int current_batch_size = 10; // Domyślnie 10 próbek na pakiet UDP (100 ms)

struct __attribute__((packed)) SensorData {
  uint32_t timestamp_ms;

  float accel_x;
  float accel_y;
  float accel_z;

  float gyro_x;
  float gyro_y;
  float gyro_z;

  float mag_x;
  float mag_y;
  float mag_z;

  bool id;
};

unsigned long lastBlink = 0;
unsigned long lastBatCheck = 0;
unsigned long last_stream_time = 0;
bool bat_status = 1;
bool bat_state = 1;
bool id = 0;
bool is_streaming = false;
int streaming_sock_fd = -1;
int command_sock_fd = -1;

Adafruit_LSM6DS3TRC lsm6ds3trc;
Adafruit_MMC5603 mmc5603;
sensors_event_t accel, gyro, magn, temp;

SensorData data;
SensorData batch_buffer[MAX_BATCH_CAPACITY];
int batch_index = 0;

struct sockaddr_in remote_addr;
char buffer[32]; // Bufor na komendy TCP (np. "B3\n", "0", "1")

void setData() {
  data.timestamp_ms = millis();

  data.accel_x = accel.acceleration.x;
  data.accel_y = accel.acceleration.y;
  data.accel_z = accel.acceleration.z;

  data.gyro_x = gyro.gyro.x;
  data.gyro_y = gyro.gyro.y;
  data.gyro_z = gyro.gyro.z;

  data.mag_x = magn.magnetic.x;
  data.mag_y = magn.magnetic.y;
  data.mag_z = magn.magnetic.z;

  data.id = id;
}

bool resolveRemote(struct sockaddr_in* out_addr) {
  Serial.print("Resolving host: ");
  Serial.println(remote_host);

  memset(out_addr, 0, sizeof(struct sockaddr_in));
  out_addr->sin_family = AF_INET;
  out_addr->sin_port = htons(remote_port);

  // 1. Próba przez mDNS (dla nazw *.local, np. acer.local -> "acer")
  if (MDNS.begin("imu-wand")) {
    String hostNoLocal = String(remote_host);
    if (hostNoLocal.endsWith(".local")) {
      hostNoLocal.replace(".local", "");
    }
    IPAddress mdnsIp = MDNS.queryHost(hostNoLocal.c_str(), 2000);
    if ((uint32_t)mdnsIp != 0 && (uint32_t)mdnsIp != INADDR_NONE) {
      out_addr->sin_addr.s_addr = (uint32_t)mdnsIp;
      Serial.print("mDNS resolution success: ");
      Serial.println(mdnsIp);
      return true;
    }
  }

  // 2. Próba przez standardowy DNS routera
  struct hostent* he = gethostbyname(remote_host);
  if (he && he->h_addr_list && he->h_addr_list[0]) {
    memcpy(&out_addr->sin_addr, he->h_addr_list[0], sizeof(struct in_addr));
    Serial.println("DNS resolution success");
    return true;
  }

  // 3. Fallback na wpisane IP laptopa (192.168.0.15)
  Serial.print("Trying fallback IP: ");
  Serial.println(remote_ip);

  if (inet_aton(remote_ip, &out_addr->sin_addr) != 0) {
    Serial.println("Fallback IP ok");
    return true;
  }

  Serial.println("Resolving failed completely");
  return false; 
}

void connectWiFi() {
  Serial.print("Connecting to WiFi");
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false); // Wyłączenie usypiania radia WiFi (eliminuje przycięcia co 100-300 ms!)
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("\nWiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

bool prepareSockets(){
  streaming_sock_fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if (streaming_sock_fd < 0) {
    Serial.println("Failed to create udp socket");
    return false;
  }
  return true;
}

bool in_batch_cmd = false;
int parsed_batch_val = 0;

bool reconnect() {
  is_streaming = false;
  batch_index = 0;
  in_batch_cmd = false;
  parsed_batch_val = 0;
  digitalWrite(stream_pin, LOW);

  if (command_sock_fd >= 0) {
    close(command_sock_fd);
    command_sock_fd = -1;
  }
  command_sock_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (command_sock_fd < 0) {
    Serial.println("Failed to create command socket in reconnect");
    delay(500);
    return false;
  }

  if (connect(command_sock_fd, (struct sockaddr*)&remote_addr, sizeof(remote_addr)) < 0) {
    Serial.println("Failed to connect to command socket in reconnect");
    close(command_sock_fd);
    command_sock_fd = -1;
    delay(500);
    return false;
  }

  int flags = fcntl(command_sock_fd, F_GETFL, 0);
  fcntl(command_sock_fd, F_SETFL, flags | O_NONBLOCK);

  Serial.println("Connected");
  return true;
}

void setId(){
  pinMode(config_pin, INPUT);
  delay(10);

  if (digitalRead(config_pin) == LOW) {
    id = 0; 
    Serial.println("ID: 0");
  } else {
    id = 1;  
    Serial.println("ID: 1");
  }
}

bool inputVoltageOk() {
  uint32_t mv = analogReadMilliVolts(bat_pin); 
  float pinVoltage = mv / 1000.0;
  float batteryVoltage = pinVoltage * ((r_plus + r_gnd) / r_gnd);

  if(batteryVoltage > 7.4){
    return true;
  }
  return false;
}

void setupLeds() {
  pinMode(status_pin, OUTPUT);
  pinMode(stream_pin, OUTPUT);

  digitalWrite(status_pin, LOW);
  digitalWrite(stream_pin, LOW);
}

bool prepareSensors() {
  Wire.begin(21, 22); // Jawne piny I2C ESP32 (SDA=21, SCL=22)
  Wire.setClock(400000); 

  if (!lsm6ds3trc.begin_I2C()) {
    Serial.println("LSM6DS3TRC not found!");
    delay(500);
    return false;
  }

  lsm6ds3trc.setAccelRange(LSM6DS_ACCEL_RANGE_8_G);
  lsm6ds3trc.setGyroRange(LSM6DS_GYRO_RANGE_2000_DPS);    

  lsm6ds3trc.setAccelDataRate(LSM6DS_RATE_416_HZ);        
  lsm6ds3trc.setGyroDataRate(LSM6DS_RATE_416_HZ);      

  if (!mmc5603.begin(MMC56X3_DEFAULT_ADDRESS, &Wire)) {
    Serial.println("MMC5603 not found!");
    delay(500);
    return false;
  }

  mmc5603.setDataRate(100); 
  mmc5603.setContinuousMode(true); 

  Serial.println("Sensors prepared");
  return true;
}

void flushBatch() {
  if (batch_index > 0) {
    sendto(streaming_sock_fd, (const uint8_t*)batch_buffer, batch_index * sizeof(SensorData), 0,
           (struct sockaddr*)&remote_addr, sizeof(remote_addr));
    batch_index = 0;
  }
}

void take_action(char c) {
  if (in_batch_cmd) {
    if (c >= '0' && c <= '9') {
      parsed_batch_val = parsed_batch_val * 10 + (c - '0');
    } else if (c == '\n' || c == '\r' || c == ';' || c == ' ') {
      if (parsed_batch_val > 0) {
        flushBatch();
        current_batch_size = constrain(parsed_batch_val, 1, MAX_BATCH_CAPACITY);
        Serial.print("[CMD] Set BATCH_SIZE = ");
        Serial.println(current_batch_size);

        // Opcjonalne potwierdzenie TCP
        if (command_sock_fd >= 0) {
          char ack[32];
          snprintf(ack, sizeof(ack), "ACK_BATCH:%d\n", current_batch_size);
          send(command_sock_fd, ack, strlen(ack), 0);
        }
      }
      in_batch_cmd = false;
      parsed_batch_val = 0;
    } else if ((uint8_t)c >= 1 && (uint8_t)c <= MAX_BATCH_CAPACITY) {
      // Obsługa bezpośredniego bajtu binarnego (np. B\x03)
      flushBatch();
      current_batch_size = (int)(uint8_t)c;
      Serial.print("[CMD] Set BATCH_SIZE (bin) = ");
      Serial.println(current_batch_size);
      in_batch_cmd = false;
      parsed_batch_val = 0;
    } else {
      in_batch_cmd = false;
      parsed_batch_val = 0;
    }
    return;
  }

  if (c == 'B' || c == 'b') {
    in_batch_cmd = true;
    parsed_batch_val = 0;
  } else if (c == start_streaming) {
    is_streaming = true;
    batch_index = 0;
    last_stream_time = millis();
    Serial.println("Start streaming");
    digitalWrite(stream_pin, HIGH);
  } else if (c == stop_streaming) {
    flushBatch(); // Wyślij resztkę bufora przed zatrzymaniem
    Serial.println("Stop streaming");
    digitalWrite(stream_pin, LOW);
    is_streaming = false;
  } else if (c == prepare_cmd) {
    digitalWrite(stream_pin, HIGH);
    delay(50);
    digitalWrite(stream_pin, LOW);
  }
}

void sampleAndBatch() {
  lsm6ds3trc.getEvent(&accel, &gyro, &temp);
  mmc5603.getEvent(&magn);
  setData();

  batch_buffer[batch_index++] = data;
  if (batch_index >= current_batch_size) {
    flushBatch();
  }
}

void setup() {
  Serial.begin(115200);
  unsigned long serial_wait = millis();
  while (!Serial && (millis() - serial_wait < 2000));

  setId();
  connectWiFi();
  setupLeds();
  while(!prepareSensors());
  while(!resolveRemote(&remote_addr));
  while(!prepareSockets());
  while(!reconnect());
  digitalWrite(status_pin, HIGH);
}

void loop() {
  unsigned long now = millis();

  // Sprawdzanie napięcia baterii co 500 ms (zamiast blokowania ADC w każdym obiegu pętli)
  if (now - lastBatCheck >= 500) {
    bat_status = inputVoltageOk();
    lastBatCheck = now;
  }

  if (bat_status) {
    digitalWrite(status_pin, HIGH);
  } else {
    if (now - lastBlink > 300) {
      bat_state = !bat_state;
      digitalWrite(status_pin, bat_state);
      lastBlink = now;
    }
  }

  // Odbiór komend TCP (nieblokujący)
  ssize_t n = recv(command_sock_fd, buffer, sizeof(buffer) - 1, 0);
  if (n > 0) {
    for (ssize_t i = 0; i < n; i++) {
      take_action(buffer[i]);
    }
  } else if (n == 0 || (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK)) {
    reconnect();
    return;
  }

  // Stałe próbkowanie 100 Hz z wysyłką paczek (batch) po BATCH_SIZE próbek
  if (is_streaming) {
    now = millis();
    if (now - last_stream_time >= stream_interval) {
      last_stream_time += stream_interval;
      if (now - last_stream_time > stream_interval * 2) {
        last_stream_time = now; // Zabezpieczenie przed kumulacją opóźnienia
      }
      sampleAndBatch();
    }
  }
}
