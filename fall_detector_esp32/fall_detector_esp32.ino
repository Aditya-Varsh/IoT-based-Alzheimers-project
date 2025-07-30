#include <Arduino.h>
#include <Wire.h>
#include <MPU6050.h>
#include <math.h>

// Hardware pins
const int BUZZER_PIN = 25;
const int LED_PIN = 2;
const int BUTTON_PIN = 0;

// Model parameters
const int WINDOW_SIZE = 50;
const int NUM_FEATURES = 30;  // Adjust based on your model
const float FALL_THRESHOLD = 0.5;  // Adjust based on testing

// IMU sensor
MPU6050 mpu;

// Data buffers
float acc_x_buffer[WINDOW_SIZE];
float acc_y_buffer[WINDOW_SIZE];
float acc_z_buffer[WINDOW_SIZE];
float gyro_x_buffer[WINDOW_SIZE];
float gyro_y_buffer[WINDOW_SIZE];
float gyro_z_buffer[WINDOW_SIZE];

int buffer_index = 0;
bool buffer_full = false;
unsigned long last_sample_time = 0;
bool fall_detected = false;
unsigned long fall_alert_time = 0;

// Calibration variables
float acc_offset_x = 0, acc_offset_y = 0, acc_offset_z = 0;
float gyro_offset_x = 0, gyro_offset_y = 0, gyro_offset_z = 0;

// Fall confirmation variables
bool awaiting_confirmation = false;
unsigned long confirmation_start_time = 0;
const unsigned long CONFIRMATION_TIMEOUT = 10000; // 10 seconds
const int YES_BUTTON_PIN = 4;  // Add a YES button
const int NO_BUTTON_PIN = 5;   // Add a NO button (existing button can be used as NO)

void setup() {
    Serial.begin(115200);
    Wire.begin();
    
    // Initialize pins
    pinMode(BUZZER_PIN, OUTPUT);
    pinMode(LED_PIN, OUTPUT);
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    
    // Initialize MPU6050
    Serial.println("Initializing MPU6050...");
    mpu.initialize();
    
    if (!mpu.testConnection()) {
        Serial.println("MPU6050 connection failed!");
        while(1) {
            digitalWrite(LED_PIN, HIGH);
            delay(200);
            digitalWrite(LED_PIN, LOW);
            delay(200);
        }
    }
    
    Serial.println("MPU6050 connected successfully!");
    
    // Configure sensor
    mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_2);  // ±2g
    mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_250);  // ±250°/s
    mpu.setDLPFMode(6);  // Low pass filter
    
    // Calibrate sensor
    calibrateSensor();
    
    // Initialize buffers
    for (int i = 0; i < WINDOW_SIZE; i++) {
        acc_x_buffer[i] = 0;
        acc_y_buffer[i] = 0;
        acc_z_buffer[i] = 0;
        gyro_x_buffer[i] = 0;
        gyro_y_buffer[i] = 0;
        gyro_z_buffer[i] = 0;
    }
    
    Serial.println("Fall Detection System Ready!");
    digitalWrite(LED_PIN, HIGH);
    delay(1000);
    digitalWrite(LED_PIN, LOW);
}

void loop() {
    unsigned long current_time = millis();
    
    // Sample at 50Hz (20ms intervals)
    if (current_time - last_sample_time >= 20) {
        last_sample_time = current_time;
        
        // Read sensor data
        readSensorData();
        
        // Process when buffer is full
        if (buffer_full) {
            float features[NUM_FEATURES];
            extractFeatures(features);
            
            // Simple fall detection algorithm
            bool is_fall = detectFall(features);
            
            if (is_fall && !fall_detected) {
                fall_detected = true;
                fall_alert_time = current_time;
                triggerFallAlert();
                Serial.println("FALL DETECTED!");
            }
        }
    }
    
    // Reset fall detection after 5 seconds
    if (fall_detected && (current_time - fall_alert_time > 5000)) {
        fall_detected = false;
        Serial.println("Fall alert reset");
    }
    
    // Check for manual reset button
    if (digitalRead(BUTTON_PIN) == LOW) {
        fall_detected = false;
        digitalWrite(BUZZER_PIN, LOW);
        Serial.println("Manual reset");
        delay(1000);  // Debounce
    }
}

void readSensorData() {
    int16_t ax, ay, az, gx, gy, gz;
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    
    // Convert to g and deg/s, apply calibration
    float acc_x = (ax / 16384.0) - acc_offset_x;
    float acc_y = (ay / 16384.0) - acc_offset_y;
    float acc_z = (az / 16384.0) - acc_offset_z;
    float gyro_x = (gx / 131.0) - gyro_offset_x;
    float gyro_y = (gy / 131.0) - gyro_offset_y;
    float gyro_z = (gz / 131.0) - gyro_offset_z;
    
    // Add to circular buffer
    acc_x_buffer[buffer_index] = acc_x;
    acc_y_buffer[buffer_index] = acc_y;
    acc_z_buffer[buffer_index] = acc_z;
    gyro_x_buffer[buffer_index] = gyro_x;
    gyro_y_buffer[buffer_index] = gyro_y;
    gyro_z_buffer[buffer_index] = gyro_z;
    
    buffer_index++;
    if (buffer_index >= WINDOW_SIZE) {
        buffer_index = 0;
        buffer_full = true;
    }
}

void calibrateSensor() {
    Serial.println("Calibrating sensor... Keep device still!");
    
    const int cal_samples = 100;
    float sum_ax = 0, sum_ay = 0, sum_az = 0;
    float sum_gx = 0, sum_gy = 0, sum_gz = 0;
    
    for (int i = 0; i < cal_samples; i++) {
        int16_t ax, ay, az, gx, gy, gz;
        mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
        
        sum_ax += ax / 16384.0;
        sum_ay += ay / 16384.0;
        sum_az += az / 16384.0;
        sum_gx += gx / 131.0;
        sum_gy += gy / 131.0;
        sum_gz += gz / 131.0;
        
        delay(10);
    }
    
    acc_offset_x = sum_ax / cal_samples;
    acc_offset_y = sum_ay / cal_samples;
    acc_offset_z = (sum_az / cal_samples) - 1.0;  // Subtract gravity
    gyro_offset_x = sum_gx / cal_samples;
    gyro_offset_y = sum_gy / cal_samples;
    gyro_offset_z = sum_gz / cal_samples;
    
    Serial.println("Calibration complete!");
}

void extractFeatures(float* features) {
    int idx = 0;
    
    // Statistical features for each sensor
    features[idx++] = calculateMean(acc_x_buffer, WINDOW_SIZE);
    features[idx++] = calculateStd(acc_x_buffer, WINDOW_SIZE);
    features[idx++] = calculateMax(acc_x_buffer, WINDOW_SIZE);
    features[idx++] = calculateMin(acc_x_buffer, WINDOW_SIZE);
    
    features[idx++] = calculateMean(acc_y_buffer, WINDOW_SIZE);
    features[idx++] = calculateStd(acc_y_buffer, WINDOW_SIZE);
    features[idx++] = calculateMax(acc_y_buffer, WINDOW_SIZE);
    features[idx++] = calculateMin(acc_y_buffer, WINDOW_SIZE);
    
    features[idx++] = calculateMean(acc_z_buffer, WINDOW_SIZE);
    features[idx++] = calculateStd(acc_z_buffer, WINDOW_SIZE);
    features[idx++] = calculateMax(acc_z_buffer, WINDOW_SIZE);
    features[idx++] = calculateMin(acc_z_buffer, WINDOW_SIZE);
    
    features[idx++] = calculateMean(gyro_x_buffer, WINDOW_SIZE);
    features[idx++] = calculateStd(gyro_x_buffer, WINDOW_SIZE);
    features[idx++] = calculateMax(gyro_x_buffer, WINDOW_SIZE);
    features[idx++] = calculateMin(gyro_x_buffer, WINDOW_SIZE);
    
    features[idx++] = calculateMean(gyro_y_buffer, WINDOW_SIZE);
    features[idx++] = calculateStd(gyro_y_buffer, WINDOW_SIZE);
    features[idx++] = calculateMax(gyro_y_buffer, WINDOW_SIZE);
    features[idx++] = calculateMin(gyro_y_buffer, WINDOW_SIZE);
    
    features[idx++] = calculateMean(gyro_z_buffer, WINDOW_SIZE);
    features[idx++] = calculateStd(gyro_z_buffer, WINDOW_SIZE);
    features[idx++] = calculateMax(gyro_z_buffer, WINDOW_SIZE);
    features[idx++] = calculateMin(gyro_z_buffer, WINDOW_SIZE);
    
    // Magnitude features
    float acc_magnitude[WINDOW_SIZE];
    for (int i = 0; i < WINDOW_SIZE; i++) {
        acc_magnitude[i] = sqrt(acc_x_buffer[i]*acc_x_buffer[i] + 
                               acc_y_buffer[i]*acc_y_buffer[i] + 
                               acc_z_buffer[i]*acc_z_buffer[i]);
    }
    
    features[idx++] = calculateMean(acc_magnitude, WINDOW_SIZE);
    features[idx++] = calculateStd(acc_magnitude, WINDOW_SIZE);
    features[idx++] = calculateMax(acc_magnitude, WINDOW_SIZE);
    
    // SVM features (subtract gravity)
    for (int i = 0; i < WINDOW_SIZE; i++) {
        acc_magnitude[i] -= 1.0;
    }
    features[idx++] = calculateMean(acc_magnitude, WINDOW_SIZE);
    features[idx++] = calculateStd(acc_magnitude, WINDOW_SIZE);
    features[idx++] = calculateMax(acc_magnitude, WINDOW_SIZE);
}

bool detectFall(float* features) {
    // Simplified fall detection algorithm
    // You can replace this with your trained model
    
    float acc_magnitude_max = features[26];  // Index of acc_magnitude_max
    float acc_magnitude_std = features[25];  // Index of acc_magnitude_std
    float svm_max = features[29];            // Index of svm_max
    
    // Simple threshold-based detection
    // Falls typically have high acceleration magnitude and variation
    if (acc_magnitude_max > 2.5 && acc_magnitude_std > 0.5) {
        return true;
    }
    
    // Check for sudden impact (high SVM)
    if (abs(svm_max) > 2.0) {
        return true;
    }
    
    return false;
}

void triggerFallAlert() {
    // Sound buzzer pattern
    for (int i = 0; i < 20; i++) {
        digitalWrite(BUZZER_PIN, HIGH);
        digitalWrite(LED_PIN, HIGH);
        delay(100);
        digitalWrite(BUZZER_PIN, LOW);
        digitalWrite(LED_PIN, LOW);
        delay(100);
    }
    
    // Keep LED on to indicate fall state
    digitalWrite(LED_PIN, HIGH);
}

// Helper functions for feature calculation
float calculateMean(float* data, int size) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum / size;
}

float calculateStd(float* data, int size) {
    float mean = calculateMean(data, size);
    float sum = 0;
    for (int i = 0; i < size; i++) {
        float diff = data[i] - mean;
        sum += diff * diff;
    }
    return sqrt(sum / size);
}

float calculateMax(float* data, int size) {
    float max_val = data[0];
    for (int i = 1; i < size; i++) {
        if (data[i] > max_val) max_val = data[i];
    }
    return max_val;
}

float calculateMin(float* data, int size) {
    float min_val = data[0];
    for (int i = 1; i < size; i++) {
        if (data[i] < min_val) min_val = data[i];
    }
    return min_val;
}
