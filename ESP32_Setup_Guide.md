# ESP32 Fall Detection System Setup Guide

## Hardware Requirements

### Components Needed:
1. **ESP32 Development Board** (ESP32-WROOM-32 or similar)
2. **MPU6050 6-axis IMU Sensor** (Accelerometer + Gyroscope)
3. **Buzzer** (Active buzzer recommended)
4. **Jumper Wires**
5. **Breadboard** (optional)
6. **3.7V LiPo Battery** (for wearable operation)

## Wiring Connections

### MPU6050 to ESP32:
```
MPU6050    ESP32
VCC    ->  3.3V
GND    ->  GND
SCL    ->  GPIO 22 (I2C Clock)
SDA    ->  GPIO 21 (I2C Data)
```

### Buzzer to ESP32:
```
Buzzer     ESP32
VCC    ->  3.3V (or 5V if buzzer requires it)
GND    ->  GND
Signal ->  GPIO 25
```

### Optional Components:
```
LED (Status)   ->  GPIO 2
Button (Reset) ->  GPIO 0
```

## ESP32 Code Libraries Required

### Arduino IDE Libraries:
1. **MPU6050 Library** by Electronic Cats
2. **ArduinoJson** by Benoit Blanchon
3. **WiFi** (built-in)

### Installation Steps:
1. Open Arduino IDE
2. Go to Tools -> Manage Libraries
3. Search and install:
   - "MPU6050" by Electronic Cats
   - "ArduinoJson"

## Code Features

### Real-time Processing:
- **50Hz sampling rate** (20ms intervals)
- **50-sample sliding window** (1 second of data)
- **Lightweight feature extraction**
- **Random Forest prediction**

### Alert System:
- **Buzzer patterns** for different alert types
- **Serial output** for debugging
- **WiFi notifications** (optional)
- **LED indicators** (optional)

## Power Consumption Optimization

### Sleep Modes:
```cpp
// Light sleep when no motion detected
esp_sleep_enable_timer_wakeup(100000); // 100ms
esp_light_sleep_start();
```

### Sensor Configuration:
```cpp
// Configure MPU6050 for low power
mpu.setDLPFMode(6);           // Low pass filter
mpu.setDHPFMode(0);           // High pass filter
mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_2);
mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_250);
```

## Wearable Placement

### Recommended Positions:
1. **Chest/Upper Torso** - Best for fall detection
2. **Waist/Belt** - Good compromise
3. **Wrist** - Less accurate but convenient

### Orientation:
- **X-axis**: Forward/Backward
- **Y-axis**: Left/Right  
- **Z-axis**: Up/Down (gravity direction)

## Testing Procedure

### 1. Basic Functionality:
```
1. Upload code to ESP32
2. Open Serial Monitor (115200 baud)
3. Check sensor readings
4. Verify buzzer operation
```

### 2. Calibration:
```
1. Record normal activities (walking, sitting, standing)
2. Record simulated falls (onto bed/couch safely)
3. Adjust threshold if needed
```

### 3. Real-world Testing:
```
1. Wear device in recommended position
2. Perform daily activities
3. Monitor false positive rate
4. Test with safe controlled falls
```

## Troubleshooting

### Common Issues:

#### MPU6050 Not Detected:
- Check wiring connections
- Verify 3.3V power supply
- Try different I2C pins

#### False Positives:
- Adjust fall detection threshold
- Improve sensor placement
- Add activity context (standing vs sitting)

#### Battery Life:
- Implement sleep modes
- Reduce sampling rate
- Use larger battery

## Safety Considerations

⚠️ **IMPORTANT SAFETY NOTES:**

1. **This is a prototype** - Not for medical use
2. **Test thoroughly** before relying on it
3. **Have backup systems** (manual alert button)
4. **Regular maintenance** (battery, sensor checks)
5. **Consider false alarms** in emergency response

## Future Enhancements

### Possible Improvements:
1. **Machine Learning on device** (TensorFlow Lite)
2. **Multi-sensor fusion** (heart rate, GPS)
3. **Cloud connectivity** (smartphone app)
4. **Adaptive thresholds** (learning user patterns)
5. **Emergency contacts** (SMS/call integration)
