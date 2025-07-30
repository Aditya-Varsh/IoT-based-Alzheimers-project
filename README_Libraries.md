# Arduino Libraries Required for Fall Detection ESP32

## Required Libraries:
1. **Wire** - Built-in Arduino library for I2C communication
2. **MPU6050** - Library for MPU6050 sensor

## Installation Instructions:

### Method 1: Arduino IDE Library Manager
1. Open Arduino IDE
2. Go to Tools > Manage Libraries...
3. Search for "MPU6050" and install "MPU6050 by Electronic Cats" or "Adafruit MPU6050"
4. Wire library is built-in, no installation needed

### Method 2: Manual Installation
1. Download MPU6050 library from: https://github.com/ElectronicCats/mpu6050
2. Extract to Arduino/libraries/ folder

### Method 3: PlatformIO (Recommended)
The platformio.ini file is already configured with the required libraries.
Just open this project in PlatformIO and it will auto-install dependencies.

## Board Configuration:
- Board: ESP32 Dev Module
- Upload Speed: 921600
- CPU Frequency: 240MHz
- Flash Frequency: 80MHz
- Core Debug Level: None
