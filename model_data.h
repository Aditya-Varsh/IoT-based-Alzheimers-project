#ifndef MODEL_DATA_H
#define MODEL_DATA_H

// Model configuration
const int NUM_FEATURES = 27;
const int NUM_TREES = 3;

// Feature scaling parameters
const float SCALER_MEAN[NUM_FEATURES] = {
    7.484342f, 32.675752f, 65.677632f, -50.131579f, -261.136711f, 49.417617f, -189.328947f, -368.526316f, -23.058553f, 36.219352f, 43.592105f, -94.651316f, -51.019868f, 262.115293f, 417.644737f, -561.493421f, 76.399079f, 319.116156f, 592.572368f, -467.236842f, 2.974868f, 329.572239f, 524.263158f, -530.217105f, 269.831798f, 48.271694f, 377.888676f
};

const float SCALER_SCALE[NUM_FEATURES] = {
    18.840576f, 15.244630f, 33.371238f, 34.330569f, 25.602593f, 25.571457f, 30.138011f, 75.780496f, 28.157520f, 18.308420f, 39.672235f, 50.957575f, 107.128729f, 124.755049f, 254.448563f, 331.722954f, 374.508650f, 144.156720f, 401.131928f, 421.187243f, 291.549968f, 170.639956f, 373.511409f, 438.111496f, 29.806304f, 24.469810f, 80.670121f
};

// Simple threshold-based prediction (simplified for demo)
// You can replace this with the full tree implementation later
bool predict_fall_simple(float* features) {
    // Normalize features
    float normalized_features[NUM_FEATURES];
    for (int i = 0; i < NUM_FEATURES; i++) {
        normalized_features[i] = (features[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
    }
    
    // Simple thresholding (replace with actual model logic)
    // This is based on acceleration magnitude features
    float acc_mag_mean = normalized_features[24];  // acc_magnitude_mean
    float acc_mag_std = normalized_features[25];   // acc_magnitude_std
    float acc_mag_max = normalized_features[26];   // acc_magnitude_max
    
    // Fall detection logic (adjust thresholds based on your data)
    if (acc_mag_max > 1.5 || acc_mag_std > 1.0) {
        return true;  // Potential fall detected
    }
    
    return false;
}

#endif // MODEL_DATA_H
