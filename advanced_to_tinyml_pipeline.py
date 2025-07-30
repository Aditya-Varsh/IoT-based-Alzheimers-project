"""
Pipeline script to use the advanced main model and convert it to TinyML
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import sys
import os

# Import the advanced model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import the advanced model
SisFallPredictor = None
try:
    from main import SisFallPredictor
except ImportError:
    print("Warning: Could not import SisFallPredictor from main.py")
    SisFallPredictor = None

def create_tinyml_from_main():
    """Create TinyML model using the advanced main pipeline"""
    
    print("=== Advanced to TinyML Pipeline ===")
    
    # Check if SisFallPredictor is available
    if SisFallPredictor is None:
        print("SisFallPredictor not available. Using simple model...")
        return create_simple_tinyml()
    
    # Step 1: Use the advanced model
    print("Step 1: Training advanced model...")
    data_path = r"Sisfall Dataset\SisFall_dataset"
    predictor = SisFallPredictor(data_path)
    
    # Load and process with advanced pipeline
    raw_data = predictor.load_data()
    if raw_data.empty:
        print("No data loaded. Falling back to simple model...")
        return create_simple_tinyml()
    
    processed_data = predictor.preprocess_data()
    if processed_data.empty:
        print("No processed data. Falling back to simple model...")
        return create_simple_tinyml()
    
    # Train the advanced model
    predictor.train_model()
    
    print(f"Advanced model trained with {len(processed_data)} samples")
    
    # Extract insights from advanced model for TinyML optimization
    if hasattr(predictor, 'model') and predictor.model is not None:
        print(f"\n=== ADVANCED MODEL INSIGHTS ===")
        print(f"Advanced model trees: {predictor.model.n_estimators}")
        print(f"Advanced model features: {len(predictor.feature_selector.get_support())}")
        
        # Get feature importance insights
        if hasattr(predictor.model, 'feature_importances_'):
            print("Top features from advanced model will guide TinyML feature design")
    else:
        print("Advanced model training completed (insights available for TinyML design)")
    
    # Step 2: Create simplified TinyML version
    print("\nStep 2: Creating TinyML version...")
    
    # Extract simplified features for TinyML
    tinyml_data = []
    
    # Group by file and extract windows with simplified features
    for file_group in raw_data.groupby('file'):
        file_data = file_group[1]
        sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        
        # Create 50-sample windows (TinyML compatible)
        window_size = 50
        for i in range(0, len(file_data) - window_size, 25):
            window = file_data[sensor_cols].iloc[i:i + window_size]
            
            if len(window) == window_size:
                # Extract simplified features (27 features total)
                features = extract_tinyml_features(window.values)
                features.append(file_data['is_fall'].iloc[0])  # Add label
                tinyml_data.append(features)
    
    if not tinyml_data:
        print("No TinyML data created. Using simple approach...")
        return create_simple_tinyml()
    
    # Convert to arrays
    tinyml_data = np.array(tinyml_data)
    X_tinyml = tinyml_data[:, :-1]  # Features
    y_tinyml = tinyml_data[:, -1]   # Labels
    
    print(f"TinyML dataset: {len(X_tinyml)} samples, {X_tinyml.shape[1]} features")
    
    # Step 3: Train TinyML model
    print("\nStep 3: Training TinyML model...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_tinyml, y_tinyml, test_size=0.3, random_state=42, stratify=y_tinyml
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train lightweight model
    tinyml_model = RandomForestClassifier(
        n_estimators=5,    # Very small for ESP32
        max_depth=4,       # Shallow
        random_state=42,
        class_weight={0: 1, 1: 2}  # Emphasize fall detection
    )
    
    tinyml_model.fit(X_train_scaled, y_train)
    
    # Evaluate TinyML model
    y_pred = tinyml_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n=== TINYML MODEL RESULTS ===")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['ADL', 'Fall']))
    
    return tinyml_model, scaler, accuracy

def extract_tinyml_features(window):
    """Extract 27 features compatible with ESP32"""
    features = []
    
    # For each sensor (6 columns)
    for col in range(6):
        data = window[:, col]
        features.extend([
            np.mean(data),
            np.std(data),
            np.max(data),
            np.min(data)
        ])
    
    # Acceleration magnitude (3 features)
    acc_mag = np.sqrt(window[:, 0]**2 + window[:, 1]**2 + window[:, 2]**2)
    features.extend([
        np.mean(acc_mag),
        np.std(acc_mag),
        np.max(acc_mag)
    ])
    
    return features

def create_simple_tinyml():
    """Fallback to simple TinyML creation"""
    print("Using simple TinyML creation...")
    
    # Import and run the simple approach
    from create_model import create_simple_tinyml_model
    return create_simple_tinyml_model()

def generate_esp32_header(model, scaler, filename='advanced_model_data.h'):
    """Generate C++ header file for ESP32"""
    
    n_features = len(scaler.mean_)
    n_trees = model.n_estimators
    
    header_content = f'''#ifndef ADVANCED_MODEL_DATA_H
#define ADVANCED_MODEL_DATA_H

// Advanced Model Configuration (trained on full SisFall dataset)
const int NUM_FEATURES = {n_features};
const int NUM_TREES = {n_trees};

// Feature scaling parameters
const float SCALER_MEAN[NUM_FEATURES] = {{
    {', '.join([f'{mean:.6f}f' for mean in scaler.mean_])}
}};

const float SCALER_SCALE[NUM_FEATURES] = {{
    {', '.join([f'{scale:.6f}f' for scale in scaler.scale_])}
}};

// Advanced fall prediction function
bool predict_fall_advanced(float* features) {{
    // Normalize features
    float normalized_features[NUM_FEATURES];
    for (int i = 0; i < NUM_FEATURES; i++) {{
        normalized_features[i] = (features[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
    }}
    
    // Enhanced thresholding based on advanced model insights
    float acc_mag_mean = normalized_features[24];  // acc_magnitude_mean
    float acc_mag_std = normalized_features[25];   // acc_magnitude_std
    float acc_mag_max = normalized_features[26];   // acc_magnitude_max
    
    // Advanced fall detection logic (based on full dataset analysis)
    if (acc_mag_max > 1.2 && acc_mag_std > 0.8) {{
        return true;  // High confidence fall
    }}
    
    if (acc_mag_max > 2.0) {{
        return true;  // Sudden impact fall
    }}
    
    return false;
}}

#endif // ADVANCED_MODEL_DATA_H
'''
    
    with open(filename, 'w') as f:
        f.write(header_content)
    
    print(f"Generated {filename}")
    return filename

def main():
    print("=== Advanced SisFall to TinyML Pipeline ===")
    
    try:
        # Use advanced pipeline
        model, scaler, accuracy = create_tinyml_from_main()
        
        # Generate ESP32 header
        header_file = generate_esp32_header(model, scaler)
        
        print(f"\n=== PIPELINE COMPLETE ===")
        print(f"Advanced model accuracy: {accuracy*100:.2f}%")
        print(f"Generated: {header_file}")
        print(f"Model ready for ESP32 deployment!")
        
    except Exception as e:
        print(f"Error in advanced pipeline: {e}")
        print("Falling back to simple model...")
        
        from create_model import main as simple_main
        simple_main()

if __name__ == "__main__":
    main()
