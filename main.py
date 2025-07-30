import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import os
import glob
from scipy import stats
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

class SisFallPredictor:
    def __init__(self, data_path):
        """
        Initialize the SisFall fall prediction model
        
        Args:
            data_path: Path to the SisFall dataset directory
        """
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.model = None
        self.feature_selector = None
        
    def load_data(self):
        """
        Load and parse SisFall dataset files
        The dataset contains .txt files with accelerometer and gyroscope data
        """
        all_data = []
        
        # Get all .txt files in the dataset directory and subdirectories
        file_pattern = os.path.join(self.data_path, "**", "*.txt")
        files = glob.glob(file_pattern, recursive=True)
        
        print(f"Found {len(files)} files in the dataset")
        
        for file_path in files:
            filename = os.path.basename(file_path)
            
            # Parse filename to extract activity info
            # SisFall format: D01_SA01_R01.txt (Activity_Subject_Trial)
            parts = filename.split('.')[0]
            
            # Skip files that don't follow the expected naming convention
            if len(parts) < 4:
                continue
            
            # Extract activity code (e.g., D01, F01, etc.)
            # Activity code is the first 3 characters (D01, F01, etc.)
            activity_code = parts[:3]
            
            # Determine if it's a fall or ADL
            # F01-F15 are falls, D01-D19 are ADLs
            is_fall = 1 if activity_code.startswith('F') else 0
            
            try:
                # Read the data file - SisFall uses comma separation with semicolon line endings
                data = pd.read_csv(file_path, header=None, delimiter=',')
                
                # Remove the semicolon from the last column if present
                if data.shape[1] > 0:
                    last_col = data.iloc[:, -1].astype(str).str.replace(';', '', regex=False)
                    data.iloc[:, -1] = pd.to_numeric(last_col, errors='coerce')
                
                # SisFall has 9 columns: typically 3 accel + 3 gyro + 3 additional sensors
                # We'll use the first 6 columns as the main sensor data
                if data.shape[1] >= 6:
                    # Use first 6 columns as accelerometer and gyroscope data
                    sensor_data = data.iloc[:, :6].copy()
                    sensor_data.columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
                    data = sensor_data
                else:
                    print(f"Warning: {filename} has only {data.shape[1]} columns, expected at least 6")
                    continue
                
                # Add metadata
                data['activity_code'] = activity_code
                data['is_fall'] = is_fall
                data['file'] = filename
                
                all_data.append(data)
                
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
        
        # Check if any data was loaded
        if not all_data:  # Check if list is empty
            print("No data files found. Please check your dataset path and file formats.")
            self.raw_data = pd.DataFrame()
            return self.raw_data
        
        # Combine all data
        self.raw_data = pd.concat(all_data, ignore_index=True)
        print(f"Loaded {len(self.raw_data)} samples")
        print(f"Falls: {self.raw_data['is_fall'].sum()}")
        print(f"ADLs: {len(self.raw_data) - self.raw_data['is_fall'].sum()}")
        
        return self.raw_data
    
    def preprocess_data(self, window_size=200, overlap=100):
        """
        Preprocess the data using sliding window approach
        
        Args:
            window_size: Size of the sliding window
            overlap: Overlap between windows
        """
        # Check if we have any data to process
        if self.raw_data is None or self.raw_data.empty:
            print("No data available for preprocessing. Load data first.")
            self.processed_data = pd.DataFrame()
            return self.processed_data
            
        processed_data = []
        
        # Group by file to process each activity separately
        for file_group in self.raw_data.groupby('file'):
            file_data = file_group[1]
            
            # Apply low-pass filter to reduce noise
            sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
            filtered_data = file_data[sensor_cols].copy()
            
            # Butterworth low-pass filter
            b, a = butter(3, 0.3, btype='low')
            for col in sensor_cols:
                filtered_data[col] = filtfilt(b, a, filtered_data[col])
            
            # Sliding window extraction
            step = window_size - overlap
            for i in range(0, len(filtered_data) - window_size + 1, step):
                window = filtered_data.iloc[i:i+window_size]
                
                if len(window) == window_size:
                    # Extract features for this window
                    features = self.extract_features(window)
                    features['is_fall'] = file_data['is_fall'].iloc[0]
                    features['activity_code'] = file_data['activity_code'].iloc[0]
                    
                    processed_data.append(features)
        
        self.processed_data = pd.DataFrame(processed_data)
        print(f"Created {len(self.processed_data)} windows")
        
        return self.processed_data
    
    def extract_features(self, window):
        """
        Extract statistical and domain-specific features from a window
        
        Args:
            window: DataFrame with sensor data for a time window
        """
        features = {}
        
        # Statistical features for each sensor
        for col in ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']:
            features[f'{col}_mean'] = window[col].mean()
            features[f'{col}_std'] = window[col].std()
            features[f'{col}_min'] = window[col].min()
            features[f'{col}_max'] = window[col].max()
            features[f'{col}_range'] = features[f'{col}_max'] - features[f'{col}_min']
            features[f'{col}_skew'] = stats.skew(window[col])
            features[f'{col}_kurtosis'] = stats.kurtosis(window[col])
            features[f'{col}_rms'] = np.sqrt(np.mean(window[col]**2))
        
        # Magnitude features
        acc_magnitude = np.sqrt(window['acc_x']**2 + window['acc_y']**2 + window['acc_z']**2)
        gyro_magnitude = np.sqrt(window['gyro_x']**2 + window['gyro_y']**2 + window['gyro_z']**2)
        
        features['acc_magnitude_mean'] = acc_magnitude.mean()
        features['acc_magnitude_std'] = acc_magnitude.std()
        features['acc_magnitude_max'] = acc_magnitude.max()
        features['acc_magnitude_min'] = acc_magnitude.min()
        
        features['gyro_magnitude_mean'] = gyro_magnitude.mean()
        features['gyro_magnitude_std'] = gyro_magnitude.std()
        features['gyro_magnitude_max'] = gyro_magnitude.max()
        features['gyro_magnitude_min'] = gyro_magnitude.min()
        
        # Signal Vector Magnitude (SVM)
        svm = acc_magnitude - 1.0  # Subtract gravity
        features['svm_mean'] = svm.mean()
        features['svm_std'] = svm.std()
        features['svm_max'] = svm.max()
        
        # Tilt angle features
        features['tilt_mean'] = np.mean(np.arctan2(window['acc_y'], window['acc_z']))
        features['tilt_std'] = np.std(np.arctan2(window['acc_y'], window['acc_z']))
        
        # Correlation features
        features['acc_x_y_corr'] = window['acc_x'].corr(window['acc_y'])
        features['acc_x_z_corr'] = window['acc_x'].corr(window['acc_z'])
        features['acc_y_z_corr'] = window['acc_y'].corr(window['acc_z'])
        
        # Fill NaN values with 0 (in case of constant signals)
        for key in features:
            if pd.isna(features[key]):
                features[key] = 0
        
        return features
    
    def train_model(self, test_size=0.2, random_state=42):
        """
        Train the fall prediction model
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        # Prepare features and labels
        feature_cols = [col for col in self.processed_data.columns 
                       if col not in ['is_fall', 'activity_code']]
        
        X = self.processed_data[feature_cols]
        y = self.processed_data['is_fall']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection
        self.feature_selector = SelectKBest(score_func=f_classif, k=20)
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        # Train Random Forest model with adjusted parameters for better fall detection
        self.model = RandomForestClassifier(
            n_estimators=200,  # Increased from 100
            max_depth=15,      # Increased from 10
            random_state=random_state,
            class_weight={0: 1, 1: 2}  # Give more weight to fall class (1) to improve recall
        )
        
        self.model.fit(X_train_selected, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test_selected)
        
        print("Model Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['ADL', 'Fall']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['ADL', 'Fall'], yticklabels=['ADL', 'Fall'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
        
        # Feature importance
        feature_names = [feature_cols[i] for i in self.feature_selector.get_support(indices=True)]
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df.head(15), x='importance', y='feature')
        plt.title('Top 15 Feature Importances')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
        
        return self.model
    
    def predict(self, new_data):
        """
        Predict fall for new sensor data
        
        Args:
            new_data: DataFrame with sensor readings
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Extract features from new data
        features = self.extract_features(new_data)
        
        # Convert to DataFrame
        feature_df = pd.DataFrame([features])
        
        # Scale features
        feature_scaled = self.scaler.transform(feature_df)
        
        # Select features
        feature_selected = self.feature_selector.transform(feature_scaled)
        
        # Predict
        prediction = self.model.predict(feature_selected)[0]
        probability = self.model.predict_proba(feature_selected)[0]
        
        return {
            'prediction': 'Fall' if prediction == 1 else 'ADL',
            'fall_probability': probability[1],
            'confidence': max(probability)
        }

# Example usage
def main():
    # Initialize the predictor
    # Replace with your actual path to SisFall dataset
    data_path = r"Sisfall Dataset\SisFall_dataset"  # Windows example
    # data_path = "/home/user/sisfall/dataset"      # Linux/Mac example
    
    predictor = SisFallPredictor(data_path)
    
    # Load and preprocess data
    print("Loading data...")
    raw_data = predictor.load_data()
    
    # Only proceed if data was loaded successfully
    if raw_data.empty:
        print("No data loaded. Exiting...")
        return
    
    print("Preprocessing data...")
    processed_data = predictor.preprocess_data()
    
    # Only proceed if preprocessing was successful
    if processed_data.empty:
        print("No processed data available. Exiting...")
        return
    
    # Train the model
    print("Training model...")
    model = predictor.train_model()
    
    # Example prediction on new data
    # Create sample data (replace with actual sensor readings)
    sample_data = pd.DataFrame({
        'acc_x': np.random.randn(200),
        'acc_y': np.random.randn(200),
        'acc_z': np.random.randn(200) + 1,  # Add gravity
        'gyro_x': np.random.randn(200) * 0.1,
        'gyro_y': np.random.randn(200) * 0.1,
        'gyro_z': np.random.randn(200) * 0.1
    })
    
    result = predictor.predict(sample_data)
    print(f"\nPrediction Result: {result}")

if __name__ == "__main__":
    main()