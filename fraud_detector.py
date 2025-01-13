import os
import pandas as pd
import numpy as np
from datetime import datetime
import PyPDF2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class BankStatementReader:
    def __init__(self, statements_dir):
        self.statements_dir = statements_dir
        self.supported_extensions = {'.pdf', '.csv', '.xlsx', '.xls', '.xlsb'}
    
    def read_pdf(self, file_path):
        # Basic PDF reading implementation
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            print(f"Error reading PDF {file_path}: {str(e)}")
            return None

    def read_excel(self, file_path):
        try:
            if file_path.endswith('.xlsb'):
                df = pd.read_excel(file_path, engine='pyxlsb')
            else:
                df = pd.read_excel(file_path)
            return df
        except Exception as e:
            print(f"Error reading Excel {file_path}: {str(e)}")
            return None

    def read_csv(self, file_path):
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            print(f"Error reading CSV {file_path}: {str(e)}")
            return None

class FraudDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
    
    def extract_features(self, transactions_df):
        features = []
        
        # Basic statistical features
        features.extend([
            transactions_df['amount'].mean(),
            transactions_df['amount'].std(),
            transactions_df['amount'].max(),
            transactions_df['amount'].min(),
            len(transactions_df),
            transactions_df['amount'].sum()
        ])
        
        # Transaction frequency features
        if 'date' in transactions_df.columns:
            transactions_df['date'] = pd.to_datetime(transactions_df['date'])
            date_diffs = transactions_df['date'].diff().dt.total_seconds()
            features.extend([
                date_diffs.mean(),
                date_diffs.std(),
                date_diffs.max(),
                date_diffs.min()
            ])
        
        return np.array(features)

    def build_model(self, input_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        
        return model

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build and train model
        self.model = self.build_model(X_train.shape[1])
        self.model.fit(X_train_scaled, y_train,
                      epochs=50,
                      batch_size=32,
                      validation_data=(X_test_scaled, y_test),
                      verbose=1)
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

def main():
    # Initialize readers and detector
    statements_dir = "Bank Statements"
    reader = BankStatementReader(statements_dir)
    detector = FraudDetector()
    
    # Process files and extract features
    all_features = []
    for filename in os.listdir(statements_dir):
        file_path = os.path.join(statements_dir, filename)
        _, ext = os.path.splitext(filename)
        
        if ext.lower() not in reader.supported_extensions:
            continue
            
        print(f"Processing {filename}...")
        
        # Read file based on extension
        if ext.lower() == '.pdf':
            data = reader.read_pdf(file_path)
            # TODO: Implement PDF text parsing to extract transaction data
            continue
        elif ext.lower() in ['.xlsx', '.xls', '.xlsb']:
            data = reader.read_excel(file_path)
        elif ext.lower() == '.csv':
            data = reader.read_csv(file_path)
        
        if data is not None:
            # Extract features
            try:
                features = detector.extract_features(data)
                all_features.append(features)
            except Exception as e:
                print(f"Error extracting features from {filename}: {str(e)}")

    if all_features:
        # Convert to numpy array
        X = np.array(all_features)
        
        # For demonstration, generate random labels (0: normal, 1: fraud)
        # In real application, you would need labeled data
        y = np.random.randint(0, 2, size=len(all_features))
        
        # Train the model
        detector.train(X, y)
        
        # Make predictions
        predictions = detector.predict(X)
        
        # Print results
        for filename, pred in zip(os.listdir(statements_dir), predictions):
            if os.path.splitext(filename)[1].lower() in reader.supported_extensions:
                print(f"{filename}: {'Potential Fraud' if pred > 0.5 else 'Normal'} (Score: {pred[0]:.2f})")

if __name__ == "__main__":
    main() 