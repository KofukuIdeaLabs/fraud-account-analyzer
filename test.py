# Import required libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from tqdm import tqdm
import time

# Enhanced fraud detection class that implements multiple detection patterns and methods
class EnhancedFraudDetector:
    def __init__(self):
        # Configurable thresholds for different fraud detection patterns
        self.config = {
            # Thresholds for small transfer detection
            'SMALL_TRANSFER_MIN': 2000,
            'SMALL_TRANSFER_MAX': 7000,
            'SIMILAR_TRANSFER_WINDOW': '48h',
            'MIN_SIMILAR_TRANSFERS': 5,
            'QUICK_WITHDRAWAL_WINDOW': '24h',
            
            # Thresholds for structuring detection
            'STRUCTURING_WINDOW': '72h',
            'STRUCTURING_MIN_TRANSACTIONS': 3,
            'STRUCTURING_AMOUNT_THRESHOLD': 50000,
            
            # Thresholds for rapid movement detection
            'RAPID_MOVEMENT_WINDOW': '24h',
            'RAPID_MOVEMENT_MIN_CYCLES': 2,
            
            # Thresholds for timing analysis
            'OFF_HOURS_START': 22,
            'OFF_HOURS_END': 5,
            'WEEKEND_THRESHOLD': 0.7,
            
            # Thresholds for network analysis
            'COMMON_BENEFICIARY_THRESHOLD': 3,
            'NETWORK_ANALYSIS_WINDOW': '7D',
            
            # New configuration parameters for additional rules
            'TRANSACTION_SPIKE_WINDOW': '7D',  # Window to analyze transaction spikes
            'TRANSACTION_SPIKE_THRESHOLD': 2.0,  # Multiple of standard deviation to consider as spike
            'UNUSUAL_SMALL_TXN_THRESHOLD': 100,  # Amount below which to consider as unusual small transaction
            'UNUSUAL_SMALL_TXN_FREQUENCY': 10,  # Number of small transactions to consider suspicious
            'ROUND_TRIP_WINDOW': '72h',  # Window to detect round trip transactions
            'ROUND_TRIP_THRESHOLD': 0.95,  # Percentage match to consider as round trip
            'DORMANCY_PERIOD': '90D',  # Period of inactivity to consider account as dormant
            'ACTIVITY_SPIKE_THRESHOLD': 5,  # Number of transactions to consider as sudden activity
            'HIGH_WITHDRAWAL_THRESHOLD': 0.8,  # Percentage of balance for high withdrawal alert
            'UNIQUE_RECIPIENT_WINDOW': '30D',  # Window to analyze unique recipients
            'UNIQUE_RECIPIENT_THRESHOLD': 10  # Number of unique recipients to consider suspicious
        }
        
        # Risk weights for different fraud patterns
        self.risk_weights = {
            # Weights for new patterns
            'TRANSACTION_SPIKE': 0.7,
            'UNUSUAL_SMALL_TRANSACTIONS': 0.4,
            'ROUND_TRIP_DETECTED': 0.8,
            'DORMANT_ACCOUNT_ACTIVITY': 0.9,
            'HIGH_WITHDRAWAL': 0.8,
            'MULTIPLE_UNIQUE_RECIPIENTS': 0.6,
            
            # Weights for existing patterns
            'SCAM_PATTERN_SMALL_TRANSFERS': 0.8,
            'STRUCTURING_PATTERN': 0.7,
            'RAPID_MONEY_MOVEMENT': 0.6,
            'SUSPICIOUS_TIMING': 0.4,
            'SUSPICIOUS_NETWORK_PATTERN': 0.5
        }

    def standardize_columns(self, df):
        """
        Standardize column names using regex patterns to handle different bank statement formats.
        
        Args:
            df: DataFrame containing transaction data
        Returns:
            DataFrame: Data with standardized column names
        """
        print("\nStandardizing column names...")
        time.sleep(2)
        cols = df.columns
        standardized_df = df.copy()
        
        # Define regex patterns for different column types
        date_pattern = re.compile(r'(trans(action)?[\s_]?)?date|dt|txn[\s_]?d(ate|t)', re.IGNORECASE)
        desc_pattern = re.compile(r'desc(ription)?|narration|particular(s)?|details|transaction|memo|reference|remark(s)?', re.IGNORECASE)
        debit_pattern = re.compile(r'debit(s)?|withdraw(al)?s?(\(dr\))?|paid[\s_]?out|outflow', re.IGNORECASE)
        credit_pattern = re.compile(r'credit(s)?|deposit(s)?(\(cr\))?|paid[\s_]?in|inflow', re.IGNORECASE)
        
        # Find matching columns using regex patterns
        date_col = next((col for col in cols if date_pattern.search(col)), None)
        desc_col = next((col for col in cols if desc_pattern.search(col)), None)
        debit_col = next((col for col in cols if debit_pattern.search(col)), None)
        credit_col = next((col for col in cols if credit_pattern.search(col)), None)
        amount_col = next((col for col in cols if 'amount' in col.lower()), None)
        
        # Standardize column names
        if date_col:
            standardized_df = standardized_df.rename(columns={date_col: 'Date'})
        
        if desc_col:
            standardized_df = standardized_df.rename(columns={desc_col: 'Description'})
        
        # Handle amount columns
        if amount_col and not (debit_col and credit_col):
            # Split single amount column into debit/credit
            standardized_df['Debit'] = standardized_df[amount_col].apply(lambda x: abs(x) if x < 0 else 0)
            standardized_df['Credit'] = standardized_df[amount_col].apply(lambda x: x if x > 0 else 0)
            standardized_df = standardized_df.drop(columns=[amount_col])
        else:
            # Rename existing debit/credit columns
            if debit_col:
                standardized_df = standardized_df.rename(columns={debit_col: 'Debit'})
            if credit_col:
                standardized_df = standardized_df.rename(columns={credit_col: 'Credit'})
        
        # Fill missing values with 0
        if 'Debit' in standardized_df.columns:
            standardized_df['Debit'] = standardized_df['Debit'].fillna(0)
        if 'Credit' in standardized_df.columns:
            standardized_df['Credit'] = standardized_df['Credit'].fillna(0)
        
        print("\nColumn standardization complete.")
        time.sleep(2)
        return standardized_df

    def detect_small_transfer_patterns(self, df):
        """
        Detects pattern of multiple small incoming transfers followed by withdrawals.
        This pattern is typical in scams where multiple victims send small amounts.
        
        Args:
            df: DataFrame containing transaction data
        Returns:
            list: Flags indicating suspicious small transfer patterns
        """
        print("\nDetecting small transfer patterns...")
        time.sleep(2)
        flags = []
        
        # Analyze each transaction
        for idx in tqdm(range(len(df)), desc="Analyzing transfers"):
            row = df.iloc[idx]
            
            # Get transactions within configured time window before current transaction
            window_before = df[
                (df['Date'] >= row['Date'] - pd.Timedelta(self.config['SIMILAR_TRANSFER_WINDOW'])) &
                (df['Date'] <= row['Date'])
            ]
            
            # Look for similar small credits within amount thresholds
            small_credits = window_before[
                (window_before['Credit'] >= self.config['SMALL_TRANSFER_MIN']) &
                (window_before['Credit'] <= self.config['SMALL_TRANSFER_MAX'])
            ]
            
            # Check if number of small transfers exceeds threshold
            if len(small_credits) >= self.config['MIN_SIMILAR_TRANSFERS']:
                # Look for subsequent withdrawals
                window_after = df[
                    (df['Date'] > row['Date']) &
                    (df['Date'] <= row['Date'] + pd.Timedelta(self.config['QUICK_WITHDRAWAL_WINDOW']))
                ]
                
                total_credits = small_credits['Credit'].sum()
                total_subsequent_debits = window_after['Debit'].sum()
                
                # Flag if 80% or more of credited amount is withdrawn quickly
                if total_subsequent_debits >= total_credits * 0.8:
                    flags.append('SCAM_PATTERN_SMALL_TRANSFERS')
                    continue
            
            flags.append(None)
        return flags

    def detect_structuring(self, df):
        """
        Detects structuring patterns where large amounts are broken into smaller ones.
        Common in money laundering and fraud schemes to avoid detection thresholds.
        
        Args:
            df: DataFrame containing transaction data
        Returns:
            list: Flags indicating potential structuring
        """
        print("\nDetecting structuring patterns...")
        time.sleep(2)
        flags = []
        
        # Analyze each transaction
        for idx in tqdm(range(len(df)), desc="Analyzing structuring"):
            row = df.iloc[idx]
            
            # Get transactions within configured structuring window
            window = df[
                (df['Date'] >= row['Date'] - pd.Timedelta(self.config['STRUCTURING_WINDOW'])) &
                (df['Date'] <= row['Date'] + pd.Timedelta(self.config['STRUCTURING_WINDOW']))
            ]
            
            # Check both credits and debits for structuring patterns
            for txn_type in ['Credit', 'Debit']:
                transactions = window[window[txn_type] > 0][txn_type]
                
                # Check if number of transactions meets minimum threshold
                if len(transactions) >= self.config['STRUCTURING_MIN_TRANSACTIONS']:
                    total_amount = transactions.sum()
                    
                    # Check if total amount exceeds structuring threshold
                    if total_amount >= self.config['STRUCTURING_AMOUNT_THRESHOLD']:
                        # Check if transaction amounts are suspiciously similar
                        amount_diffs = np.diff(sorted(transactions))
                        if np.std(amount_diffs) < 1000:  # Transactions of similar size
                            flags.append('STRUCTURING_PATTERN')
                            break
            
            if len(flags) < idx + 1:
                flags.append(None)
        return flags

    def detect_rapid_money_movement(self, df):
        """
        Detects rapid cycles of money moving in and out of account.
        Common in money laundering and fraudulent schemes.
        
        Args:
            df: DataFrame containing transaction data
        Returns:
            list: Flags indicating suspicious rapid money movement
        """
        print("\nDetecting rapid money movement patterns...")
        time.sleep(2)
        flags = []
        
        # Analyze each transaction
        for idx in tqdm(range(len(df)), desc="Analyzing money movement"):
            row = df.iloc[idx]
            
            # Get transactions within rapid movement window
            window = df[
                (df['Date'] >= row['Date']) &
                (df['Date'] <= row['Date'] + pd.Timedelta(self.config['RAPID_MOVEMENT_WINDOW']))
            ]
            
            cycles = 0
            current_cycle = {'in': 0, 'out': 0}
            
            # Track money movement cycles
            for _, txn in window.iterrows():
                # Add credits to cycle
                if txn['Credit'] > 0:
                    current_cycle['in'] += txn['Credit']
                # Add debits to cycle
                elif txn['Debit'] > 0:
                    current_cycle['out'] += txn['Debit']
                    
                    # Check if current cycle is complete (in/out amounts within 20% of each other)
                    if abs(current_cycle['in'] - current_cycle['out']) / max(current_cycle['in'], current_cycle['out']) < 0.2:
                        cycles += 1
                        current_cycle = {'in': 0, 'out': 0}
            
            flags.append('RAPID_MONEY_MOVEMENT' if cycles >= self.config['RAPID_MOVEMENT_MIN_CYCLES'] else None)
        return flags

    def detect_unusual_timing(self, df):
        """
        Detects suspicious transaction timing patterns.
        Focuses on concentrated activity during off-hours or weekends.
        
        Args:
            df: DataFrame containing transaction data
        Returns:
            list: Flags indicating suspicious timing patterns
        """
        print("\nDetecting unusual timing patterns...")
        time.sleep(2)
        flags = []
        
        # Analyze each transaction
        for idx in tqdm(range(len(df)), desc="Analyzing timing"):
            row = df.iloc[idx]
            hour = row['Date'].hour
            is_weekend = row['Date'].weekday() >= 5
            
            # Get transactions within a week window
            window = df[
                (df['Date'] >= row['Date'] - pd.Timedelta('7D')) &
                (df['Date'] <= row['Date'])
            ]
            
            # Calculate percentage of weekend transactions
            weekend_txns = window[window['Date'].dt.weekday >= 5]
            weekend_ratio = len(weekend_txns) / len(window) if len(window) > 0 else 0
            
            # Flag if transaction is during off-hours and there's high weekend activity
            if (hour >= self.config['OFF_HOURS_START'] or hour <= self.config['OFF_HOURS_END']):
                if weekend_ratio >= self.config['WEEKEND_THRESHOLD']:
                    flags.append('SUSPICIOUS_TIMING')
                    continue
            
            flags.append(None)
        return flags

    def detect_network_patterns(self, df):
        """
        Analyzes transaction network patterns to identify suspicious connections between accounts.
        
        Args:
            df: DataFrame containing transaction data
        Returns:
            list: Flags indicating suspicious network patterns
        """
        print("\nDetecting network patterns...")
        time.sleep(2)
        flags = []
        
        # Analyze each transaction
        for idx in tqdm(range(len(df)), desc="Analyzing networks"):
            row = df.iloc[idx]
            
            # Get transactions within network analysis window
            window = df[
                (df['Date'] >= row['Date'] - pd.Timedelta(self.config['NETWORK_ANALYSIS_WINDOW'])) &
                (df['Date'] <= row['Date'])
            ]
            
            # Analyze beneficiary patterns if beneficiary data is available
            if 'Beneficiary' in df.columns:
                # Count unique descriptions per beneficiary
                beneficiary_sources = window.groupby('Beneficiary')['Description'].nunique()
                suspicious_beneficiaries = beneficiary_sources[
                    beneficiary_sources >= self.config['COMMON_BENEFICIARY_THRESHOLD']
                ]
                
                # Flag if current beneficiary is suspicious
                if not suspicious_beneficiaries.empty and row['Beneficiary'] in suspicious_beneficiaries.index:
                    flags.append('SUSPICIOUS_NETWORK_PATTERN')
                    continue
            
            flags.append(None)
        return flags

    # New detection methods
    def detect_transaction_spikes(self, df):
        """
        Detects unusual spikes in transaction volume or values.
        
        Args:
            df: DataFrame containing transaction data
        Returns:
            list: Flags indicating transaction spikes
        """
        print("\nDetecting transaction spikes...")
        flags = []
        
        # Analyze each transaction
        for idx in tqdm(range(len(df)), desc="Analyzing spikes"):
            row = df.iloc[idx]
            
            # Get transactions within spike analysis window
            window = df[
                (df['Date'] >= row['Date'] - pd.Timedelta(self.config['TRANSACTION_SPIKE_WINDOW'])) &
                (df['Date'] < row['Date'])
            ]
            
            # Analyze both transaction volume and values
            if len(window) > 0:
                # Check transaction volume
                daily_counts = window.groupby(window['Date'].dt.date).size()
                avg_daily_count = daily_counts.mean()
                std_daily_count = daily_counts.std()
                
                current_day_count = len(df[df['Date'].dt.date == row['Date'].date()])
                
                # Check transaction values
                avg_value = window['Credit'].mean() + window['Debit'].mean()
                std_value = window['Credit'].std() + window['Debit'].std()
                current_value = row['Credit'] + row['Debit']
                
                # Flag if current activity exceeds thresholds
                if (current_day_count > avg_daily_count + self.config['TRANSACTION_SPIKE_THRESHOLD'] * std_daily_count or
                    current_value > avg_value + self.config['TRANSACTION_SPIKE_THRESHOLD'] * std_value):
                    flags.append('TRANSACTION_SPIKE')
                else:
                    flags.append(None)
            else:
                flags.append(None)
        
        return flags

    def detect_unusual_small_transactions(self, df):
        """
        Detects patterns of unusual small transactions.
        
        Args:
            df: DataFrame containing transaction data
        Returns:
            list: Flags indicating unusual small transaction patterns
        """
        print("\nDetecting unusual small transactions...")
        flags = []
        
        # Analyze each transaction
        for idx in tqdm(range(len(df)), desc="Analyzing small transactions"):
            row = df.iloc[idx]
            
            # Get transactions within 24-hour window
            window = df[
                (df['Date'] >= row['Date'] - pd.Timedelta('24h')) &
                (df['Date'] <= row['Date'])
            ]
            
            # Find small transactions below threshold
            small_txns = window[
                ((window['Credit'] > 0) & (window['Credit'] < self.config['UNUSUAL_SMALL_TXN_THRESHOLD'])) |
                ((window['Debit'] > 0) & (window['Debit'] < self.config['UNUSUAL_SMALL_TXN_THRESHOLD']))
            ]
            
            # Flag if number of small transactions exceeds frequency threshold
            if len(small_txns) >= self.config['UNUSUAL_SMALL_TXN_FREQUENCY']:
                flags.append('UNUSUAL_SMALL_TRANSACTIONS')
            else:
                flags.append(None)
                
        return flags

    def detect_round_trip_transactions(self, df):
        """
        Detects round-trip transactions where money moves out and returns.
        
        Args:
            df: DataFrame containing transaction data
        Returns:
            list: Flags indicating round-trip transactions
        """
        print("\nDetecting round-trip transactions...")
        flags = []
        
        # Analyze each transaction
        for idx in tqdm(range(len(df)), desc="Analyzing round trips"):
            row = df.iloc[idx]
            
            # Only check outgoing transactions
            if row['Debit'] > 0:
                # Look for matching returns within window
                window_after = df[
                    (df['Date'] > row['Date']) &
                    (df['Date'] <= row['Date'] + pd.Timedelta(self.config['ROUND_TRIP_WINDOW']))
                ]
                
                # Check each potential return transaction
                for _, return_tx in window_after.iterrows():
                    if (return_tx['Credit'] > 0 and
                        abs(return_tx['Credit'] - row['Debit']) / row['Debit'] <= (1 - self.config['ROUND_TRIP_THRESHOLD'])):
                        flags.append('ROUND_TRIP_DETECTED')
                        break
                else:
                    flags.append(None)
            else:
                flags.append(None)
                
        return flags

    def detect_dormant_account_activity(self, df):
        """
        Detects sudden activity in previously dormant accounts.
        
        Args:
            df: DataFrame containing transaction data
        Returns:
            list: Flags indicating dormant account activity
        """
        print("\nDetecting dormant account activity...")
        flags = []
        
        # Analyze each transaction
        for idx in tqdm(range(len(df)), desc="Analyzing account activity"):
            row = df.iloc[idx]
            
            # Check for activity in dormancy period
            window_before = df[
                (df['Date'] >= row['Date'] - pd.Timedelta(self.config['DORMANCY_PERIOD'])) &
                (df['Date'] < row['Date'])
            ]
            
            # If no activity in dormancy period, check for sudden activity
            if len(window_before) == 0:
                window_after = df[
                    (df['Date'] > row['Date']) &
                    (df['Date'] <= row['Date'] + pd.Timedelta('7D'))
                ]
                
                # Flag if activity exceeds threshold
                if len(window_after) >= self.config['ACTIVITY_SPIKE_THRESHOLD']:
                    flags.append('DORMANT_ACCOUNT_ACTIVITY')
                else:
                    flags.append(None)
            else:
                flags.append(None)
                
        return flags

    def detect_high_withdrawals(self, df):
        """
        Detects high-value withdrawals relative to account balance.
        
        Args:
            df: DataFrame containing transaction data
        Returns:
            list: Flags indicating high-value withdrawals
        """
        print("\nDetecting high-value withdrawals...")
        flags = []
        
        # Calculate running balance
        df['Balance'] = (df['Credit'] - df['Debit']).cumsum()
        
        # Analyze each transaction
        for idx in tqdm(range(len(df)), desc="Analyzing withdrawals"):
            row = df.iloc[idx]
            
            # Only check withdrawals
            if row['Debit'] > 0:
                # Flag if withdrawal exceeds threshold percentage of balance
                if row['Debit'] >= row['Balance'] * self.config['HIGH_WITHDRAWAL_THRESHOLD']:
                    flags.append('HIGH_WITHDRAWAL')
                else:
                    flags.append(None)
            else:
                flags.append(None)
                
        return flags

    def detect_unique_recipients(self, df):
        """
        Detects transactions to multiple unique recipients.
        
        Args:
            df: DataFrame containing transaction data
        Returns:
            list: Flags indicating suspicious recipient patterns
        """
        print("\nDetecting unique recipient patterns...")
        flags = []
        
        # Analyze each transaction
        for idx in tqdm(range(len(df)), desc="Analyzing recipients"):
            row = df.iloc[idx]
            
            # Get transactions within recipient analysis window
            window = df[
                (df['Date'] >= row['Date'] - pd.Timedelta(self.config['UNIQUE_RECIPIENT_WINDOW'])) &
                (df['Date'] <= row['Date'])
            ]
            
            # Extract recipient information from Description field
            recipients = window['Description'].nunique()
            
            # Flag if number of unique recipients exceeds threshold
            if recipients >= self.config['UNIQUE_RECIPIENT_THRESHOLD']:
                flags.append('MULTIPLE_UNIQUE_RECIPIENTS')
            else:
                flags.append(None)
                
        return flags

    def analyze_transactions(self, df):
        """
        Enhanced main method to analyze transactions using all detection methods.
        
        Args:
            df: DataFrame containing transaction data
        Returns:
            DataFrame: Analyzed transactions with fraud flags and risk scores
        """
        print("\nStarting comprehensive fraud detection analysis...")
        
        # Standardize data format
        df = self.standardize_columns(df)
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        
        # Execute all detection methods
        all_flags = {
            'pattern_flags': self.detect_small_transfer_patterns(df),
            'structuring_flags': self.detect_structuring(df),
            'movement_flags': self.detect_rapid_money_movement(df),
            'timing_flags': self.detect_unusual_timing(df),
            'network_flags': self.detect_network_patterns(df),
            # New detection methods
            'spike_flags': self.detect_transaction_spikes(df),
            'small_txn_flags': self.detect_unusual_small_transactions(df),
            'round_trip_flags': self.detect_round_trip_transactions(df),
            'dormant_flags': self.detect_dormant_account_activity(df),
            'withdrawal_flags': self.detect_high_withdrawals(df),
            'recipient_flags': self.detect_unique_recipients(df)
        }
        
        # Combine all flags
        combined_flags = []
        for i in range(len(df)):
            row_flags = []
            for flag_type in all_flags.values():
                if flag_type[i]:
                    row_flags.append(flag_type[i])
            combined_flags.append('|'.join(row_flags) if row_flags else None)
        
        df['Fraud_Flags'] = combined_flags
        
        # Calculate comprehensive risk score
        df['Risk_Score'] = self.calculate_comprehensive_risk_score(df)
        
        return df

    def calculate_comprehensive_risk_score(self, df):
        """
        Calculates a more comprehensive risk score based on all detected patterns.
        
        Args:
            df: DataFrame containing transaction data with fraud flags
        Returns:
            list: Risk scores for each transaction
        """
        print("\nCalculating comprehensive risk scores...")
        risk_scores = []
        
        # Calculate risk score for each transaction
        for flags in tqdm(df['Fraud_Flags'].fillna(''), desc="Calculating scores"):
            if flags:
                flag_list = flags.split('|')
                
                # Calculate base score from risk weights
                base_score = sum(self.risk_weights.get(flag, 0) for flag in flag_list)
                
                # Additional risk factors
                num_flags = len(flag_list)
                pattern_diversity = len(set(flag_list))
                
                # Adjust score based on number and diversity of flags
                multiplier = 1.0
                if num_flags > 2:
                    multiplier += 0.2  # Increase score for multiple flags
                if pattern_diversity > 1:
                    multiplier += 0.1 * pattern_diversity  # Increase score for diverse patterns
                
                # Cap final score at 1.0
                final_score = min(base_score * multiplier, 1.0)
                risk_scores.append(final_score)
            else:
                risk_scores.append(0.0)
        
        return risk_scores

# Example usage
# Initialize detector
detector = EnhancedFraudDetector()

# Load and prepare transaction data
df = pd.read_csv('Bank Statements/Axis/923010030924818-01-01-2023to18-11-2024bankaxis.csv')

# Analyze transactions
analyzed_df = detector.analyze_transactions(df)

# Calculate risk scores
analyzed_df['Risk_Score'] = detector.get_risk_score(analyzed_df)

# Show high-risk transactions (risk score > 0.7)
high_risk = analyzed_df[analyzed_df['Risk_Score'] > 0.7]

print(high_risk)