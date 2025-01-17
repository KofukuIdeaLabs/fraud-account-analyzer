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
            # Pattern 1: Multiple small incoming transfers followed by quick withdrawals
            # Common in scams where fraudsters collect small amounts from multiple victims
            'SMALL_TRANSFER_MIN': 1000,  # Minimum amount to consider as suspicious small transfer
            'SMALL_TRANSFER_MAX': 5000,  # Maximum amount to consider as suspicious small transfer  
            'SIMILAR_TRANSFER_WINDOW': '48h',  # Time window to look for similar transfers
            'MIN_SIMILAR_TRANSFERS': 5,  # Minimum number of similar transfers to flag as suspicious
            'QUICK_WITHDRAWAL_WINDOW': '24h',  # Time window to check for withdrawals after transfers
            
            # Pattern 2: Structuring (Breaking large amounts into smaller ones to avoid detection)
            'STRUCTURING_WINDOW': '72h',  # Time window to check for structured transactions
            'STRUCTURING_MIN_TRANSACTIONS': 3,  # Minimum number of related transactions to consider structuring
            'STRUCTURING_AMOUNT_THRESHOLD': 50000,  # Total amount threshold for structuring pattern
            
            # Pattern 3: Rapid money movement (Quick cycles of deposits and withdrawals)
            'RAPID_MOVEMENT_WINDOW': '24h',  # Window to check for rapid money movement
            'RAPID_MOVEMENT_MIN_CYCLES': 2,  # Minimum number of deposit-withdrawal cycles to flag
            
            # Pattern 4: Unusual transaction timing (Off-hours and weekend activity)
            'OFF_HOURS_START': 22,  # Start of off-hours period (10 PM)
            'OFF_HOURS_END': 5,     # End of off-hours period (5 AM)
            'WEEKEND_THRESHOLD': 0.7,  # Percentage of weekend transactions to consider suspicious
            
            # Pattern 5: Network analysis (Connections between accounts)
            'COMMON_BENEFICIARY_THRESHOLD': 3,  # Number of accounts sending to same beneficiary to flag
            'NETWORK_ANALYSIS_WINDOW': '7D'  # Time window for analyzing account networks
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
        
        # Date column regex pattern
        date_pattern = re.compile(r'(trans(action)?[\s_]?)?date|dt|txn[\s_]?d(ate|t)', re.IGNORECASE)
        date_col = next((col for col in cols if date_pattern.search(col)), None)
        
        # Description column regex pattern  
        desc_pattern = re.compile(r'desc(ription)?|narration|particular(s)?|details|transaction|memo|reference|remark(s)?', re.IGNORECASE)
        desc_col = next((col for col in cols if desc_pattern.search(col)), None)
        
        # Debit column regex pattern
        debit_pattern = re.compile(r'debit(s)?|withdraw(al)?s?(\(dr\))?|paid[\s_]?out|outflow', re.IGNORECASE)
        debit_col = next((col for col in cols if debit_pattern.search(col)), None)
        
        # Credit column regex pattern
        credit_pattern = re.compile(r'credit(s)?|deposit(s)?(\(cr\))?|paid[\s_]?in|inflow', re.IGNORECASE)
        credit_col = next((col for col in cols if credit_pattern.search(col)), None)
        
        # If separate debit/credit columns don't exist, look for amount column
        amount_col = next((col for col in cols if 'amount' in col.lower()), None)
        
        # Standardize the columns
        if date_col:
            standardized_df = standardized_df.rename(columns={date_col: 'Date'})
        
        if desc_col:
            standardized_df = standardized_df.rename(columns={desc_col: 'Description'})
        
        if amount_col and not (debit_col and credit_col):
            standardized_df['Debit'] = standardized_df[amount_col].apply(lambda x: abs(x) if x < 0 else 0)
            standardized_df['Credit'] = standardized_df[amount_col].apply(lambda x: x if x > 0 else 0)
            standardized_df = standardized_df.drop(columns=[amount_col])
        else:
            if debit_col:
                standardized_df = standardized_df.rename(columns={debit_col: 'Debit'})
            if credit_col:
                standardized_df = standardized_df.rename(columns={credit_col: 'Credit'})
        
        # Fill NaN values with 0 for Debit/Credit columns
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
            
            if len(small_credits) >= self.config['MIN_SIMILAR_TRANSFERS']:
                # Check for subsequent withdrawals in configured window
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
                if len(transactions) >= self.config['STRUCTURING_MIN_TRANSACTIONS']:
                    total_amount = transactions.sum()
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
                if txn['Credit'] > 0:
                    current_cycle['in'] += txn['Credit']
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
        for idx in tqdm(range(len(df)), desc="Analyzing networks"):
            row = df.iloc[idx]
            # Get transactions within network analysis window
            window = df[
                (df['Date'] >= row['Date'] - pd.Timedelta(self.config['NETWORK_ANALYSIS_WINDOW'])) &
                (df['Date'] <= row['Date'])
            ]
            
            # Analyze beneficiary patterns if beneficiary data is available
            if 'Beneficiary' in df.columns:
                beneficiary_sources = window.groupby('Beneficiary')['Description'].nunique()
                suspicious_beneficiaries = beneficiary_sources[
                    beneficiary_sources >= self.config['COMMON_BENEFICIARY_THRESHOLD']
                ]
                
                if not suspicious_beneficiaries.empty and row['Beneficiary'] in suspicious_beneficiaries.index:
                    flags.append('SUSPICIOUS_NETWORK_PATTERN')
                    continue
            
            flags.append(None)
        return flags

    def analyze_transactions(self, df):
        """
        Main method to analyze transactions using all detection methods.
        
        Args:
            df: DataFrame containing transaction data
        Returns:
            DataFrame: Original data with added fraud flags
        """
        print("\nStarting fraud detection analysis...")
        time.sleep(2)
        print("\nTotal transactions to analyze:", len(df))
        time.sleep(2)
        
        # Standardize column names first
        df = self.standardize_columns(df)
        
        # Ensure date column is in datetime format
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        
        # Apply all detection methods
        pattern_flags = self.detect_small_transfer_patterns(df)
        structuring_flags = self.detect_structuring(df)
        movement_flags = self.detect_rapid_money_movement(df)
        timing_flags = self.detect_unusual_timing(df)
        network_flags = self.detect_network_patterns(df)
        
        print("\nCombining detection results...")
        time.sleep(2)
        # Combine all flags for each transaction
        combined_flags = []
        for i in range(len(df)):
            row_flags = []
            for flags in [pattern_flags, structuring_flags, movement_flags, timing_flags, network_flags]:
                if flags[i]:
                    row_flags.append(flags[i])
            combined_flags.append('|'.join(row_flags) if row_flags else None)
        
        df['Fraud_Flags'] = combined_flags
        
        # Print summary of findings
        flagged_transactions = df['Fraud_Flags'].notna().sum()
        print(f"\nAnalysis complete!")
        time.sleep(2)
        print(f"\nFound {flagged_transactions} suspicious transactions out of {len(df)} total transactions")
        print(f"\nDetection rate: {(flagged_transactions/len(df)*100):.2f}%")
        
        return df

    def get_risk_score(self, df):
        """
        Calculates risk score for each transaction based on detected flags.
        
        Args:
            df: DataFrame containing transaction data with fraud flags
        Returns:
            list: Risk scores between 0 and 1 for each transaction
        """
        print("\nCalculating risk scores...")
        time.sleep(2)
        # Weights for different types of flags
        flag_weights = {
            'SCAM_PATTERN_SMALL_TRANSFERS': 0.8,  # Highest risk
            'STRUCTURING_PATTERN': 0.7,
            'RAPID_MONEY_MOVEMENT': 0.6,
            'SUSPICIOUS_TIMING': 0.4,  # Lower risk
            'SUSPICIOUS_NETWORK_PATTERN': 0.5
        }
        
        risk_scores = []
        for flags in tqdm(df['Fraud_Flags'].fillna(''), desc="Calculating scores"):
            if flags:
                flag_list = flags.split('|')
                score = sum(flag_weights.get(flag, 0) for flag in flag_list)
                risk_scores.append(min(score, 1.0))  # Cap score at 1.0
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