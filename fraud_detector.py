import os
import re
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BankStatementParser:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.transactions_df = None
        
    def load_csv(self):
        """Load transaction data from CSV file"""
        try:
            self.transactions_df = pd.read_csv(self.csv_path)
            # Convert date column to datetime
            date_col = next(col for col in self.transactions_df.columns if 'date' in col.lower())
            self.transactions_df[date_col] = pd.to_datetime(self.transactions_df[date_col])
            return True
        except Exception as e:
            print(f"Error reading CSV {self.csv_path}: {str(e)}")
            return False

    def analyze_transactions(self):
        """Analyze transactions with unified fraud detection rules"""
        if self.transactions_df is None or self.transactions_df.empty:
            print("No transactions to analyze")
            return
            
        # Standardize column names using regex patterns
        cols = self.transactions_df.columns
        
        # Date column regex pattern
        date_pattern = re.compile(r'(trans(action)?[\s_]?)?date|dt|txn[\s_]?d(ate|t)', re.IGNORECASE)
        date_col = next(col for col in cols if date_pattern.search(col))
        
        # Description column regex pattern  
        desc_pattern = re.compile(r'desc(ription)?|narration|particular(s)?|details|transaction|memo|reference|remark(s)?', re.IGNORECASE)
        desc_col = next(col for col in cols if desc_pattern.search(col))
        
        # Debit column regex pattern
        debit_pattern = re.compile(r'debit(s)?|withdraw(al)?s?(\(dr\))?|paid[\s_]?out|outflow', re.IGNORECASE)
        debit_col = next((col for col in cols if debit_pattern.search(col)), None)
        
        # Credit column regex pattern
        credit_pattern = re.compile(r'credit(s)?|deposit(s)?(\(cr\))?|paid[\s_]?in|inflow', re.IGNORECASE)
        credit_col = next((col for col in cols if credit_pattern.search(col)), None)
        
        # If separate debit/credit columns don't exist, look for amount column
        amount_col = next((col for col in cols if 'amount' in col.lower()), None)
        if amount_col and not (debit_col and credit_col):
            # Create debit/credit columns based on amount sign
            self.transactions_df['Debit'] = self.transactions_df[amount_col].apply(lambda x: abs(x) if x < 0 else 0)
            self.transactions_df['Credit'] = self.transactions_df[amount_col].apply(lambda x: x if x > 0 else 0)
        else:
            # Standardize column names
            self.transactions_df = self.transactions_df.rename(columns={
                debit_col: 'Debit',
                credit_col: 'Credit'
            })
            
        self.transactions_df = self.transactions_df.rename(columns={
            date_col: 'Date',
            desc_col: 'Description'
        })
        
        # Fill NaN values with 0 for Debit/Credit columns
        self.transactions_df['Debit'] = self.transactions_df['Debit'].fillna(0)
        self.transactions_df['Credit'] = self.transactions_df['Credit'].fillna(0)
        
        # Sort by date
        self.transactions_df = self.transactions_df.sort_values('Date')
        
        # Add fraud detection flags
        self._add_fraud_flags()
        
        # Print analysis results
        self._print_analysis()

    def _add_fraud_flags(self):
        """Add fraud detection flags based on various rules"""
        flags = []
        
        # Configuration for detection rules
        HIGH_VALUE_THRESHOLD = 150000  # Adjust based on typical transaction amounts
        FREQUENT_TXN_WINDOW = '24H'
        FREQUENT_TXN_THRESHOLD = 15
        # SUSPICIOUS_KEYWORDS = ['casino', 'betting', 'gaming', 'crypto', 'bitcoin', 'forex']
        ROUND_AMOUNT_THRESHOLD = 10000  # Flag round amounts above this threshold
        HIGH_DEBIT_WINDOW = '24H'  # Window to check for multiple high-value debits
        HIGH_DEBIT_THRESHOLD = 3  # Number of high-value debits to trigger flag
        
        for idx, row in self.transactions_df.iterrows():
            row_flags = []
            
            # High value transactions
            transaction_amount = max(row['Debit'], row['Credit'])
            if transaction_amount > HIGH_VALUE_THRESHOLD:
                row_flags.append('HIGH_VALUE')
            
            # # Suspicious keywords in description
            # if any(keyword in str(row['Description']).lower() for keyword in SUSPICIOUS_KEYWORDS):
            #     row_flags.append('SUSPICIOUS_MERCHANT')
            
            # Round amounts (for amounts above threshold)
            if transaction_amount > ROUND_AMOUNT_THRESHOLD and transaction_amount % 1000 == 0:
                row_flags.append('ROUND_AMOUNT')
            
            # Frequent transactions
            window_txns = self.transactions_df[
                (self.transactions_df['Date'] >= row['Date'] - pd.Timedelta(FREQUENT_TXN_WINDOW)) &
                (self.transactions_df['Date'] <= row['Date'])
            ]
            if len(window_txns) >= FREQUENT_TXN_THRESHOLD:
                row_flags.append('FREQUENT_TXN')
            
            # Multiple high-value debits in short period
            debit_window = self.transactions_df[
                (self.transactions_df['Date'] >= row['Date'] - pd.Timedelta(HIGH_DEBIT_WINDOW)) &
                (self.transactions_df['Date'] <= row['Date']) &
                (self.transactions_df['Debit'] > HIGH_VALUE_THRESHOLD)
            ]
            if len(debit_window) >= HIGH_DEBIT_THRESHOLD:
                row_flags.append('MULTIPLE_HIGH_DEBITS')
            
            # Large debit followed by multiple small credits
            if row['Debit'] > HIGH_VALUE_THRESHOLD:
                future_week = self.transactions_df[
                    (self.transactions_df['Date'] > row['Date']) &
                    (self.transactions_df['Date'] <= row['Date'] + pd.Timedelta('7D'))
                ]
                if len(future_week[future_week['Credit'] > 0]) >= 3:
                    row_flags.append('SPLIT_DEPOSITS')
            
            flags.append('|'.join(row_flags) if row_flags else None)
        
        self.transactions_df['Flags'] = flags

    def _print_analysis(self):
        """Print analysis results and statistics"""
        print("\nTransaction Analysis Results:")
        print("-" * 50)
        print(f"Total Transactions: {len(self.transactions_df)}")
        print(f"Date Range: {self.transactions_df['Date'].min().date()} to {self.transactions_df['Date'].max().date()}")
        print(f"Total Debits: ₹{self.transactions_df['Debit'].sum():,.2f}")
        print(f"Total Credits: ₹{self.transactions_df['Credit'].sum():,.2f}")
        
        # Flagged transactions summary
        flagged = self.transactions_df[self.transactions_df['Flags'].notna()]
        if not flagged.empty:
            print("\nFlagged Transactions Summary:")
            print("-" * 50)
            flag_counts = pd.Series([
                flag
                for flags in self.transactions_df['Flags'].dropna()
                for flag in flags.split('|')
            ]).value_counts()
            print(flag_counts)
            
            print("\nDetailed Flagged Transactions:")
            print("-" * 50)
            print(flagged[['Date', 'Description', 'Debit', 'Credit', 'Flags']])

def main():
    statements_dir = "Bank Statements/Axis"
    csv_file = "923010030924818-01-01-2023to18-11-2024bankaxis.csv"  # Process single CSV file
    csv_path = os.path.join(statements_dir, csv_file)
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    print(f"\nProcessing: {csv_file}")
    print("=" * 50)
    
    # Parse and analyze the CSV
    parser = BankStatementParser(csv_path)
    if parser.load_csv():
        parser.analyze_transactions()
        
        # Save analyzed data with flags
        if parser.transactions_df is not None:
            output_file = os.path.splitext(csv_file)[0] + '_analyzed.csv'
            output_path = os.path.join(statements_dir, output_file)
            parser.transactions_df.to_csv(output_path, index=False)
            print(f"\nAnalyzed data saved to: {output_path}")

if __name__ == "__main__":
    main() 