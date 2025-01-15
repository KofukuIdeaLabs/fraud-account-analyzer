import os
import re
import pandas as pd
import PyPDF2
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BankStatementParser:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.raw_text = None
        self.bank_name = None
        self.transactions_df = None
    
    def extract_text(self):
        """Extract text from PDF file"""
        try:
            with open(self.pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                self.raw_text = text
                return True
        except Exception as e:
            print(f"Error reading PDF {self.pdf_path}: {str(e)}")
            return False
    
    def detect_bank(self):
        """Detect bank based on PDF content"""
        text = self.raw_text.lower()
        
        if 'hdfc bank' in text:
            self.bank_name = 'HDFC'
        elif 'state bank of india' in text or 'sbi' in text:
            self.bank_name = 'SBI'
        elif 'icici bank' in text:
            self.bank_name = 'ICICI'
        elif 'axis bank' in text:
            self.bank_name = 'AXIS'
        elif 'kotak' in text:
            self.bank_name = 'KOTAK'
        elif 'yes bank' in text:
            self.bank_name = 'YES'
        elif 'federal bank' in text:
            self.bank_name = 'FEDERAL'
        elif 'canara bank' in text:
            self.bank_name = 'CANARA'
        else:
            self.bank_name = 'UNKNOWN'
        
        print(f"Detected bank: {self.bank_name}")
    
    def parse_transactions(self):
        """Parse transactions based on bank format"""
        if not self.bank_name:
            self.detect_bank()
        
        if self.bank_name == 'HDFC':
            self._parse_hdfc()
        elif self.bank_name == 'ICICI':
            self._parse_icici()
        elif self.bank_name == 'AXIS':
            self._parse_axis()
        elif self.bank_name == 'KOTAK':
            self._parse_kotak()
        elif self.bank_name == 'YES':
            self._parse_yes_bank()
        elif self.bank_name == 'FEDERAL':
            self._parse_federal()
        elif self.bank_name == 'CANARA':
            self._parse_canara()
        else:
            self._parse_generic()
    
    def _extract_date(self, date_str):
        """Try to parse date string in various formats"""
        date_formats = [
            '%d/%m/%y', '%d/%m/%Y', '%d-%m-%y', '%d-%m-%Y',
            '%Y-%m-%d', '%d%m%y', '%d%m%Y'
        ]
        
        for fmt in date_formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except:
                continue
        return None
    
    def _parse_amount(self, amount_str):
        """Parse amount string to float"""
        if isinstance(amount_str, (int, float)):
            return float(amount_str)
        
        # Remove currency symbols and commas
        amount_str = re.sub(r'[₹,]', '', str(amount_str))
        # Extract numbers (including decimals)
        match = re.search(r'[-+]?\d*\.?\d+', amount_str)
        if match:
            return float(match.group())
        return None
    
    def _parse_hdfc(self):
        """Parse HDFC Bank statement format"""
        lines = self.raw_text.split('\n')
        transactions = []
        
        # Regular expressions for HDFC format
        date_pattern = r'\d{2}/\d{2}/\d{2,4}'
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Try to find date at the start of line
            date_match = re.match(date_pattern, line.strip())
            if date_match:
                parts = line.split()
                if len(parts) >= 4:  # Minimum parts needed for a transaction
                    date = self._extract_date(parts[0])
                    
                    # Look for debit/credit amounts
                    debit = credit = 0
                    for i, part in enumerate(parts):
                        amount = self._parse_amount(part)
                        if amount is not None:
                            # Usually last two numbers are debit/credit and balance
                            if i == len(parts) - 2:
                                if 'dr' in line.lower():
                                    debit = amount
                                else:
                                    credit = amount
                            
                    # Get description (everything between date and amounts)
                    desc_start = line.find(parts[0]) + len(parts[0])
                    desc_end = line.rfind(parts[-1])
                    description = line[desc_start:desc_end].strip()
                    
                    if date:
                        transactions.append({
                            'Date': date,
                            'Description': description,
                            'Debit': debit,
                            'Credit': credit,
                            'Balance': self._parse_amount(parts[-1])
                        })
        
        self.transactions_df = pd.DataFrame(transactions)
        if not self.transactions_df.empty:
            self.transactions_df = self.transactions_df.sort_values('Date')
            print(f"\nExtracted {len(self.transactions_df)} transactions")
            print("\nFirst few transactions:")
            print(self.transactions_df.head())
        else:
            print("No transactions could be extracted")
    
    def _parse_icici(self):
        """Parse ICICI Bank statement format"""
        # TODO: Implement ICICI specific parsing
        self._parse_generic()
    
    def _parse_axis(self):
        """Parse Axis Bank statement format"""
        # TODO: Implement Axis specific parsing
        self._parse_generic()
    
    def _parse_kotak(self):
        """Parse Kotak Bank statement format"""
        # TODO: Implement Kotak specific parsing
        self._parse_generic()
    
    def _parse_yes_bank(self):
        """Parse Yes Bank statement format"""
        # TODO: Implement Yes Bank specific parsing
        self._parse_generic()
    
    def _parse_federal(self):
        """Parse Federal Bank statement format"""
        # TODO: Implement Federal Bank specific parsing
        self._parse_generic()
    
    def _parse_canara(self):
        """Parse Canara Bank statement format"""
        # TODO: Implement Canara Bank specific parsing
        self._parse_generic()
    
    def _parse_generic(self):
        """Generic parsing for unknown bank formats"""
        lines = self.raw_text.split('\n')
        transactions = []
        
        # Common patterns
        date_pattern = r'\d{2}[/-]\d{2}[/-]\d{2,4}'
        amount_pattern = r'(?:(?:Rs|INR|₹)?[,\d]+\.?\d*)'
        
        for line in lines:
            if not line.strip():
                continue
            
            # Look for date
            date_match = re.search(date_pattern, line)
            if date_match:
                date = self._extract_date(date_match.group())
                
                # Look for amounts
                amounts = re.findall(amount_pattern, line)
                amounts = [self._parse_amount(amt) for amt in amounts if self._parse_amount(amt) is not None]
                
                if date and amounts:
                    # Get description (text between date and first amount)
                    desc_start = line.find(date_match.group()) + len(date_match.group())
                    desc_end = line.find(str(amounts[0]))
                    description = line[desc_start:desc_end].strip()
                    
                    transaction = {
                        'Date': date,
                        'Description': description,
                        'Amount': amounts[0] if amounts else None,
                        'Balance': amounts[-1] if len(amounts) > 1 else None
                    }
                    transactions.append(transaction)
        
        self.transactions_df = pd.DataFrame(transactions)
        if not self.transactions_df.empty:
            self.transactions_df = self.transactions_df.sort_values('Date')
            print(f"\nExtracted {len(self.transactions_df)} transactions")
            print("\nFirst few transactions:")
            print(self.transactions_df.head())
        else:
            print("No transactions could be extracted")

def main():
    statements_dir = "Bank Statements"
    
    # Get the first PDF file
    pdf_files = [f for f in os.listdir(statements_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print("No PDF files found in the Bank Statements directory")
        return
    
    first_pdf = pdf_files[0]
    pdf_path = os.path.join(statements_dir, first_pdf)
    print(f"Processing: {first_pdf}")
    
    # Parse the PDF
    parser = BankStatementParser(pdf_path)
    if parser.extract_text():
        parser.parse_transactions()
        
        if parser.transactions_df is not None:
            # Save to CSV for inspection
            output_file = os.path.splitext(first_pdf)[0] + '_parsed.csv'
            parser.transactions_df.to_csv(output_file, index=False)
            print(f"\nParsed data saved to: {output_file}")
            
            # Print some basic statistics
            print("\nBasic Statistics:")
            if 'Amount' in parser.transactions_df.columns:
                print("Total Amount:", parser.transactions_df['Amount'].sum())
            elif 'Debit' in parser.transactions_df.columns and 'Credit' in parser.transactions_df.columns:
                print("Total Debits:", parser.transactions_df['Debit'].sum())
                print("Total Credits:", parser.transactions_df['Credit'].sum())
            print("Number of Transactions:", len(parser.transactions_df))
            if 'Date' in parser.transactions_df.columns:
                print("Date Range:", parser.transactions_df['Date'].min(), "to", parser.transactions_df['Date'].max())

if __name__ == "__main__":
    main() 