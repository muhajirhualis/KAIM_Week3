import pandas as pd
import numpy as np
import os

class DataLoader:
    def __init__(self, file_path):
        """
        Initializes the DataLoader with the path to the dataset.
        """
        self.file_path = file_path
        self.df = None
        
    def load_data(self):
        """
        Loads data from the txt file.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        # Load data (using pipe separation)
        try:
            self.df = pd.read_csv(self.file_path, sep= '|', low_memory=False)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            
        # Optimization: Parse dates
        if 'TransactionMonth' in self.df.columns:
            self.df['TransactionMonth'] = pd.to_datetime(self.df['TransactionMonth'], errors='coerce')
        print(f"Data Loaded. Shape: {self.df.shape}")
        return self.df
      
    def get_column_types(self):
        """
        Returns a summary of column types.
        """
        if self.df is not None:
            return self.df.dtypes
        return None