import os
import pandas as pd
import numpy as np


class DataLoader:
    def __init__(self):
        self.train_df = None
        self.test_df = None
        self.eval_df = None
    
    def load_data(self, train_path, test_path, eval_path):
        try:
            self.train_df = pd.read_csv(train_path, sep=';')
            self.test_df = pd.read_csv(test_path, sep=';')
            self.eval_df = pd.read_csv(eval_path,  sep=';')
            
            print(f"Training set size: {len(self.train_df)}")
            print(f"Test set size: {len(self.test_df)}")
            print(f"Evaluation set size: {len(self.eval_df)}")
            
            # Combine title and text
            self.train_df['combined_text'] = self.train_df['title'].fillna('') + ' ' + self.train_df['text'].fillna('')
            self.test_df['combined_text'] = self.test_df['title'].fillna('') + ' ' + self.test_df['text'].fillna('')
            self.eval_df['combined_text'] = self.eval_df['title'].fillna('') + ' ' + self.eval_df['text'].fillna('')
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def get_data(self):
        return self.train_df, self.test_df, self.eval_df
    
    def get_labels(self):
        y_train = self.train_df['label'].values
        y_test = self.test_df['label'].values
        y_eval = self.eval_df['label'].values
        
        print(f"Label distribution - Train: {np.bincount(y_train)}")
        print(f"Label distribution - Test: {np.bincount(y_test)}")
        print(f"Label distribution - Eval: {np.bincount(y_eval)}")
        
        return y_train, y_test, y_eval
    
