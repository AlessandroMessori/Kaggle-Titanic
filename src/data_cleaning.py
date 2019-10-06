import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .label_encoder import MultiColumnLabelEncoder


class DataCleaner:

    def __init__(self, df):
        self.df = df
        self.encoder = MultiColumnLabelEncoder()
        self.scaler = StandardScaler()

    def fill(self, column, mean=False):
        if mean:
            age_mean = np.mean(self.df[column])

            for i, col in enumerate(self.df[column]):
                if np.isnan(col):
                    self.df.at[i, column] = age_mean
        else:
            for i, col in enumerate(pd.isnull(self.df[column])):
                if col:
                    self.df.at[i, column] = 0
                else:
                    self.df.at[i, column] = 1

            self.df.fillna(method='bfill', inplace=True)

    def encode_and_scale(self):
        self.df = self.encoder.fit_transform(self.df)
        self.df = self.scaler.fit_transform(self.df)
