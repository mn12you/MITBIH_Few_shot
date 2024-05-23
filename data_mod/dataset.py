import os
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import pickle
class ECGDataset(Dataset):
    def __init__(self, fs=360, data_dir="./mit_bih", label_csv="mit_bih_sub/mit_bih.csv"):
        super(ECGDataset, self).__init__()
        self.df=pd.read_csv(label_csv)
        with open(Path("labelbinarizer","lb.pkl"),'rb') as tokenizer:
            self.lb=pickle.load(tokenizer)

    def __getitem__(self,index:int):
        row=self.df.iloc[index]
        label=self.lb.transform([row.labels])
        data=np.load(row.filename)
        return data,label
    def __len__(self):
        return len(self.df)