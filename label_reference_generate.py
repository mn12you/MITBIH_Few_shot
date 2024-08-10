import os
from pathlib import Path
import sys
SCRIPT_DIR =Path(__file__).parents[0]
print(SCRIPT_DIR)
data_base_dir="mit_bih_sub"
sys.path.append(SCRIPT_DIR)
from args import parse_args
import wfdb
import pandas as pd 
import numpy as np
import scipy.signal as sp

arg=parse_args()
data_record=Path(SCRIPT_DIR,arg.data_dir,'RECORDS')
df=pd.DataFrame(columns=['ecg_id','patient_id','labels','filename'])

#### filter setting ####
fs=360
window_bond=int(fs*0.36)
# window_bond=127
med_window=int(0.72*fs)
b,a=sp.butter(5,50,btype="lowpass",analog=False,fs=fs)

records=[]
with open(data_record) as f:
    records = f.readlines()

def normalize(sig):
    new_sig=(sig-sig.min())/(sig.max()-sig.min())
    return new_sig
count=0

for record in records:
    patient_id=str(record[:-1])
    patient_record = wfdb.rdrecord("./mit_bih/"+patient_id)
    record_signal=patient_record.p_signal[:,0]
    path=Path(data_base_dir,patient_id)
    if not path.exists():
        path.mkdir()
    record_signal=sp.filtfilt(b,a,record_signal)
    base=sp.medfilt(record_signal,med_window)
    record_signal=record_signal-base
    patient_annotation = wfdb.rdann("./mit_bih/"+patient_id,extension="atr")
    file_path=Path(data_base_dir,patient_id)
    if file_path.exists():
        print("dir exists")
    else:
        file_path.mkdir()

    #### filter conduct ####
    label_transform={"N":"N","L":"N","R":"N","e":"N","j":"N","A":"S","a":"S","J":"S","S":"S","V":"V","E":"V","F":"F","/":"Q","f":"Q","Q":"Q"}
    # label_transform={"N":"N","L":"L","R":"R","e":"O","j":"O","A":"A","a":"O","J":"O","S":"O","V":"V","E":"O","F":"FVN","/":"P","f":"FPN","Q":"O"}

    for ind, ann in enumerate(patient_annotation.sample):
        filename=Path(data_base_dir,patient_id,patient_id+"_"+str(ind)+".csv")
        if ann>window_bond and ann< len(record_signal)-(window_bond):
                if patient_annotation.symbol[ind] in  label_transform.keys():
                    beat_sig=record_signal[ann-window_bond:ann+window_bond+2]
                    beat_sig=normalize(beat_sig)
                    np.save(filename,beat_sig)
                    # np.savetxt(filename, beat_sig.reshape(1,-1), fmt='%.03f',delimiter=",")
                    df_temp=pd.DataFrame({'ecg_id':count,'patient_id':patient_id,'labels':label_transform[patient_annotation.symbol[ind]],'filename':filename}, index=[0])
                    df=pd.concat([df,df_temp], ignore_index=True)
                    count=count+1
    df.to_csv("./mit_bih.csv",index=False)

