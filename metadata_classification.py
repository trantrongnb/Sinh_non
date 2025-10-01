import numpy as np
import pandas as pd
import os
from utils.read_data import read_data
num_cols = ['Gestation', 'Rectime', 'Age', 'Parity', 'Abortions', 'Weight',
            'Hypertension', 'Diabetes', 'Placental_position',
            'Bleeding_first_trimester', 'Bleeding_second_trimester', 'Funneling', 'Smoker']

sig_file=[]
data_file=[]

for file in os.listdir("dataset"):
    file_path=os.path.join("dataset",file)
    if file.endswith(".hea"):
        data_file.append(file_path)
    else:
        sig_file.append(file_path)

all_metadata=[]
all_label=[]
for file in data_file:
    metadata,label=read_data(file)
    if metadata != None and label!=None:
        all_metadata.append(metadata)
        all_label.append(label[0])

df=pd.DataFrame(all_metadata)

# for col in num_cols:
#     df[col] = pd.to_numeric(df[col], errors='coerce')

# for col in df.columns:
#     mode_val = df[col].mode()
#     if len(mode_val) > 0:
#         df[col] = df[col].fillna(mode_val[0])
#     else:
#         df[col] = df[col].fillna(0)


df.to_csv('du_lieu.csv', index=False, encoding='utf-8')
