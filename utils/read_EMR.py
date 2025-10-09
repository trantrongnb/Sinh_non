import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
def read_EMR(folder_path):
    num_cols = ['Rectime','Age','Parity','Abortions','Weight','Bleeding_first_trimester','Bleeding_second_trimester','Smoker']

    file_EHG=[]
    file_EMR=[]

    for file in os.listdir(folder_path):
        if file.endswith(".hea"):
            file_EMR.append(file)
        else:
            file_EHG.append(file)
    
    all_EMR_data=[]
    all_Labels=[]
    all_Name_files=[]

    for file in file_EMR:
        file_path=os.path.join("dataset",file)
        with open(file_path,'r',encoding='utf-8') as f:
            metadata_dict = {col: None for col in num_cols}
            for line in f:
                line_split=line.strip().split()
                if line_split[0]=='#':
                    if line_split[1] in num_cols:
                        metadata_dict[line_split[1]]=line_split[2]
                    else:
                        if line_split[1]=='Gestation':
                            Ges=float(line_split[2])*7

            if float(metadata_dict['Rectime'])*7>=26*7:
                all_Name_files.append(file)
                all_EMR_data.append(metadata_dict)
                if Ges>=37*7:
                    all_Labels.append([0,Ges])
                else:
                    all_Labels.append([1,Ges])

    df=pd.DataFrame(all_EMR_data)
    df = df.replace(['None',None], np.nan)
    df['Bleeding_first_trimester']=df['Bleeding_first_trimester'].replace({'yes':1,'no':0})
    df['Bleeding_second_trimester']=df['Bleeding_second_trimester'].replace({'yes':1,'no':0})
    df['Smoker']=df['Smoker'].replace({'yes':1,'no':0})

    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in df.columns:
        mode_val = df[col].mode()
        if len(mode_val) > 0:
            df[col] = df[col].fillna(mode_val[0])
        else:
            df[col] = df[col].fillna(0)



    return all_Labels,df,all_Name_files




