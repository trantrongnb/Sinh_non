import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_data(file_path):
    num_cols = ['Gestation', 'Rectime', 'Age', 'Parity', 'Abortions', 'Weight',
            'Hypertension', 'Diabetes', 'Placental_position',
            'Bleeding_first_trimester', 'Bleeding_second_trimester', 'Funneling', 'Smoker']

    with open(file_path,'r',encoding='utf-8') as f:
        metadata_dict = {col: None for col in num_cols}
        for line in f:
            line_split=line.strip().split()
            if line_split[0]=='#':
                if line_split[1] in num_cols:
                    metadata_dict[line_split[1]]=line_split[2]
        if float(metadata_dict['Rectime'])*7>=26*7:
            if float(metadata_dict['Gestation'])*7>=37*7:
                label=0
            else:
                label=1
            return metadata_dict,[label,float(metadata_dict['Gestation'])*7.0]
        else:
            return None,None