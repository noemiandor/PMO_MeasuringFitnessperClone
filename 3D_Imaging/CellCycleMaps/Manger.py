#Python script for running extracting patches for multiple FoFs. 

import os 
import sys
from extract_patches import *
from pad_image import *
import pandas as pd 

path2FoFs = '/Volumes/Expansion/Collaboration/Moffitt_Noemi/BioinformaticsPaper/NCI-N87/A04_CellposeOutput'
path2Masks = '/Volumes/Expansion/Collaboration/Moffitt_Noemi/BioinformaticsPaper/NCI-N87/A06_multiSignals_Linked'
path2Save = '/Volumes/Expansion/Collaboration/Moffitt_Noemi/BioinformaticsPaper/NCI-N87/A07_patches'

FoF_list = []
Max_index_list = []
Max_value_list = []

for directory in os.listdir(path2FoFs):
    if not os.path.isdir(os.path.join(path2FoFs,directory)):
        continue  
    path2FoFs_temp = os.path.join(path2FoFs,directory)
    path2Masks_temp = os.path.join(path2Masks,directory)
    path2Save_temp = os.path.join(path2Save,directory)
    print(directory)
    max_index,max_value = process_FoF(path2FoFs_temp, path2Masks_temp, path2Save_temp)
    list_of_data, df_max_cell = get_cell_data(path2Save_temp,path2Save_temp)
    pad_data(list_of_data, df_max_cell)
    
    FoF_list.append(directory)
    Max_index_list.append(max_index)
    Max_value_list.append(max_value)

df = pd.DataFrame()
df['FoF'] = FoF_list
df['max_index'] = Max_index_list
df['max_value'] = Max_value_list

df.to_csv(os.path.join(path2Save,'max-slice-log.csv'),index=False)

