#Python script for running extracting patches for multiple FoFs. 

import os 
import sys
from extractPadSingleCell import extract_and_pad

import pandas as pd 

path2FoFs = '/Users/saeedalahmari/Downloads/BioInformaticsPaper/PMO_MeasuringFitnessperClone/3D_Imaging/CellCycleMaps/code/extract_pad_patches/data/NCI-N87-3/A04_CellposeOutput/FoF1001003_221018_fucci.nucleus'
path2Masks = '/Users/saeedalahmari/Downloads/BioInformaticsPaper/PMO_MeasuringFitnessperClone/3D_Imaging/CellCycleMaps/code/extract_pad_patches/data/NCI-N87-3/A06_multiSignals_Linked/FoF1001003_221018_fucci.nucleus'
path2Save = '/Users/saeedalahmari/Downloads/BioInformaticsPaper/PMO_MeasuringFitnessperClone/3D_Imaging/CellCycleMaps/code/extract_pad_patches/data/NCI-N87-3/A07_patches'


prepare_data = extract_and_pad(path2FoFs,path2Masks, path2Save, '.csv')

prepare_data.process_images() # to extract patches 
#prepare_data.get_cells_coordinates()


""" 
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
    process_FoF_one_cell_per_time(path2FoFs_temp, path2Masks_temp, path2Save_temp)
    #list_of_data, df_max_cell = get_cell_data(path2Save_temp,path2Save_temp)
    #pad_data(list_of_data, df_max_cell)
   
    FoF_list.append(directory)
    Max_index_list.append(max_index)
    Max_value_list.append(max_value)

df = pd.DataFrame()
df['FoF'] = FoF_list
df['max_index'] = Max_index_list
df['max_value'] = Max_value_list

df.to_csv(os.path.join(path2Save,'max-slice-log.csv'),index=False)
"""
