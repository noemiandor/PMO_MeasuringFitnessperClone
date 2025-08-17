import os 
import pandas as pd 
from tqdm import tqdm
import tifffile as tifffile
import numpy as np
from aicsimageio import AICSImage, imread, writers
import aicsimageio

path2Masks_csv = '/Volumes/Expansion/Collaboration/Moffitt_Noemi/BioinformaticsPaper/NCI-N87-2/A05_PostProcessCellposeOutput/'
path2Images_tif = '/Volumes/Expansion/Collaboration/Moffitt_Noemi/BioinformaticsPaper/NCI-N87-2/A04_CellposeOutput/'
path2SaveMask_tif = '/Volumes/Expansion/Collaboration/Moffitt_Noemi/BioinformaticsPaper/NCI-N87-2/T09_ome_tif_files'

if not os.path.exists(path2SaveMask_tif):
    os.makedirs(path2SaveMask_tif)

for folder in tqdm(os.listdir(path2Masks_csv)):
    if not os.path.isdir(os.path.join(path2Masks_csv, folder)):
        continue
    raw_stack_tif = tifffile.imread(os.path.join(path2Images_tif,folder, 'nucleus.t.tif'))
    mask = np.zeros_like(raw_stack_tif)
    for file in sorted(os.listdir(os.path.join(path2Masks_csv,folder,'All_Cells_coordinates'))):
        if not file.endswith('.csv') or file.startswith('.'):
            continue
        df = pd.read_csv(os.path.join(path2Masks_csv,folder,'All_Cells_coordinates',file),encoding='utf-8') # folder
        cell_id = file.split('_')[2]
        for x,y,z in zip(df['x'],df['y'],df['z']):
            mask[int(z),int(y),int(x)] = int(cell_id)

    base_name = folder
    #tifffile.imwrite(os.path.join(path2SaveMask_tif, base_name+'_masks'+'.ome.tif'),mask) #folder
    SaveIMG = os.path.join(path2SaveMask_tif, base_name+'_mask'+'.ome.tif')
    Mix_ch630X9_6hr_X3 = mask

    if int(aicsimageio.__version__.split('.')[0]) < 4:
        with writers.ome_tiff_writer.OmeTiffWriter(SaveIMG, overwrite_file=True) as writer: 
            writer.save(Mix_ch630X9_6hr_X3, dimension_order="CZYX", channel_names=['brightfield']) 
    else:
        writers.ome_tiff_writer.OmeTiffWriter.save(Mix_ch630X9_6hr_X3, SaveIMG)