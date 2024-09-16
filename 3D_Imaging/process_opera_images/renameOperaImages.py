#!/usr/bin/env python
# coding: utf-8
################################
# Code was written by Saeed Alahmari, aalahmari.saeed@gmail.com
# Python script for renaming the Opera Images
# The new names for opera images will be FoF<sk><column>_<row><frame>_<DATE>
################################

import re
import os
from util import Get_ome_tif_file_single
import argparse
import sys
from tqdm import tqdm


def get_save_dir_and_DATE(directory):
    directory_subfolders = directory.split(os.sep)
    saveTODir = directory_subfolders[-2]
    print('images will be saved in {}'.format(saveTODir))
    DATE = directory_subfolders[-2].rsplit('_',1)[1]
    return saveTODir, DATE

def get_wells(filenames):
    wells = [i.split('c',1)[0] for i  in filenames if not i.startswith('.') and i.endswith('.tiff')]
    wells = set(wells)
    return wells

def find_filenames_with_different_f(directory,path2Save,DATE):
    filenames = os.listdir(directory)
    wells = get_wells(filenames)
    for well in wells:
        well_edited = well.replace('r','_')
        pattern = well+r'c(\d+)+f(\d+)+p(\d+)-ch(\d+)+sk(\d+)fk\d+fl\d+\.tiff'
        brightfield_dict = {}
        fluorescent_dict ={}
        for filename in filenames:
            #print(filename)
            match = re.match(pattern, filename)
            if match:
                c_value = match.group(1)
                f_value = match.group(2)
                p_value = match.group(3)
                ch_value = match.group(4)
                sk_value = match.group(5)
                #key = 'FoF'+sk_value+'r02'+'f'+f_value
                if int(ch_value) == 2:
                    key = 'FoF'+sk_value+c_value+well_edited+f_value+'_'+DATE+'_'+'brightfield' 
                    if key in brightfield_dict.keys():
                        brightfield_dict[key].append(filename)
                    else:
                        brightfield_dict[key] = [filename]
                elif int(ch_value) == 1:
                    key = 'FoF'+sk_value+c_value+well_edited+f_value+'_'+DATE+'_'+'fluorescent'
                    if key in fluorescent_dict.keys():
                        fluorescent_dict[key].append(filename)
                    else:
                        fluorescent_dict[key] = [filename]

        return brightfield_dict,fluorescent_dict

# Example usage
#directory_path = '/Volumes/WD Element/Collaboration/Moffitt_Noemi/Aim1_AllenModel/data/opera_images/A01_OperaPhenix_230425/230425_FL B5-2 bin1__2023-04-25T13_57_15-Measurement 1/Images'
#path2Save = '/Volumes/WD Element/Collaboration/Moffitt_Noemi/Aim1_AllenModel/data/opera_images'

def rename_OperaImages(directory_path,path2Save,part_of_cell):
    SaveToDir,DATE = get_save_dir_and_DATE(directory_path)
    #path2Save = 'A01_OperaPhenix_230425'

    path2Save = path2Save + '/' + SaveToDir
    brightfield,fluorescent = find_filenames_with_different_f(directory_path,path2Save,DATE)

    for key in fluorescent.keys():
        if not os.path.exists(os.path.join(path2Save,key.rsplit('_',1)[0])):
            os.makedirs(os.path.join(path2Save,key.rsplit('_',1)[0]))
        path2Save_ome = os.path.join(path2Save,key.rsplit('_',1)[0])
        filenames_list = fluorescent[key]
        files_path = [directory_path + os.sep + i for i in filenames_list]
        Get_ome_tif_file_single(signal = files_path,SaveIMG = path2Save_ome+os.sep+part_of_cell+'.t.tiff',z_indicator = 'p(\d\d)-')

    for key in brightfield.keys():
        if not os.path.exists(os.path.join(path2Save,key.rsplit('_',1)[0])):
            os.makedirs(os.path.join(path2Save,key.rsplit('_',1)[0]))
        path2Save_ome = os.path.join(path2Save,key.rsplit('_',1)[0])
        filenames_list = brightfield[key]
        files_path = [directory_path + os.sep + i for i in filenames_list]
        Get_ome_tif_file_single(signal = files_path,SaveIMG = path2Save_ome+os.sep+part_of_cell+'.s.tiff',z_indicator = 'p(\d\d)-')
    return path2Save
        # Call to savae ome here (make sure to have the key and the filename sent to the function)
    #print(fluorescent)

def renameOpera(path2Data,path2Save,part_of_cell):
    if not part_of_cell.lower() in ['nucleus','mito','cytoplasm']:
        print('Error: -type must be one of: nucleus,mito,cytoplasm')
        sys.exit()
    print('Started Renaming for Opera Images, please wait ...')
    for folder in tqdm(os.listdir(path2Data)):
        if not os.path.isdir(os.path.join(path2Data,folder)):
            continue 
        directory_path = os.path.join(path2Data,folder)
        #print(part_of_cell)
        path2Save = rename_OperaImages(directory_path,path2Save,part_of_cell.lower())
    return path2Save
        

        
