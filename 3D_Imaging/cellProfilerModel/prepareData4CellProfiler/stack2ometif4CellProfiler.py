import argparse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tifffile
from tifffile import imread
from matplotlib.backends.backend_pdf import PdfPages
import glob, os, re
from aicsimageio import AICSImage, imread, writers
import aicsimageio
from skimage import color, io
import sys
from tqdm import tqdm


# Define function for creating ome.tiff
def Get_ome_tif_file_single(signal, SaveIMG,z_indicator):
    signal.sort()
    Brightfield_images = signal

    files = Brightfield_images
    #z_indicator = '_z(\d\d)'
    #z_indicator = 'p(\d\d)-'
    regex_z = re.compile(z_indicator)

    def sort_key(file):
          return regex_z.search(file).group(1)

    files.sort(key=sort_key)
    brightfld_array = np.stack([tifffile.imread(file) for file in files]) 
    # z-score normalize array.
    brightfld_array.shape

    ## @TODO: second identical copy of brightfield signal stacked here as placeholder for fluorescence to satisfy requirement of fnet predict -- this is redundant and has downstream effects. Fix at source 
    #Mix_ch630X9_6hr_X3 = np.concatenate((brightfld_array, brightfld_array ), axis=0)
    Mix_ch630X9_6hr_X3 = brightfld_array
    
    if int(aicsimageio.__version__.split('.')[0]) < 4:
        with writers.ome_tiff_writer.OmeTiffWriter(SaveIMG, overwrite_file=True) as writer: 
            writer.save(Mix_ch630X9_6hr_X3, dimension_order="CZYX", channel_names=['brightfield']) 
    else:
        writers.ome_tiff_writer.OmeTiffWriter.save(Mix_ch630X9_6hr_X3, SaveIMG)
    #print("\n\nFinalshape of "+ SaveIMG)
    #print(Mix_ch630X9_6hr_X3.shape)
    return  Mix_ch630X9_6hr_X3.shape

def clean_image_list(img_list):
    cleaned_list = [i for i in img_list if not i.rsplit('/',1)[1].startswith('.')]
    return cleaned_list


# ...existing code...

def process_folders(base_directory, save_directory, z_indicator):
    for FoF in tqdm(os.listdir(base_directory)):
        if not os.path.isdir(os.path.join(base_directory,FoF)):
            continue
        base_directory_temp = os.path.join(base_directory,FoF)
        for root, dirs, files in os.walk(base_directory_temp):
            # Filter files ending with 'ch00_tif'
            ch00_files = [os.path.join(root, file) for file in files if file.endswith('ch00.tif')]
            ch01_files = [os.path.join(root, file) for file in files if file.endswith('ch01.tif')]
            ch02_files = [os.path.join(root, file) for file in files if file.endswith('ch02.tif')]
            for files_list,ch in [[ch00_files,'ch00'],[ch01_files,'ch01'],[ch02_files,'ch02']]:
                if files_list:
                    # Clean the file list
                    cleaned_files = clean_image_list(files_list)
                    
                    # Define the output file name
                    folder_name = os.path.basename(root)
                    save_img_path = os.path.join(save_directory, f"{folder_name}_{ch}.ome.tif")
                    
                    # Call the function
                    #print(f"Processing folder: {root}")
                    Get_ome_tif_file_single(cleaned_files, save_img_path, z_indicator)

# Define the base directory and save directory
base_directory = "/Volumes/Expansion/Collaboration/Moffitt_Noemi/BioinformaticsPaper/NCI-N87-2/A01_rawData"
save_directory = "/Volumes/Expansion/Collaboration/Moffitt_Noemi/BioinformaticsPaper/NCI-N87-2/T09_ome_tif_files"




z_indicator = '_z(\\d\\d)'

# Ensure the save directory exists
os.makedirs(save_directory, exist_ok=True)

# Process the folders
process_folders(base_directory,save_directory,z_indicator)

