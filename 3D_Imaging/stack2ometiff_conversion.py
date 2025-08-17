#!/usr/bin/env python
# coding: utf-8

# # Python-iterating-through-folders-in-directory for nucleus image
# Load the image and convert into brightfield_array

import argparse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tifffile
from tifffile import imread
from matplotlib.backends.backend_pdf import PdfPages
import glob, os, re
from aicsimageio import AICSImage, imread
from aicsimageio.writers import OmeTiffWriter
from skimage import color, io

#cd '/raid/crdlab/ix1/Projects/M005_MeasuringFitnessPerClone_2019/data/GastricCancerCLs/3Dbrightfield/NCI-N87/'

# Define function for creating ome.tiff
def Get_ome_tif_file(signal, target, SaveIMG = "A02_ometiffconversion/FoF3_210803_fluorescent.nucleus.ome.tif"):

    signal.sort()
    target.sort()

    Fluro_images = []
    Grayscale_Fluro_images = []
    Brightfield_images = []

    Brightfield_images = signal
    
    Fluro_images = target
    #print(Fluro_images) 
    
    files = Brightfield_images
    z_indicator = '_z(\d\d)'
    regex_z = re.compile(z_indicator)

    def sort_key(file):
        return regex_z.search(file).group(1)

    files.sort(key=sort_key)
    brightfld_array = np.expand_dims(np.stack([color.rgb2gray(tifffile.imread(file)) for file in files]), axis=0) # stack all the sorted tiffiles and expand dim to create a "channel dim"
    # z-score normalize array.
    brightfld_array.shape
    
    files = Fluro_images
    z_indicator = '_z(\d\d)'
    regex_z = re.compile(z_indicator)

    def sort_key(file):
        return regex_z.search(file).group(1)

    files.sort(key=sort_key)
    fluroscense_array = np.expand_dims(np.stack([color.rgb2gray(tifffile.imread(file)) for file in files]), axis=0) # stack all the sorted tiffiles and expand dim to create a "channel dim"
    # z-score normalize array.
    fluroscense_array.shape #print(X3.shape)
    
    Mix_ch630X9_6hr_X3 = np.concatenate((brightfld_array, fluroscense_array ), axis=0)
    print(Mix_ch630X9_6hr_X3.shape)
    
    with OmeTiffWriter(SaveIMG, overwrite_file=True) as writer: 
        writer.save(Mix_ch630X9_6hr_X3, dimension_order="CZYX", channel_names=['brightfield', 'fluroscense']) 
  
    print("\n\nFinalshape of "+ SaveIMG)
    print(Mix_ch630X9_6hr_X3.shape)
    return  Mix_ch630X9_6hr_X3.shape


# Define function for creating ome.tiff
def Get_ome_tif_file_single(signal, SaveIMG = "A02_ometiffconversion/FoF3_210803_fluorescent.nucleus.ome.tif"):
      
    signal.sort()

    Brightfield_images = signal

    files = Brightfield_images
    z_indicator = '_z(\d\d)'
    regex_z = re.compile(z_indicator)

    def sort_key(file):
          return regex_z.search(file).group(1)

    files.sort(key=sort_key)
    brightfld_array = np.expand_dims(np.stack([color.rgb2gray(tifffile.imread(file)) for file in files]), axis=0) # stack all the sorted tiffiles and expand dim to create a "channel dim"
    # z-score normalize array.
    brightfld_array.shape

    ## @TODO: second identical copy of brightfield signal stacked here as placeholder for fluorescence to satisfy requirement of fnet predict -- this is redundant and has downstream effects. Fix at source 
    Mix_ch630X9_6hr_X3 = np.concatenate((brightfld_array, brightfld_array ), axis=0)
    # Mix_ch630X9_6hr_X3 = brightfld_array
    print(Mix_ch630X9_6hr_X3.shape)
    # Save the image
    OmeTiffWriter.save(
        Mix_ch630X9_6hr_X3, 
        SaveIMG, 
        dim_order="CZYX", 
        channel_names=['brightfield', 'fluorescence']
    )



    #with OmeTiffWriter(SaveIMG, overwrite_file=True) as writer: 
    #   writer.save(Mix_ch630X9_6hr_X3, dimension_order="CZYX", channel_names=['brightfield', 'fluroscense']) 
  
    print("\n\nFinalshape of "+ SaveIMG)
    print(Mix_ch630X9_6hr_X3.shape)
    return  Mix_ch630X9_6hr_X3.shape


# place the desired path location in the diectory 
directory = "/Volumes/Expansion/Collaboration/Moffitt_Noemi/BioinformaticsPaper/NCI-N87-2/A01_rawData"

for filename in os.listdir(directory):
    #print(os.path.join(directory, filename)) 
    # select desired subfolder from the entire lists of folders
    # if filename.endswith("fluorescent.nucleus") or filename.endswith("fluorescent.mito") or filename.endswith("fluorescent.cytoplasm"):
    if filename.startswith("FoF"):
    # if filename.startswith("FoF"):: 
    #  if filename.endswith("fluorescent.mito"): 
    # if filename.endswith("fluorescent.cytoplasm"):
        desiredfile = os.path.join(directory, filename)
        Savefile = filename
        #print(desiredfile)
        #print(Savefile)
        #only select .tif files
        signals = desiredfile + "/*ch01*.tif"
        targets = desiredfile + "/*.t_*.tif"
        # input desired directory to store the saved .ome files 
        if not os.path.exists("/Volumes/Expansion/Collaboration/Moffitt_Noemi/BioinformaticsPaper/NCI-N87-2/A02_ometiffconversion/"):
            os.makedirs("/Volumes/Expansion/Collaboration/Moffitt_Noemi/BioinformaticsPaper/NCI-N87-2/A02_ometiffconversion/")
        SaveDir = "/Volumes/Expansion/Collaboration/Moffitt_Noemi/BioinformaticsPaper/NCI-N87-2/A02_ometiffconversion/" + Savefile + ".ome.tif"
        # call the function : which convert the numpy array to .ome file 
        if len(glob.glob(targets))==0:
            Get_ome_tif_file_single(signal = glob.glob(signals), SaveIMG = SaveDir)
        else:
            Get_ome_tif_file(signal = glob.glob(signals) , target = glob.glob(targets), SaveIMG = SaveDir)

# def main(parser):
#     args = parser.parse_args()
#     sys.exit()
        
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Getting argument for running ome_tiff conversion")
#     parser.add_argument('-dir',required=True, help="Path to the folder containing the images to run cellpose")
#     main(parser)
        
