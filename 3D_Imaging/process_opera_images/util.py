#!/usr/bin/env python
# coding: utf-8


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
import cv2
import skimage.transform

# Define function for creating ome.tiff
def Get_ome_tif_file_single(signal, SaveIMG,z_indicator,concatenate=False):
    signal.sort()
    Brightfield_images = signal

    files = Brightfield_images
    #z_indicator = '_z(\d\d)'
    #z_indicator = 'p(\d\d)-'
    regex_z = re.compile(z_indicator)

    def sort_key(file):
          return regex_z.search(file).group(1)

    files.sort(key=sort_key)
    
    ## @TODO: second identical copy of brightfield signal stacked here as placeholder for fluorescence to satisfy requirement of fnet predict -- this is redundant and has downstream effects. Fix at source 
    if concatenate:
        brightfld_array = np.expand_dims(np.stack([tifffile.imread(file) for file in files]), axis=0) # stack all the sorted tiffiles and expand dim to create a "channel dim"
        brightfld_array.shape
        Mix_ch630X9_6hr_X3 = np.concatenate((brightfld_array, brightfld_array ), axis=0)
        channel_names = ['brightfield','fluroscense']
    else:
        brightfld_array = np.stack([tifffile.imread(file) for file in files]) # stack all the sorted tiffiles and expand dim to create a "channel dim"
        brightfld_array.shape
        Mix_ch630X9_6hr_X3 = brightfld_array
        channel_names = ['brightfield']

    
    if int(aicsimageio.__version__.split('.')[0]) < 4:
        with writers.ome_tiff_writer.OmeTiffWriter(SaveIMG, overwrite_file=True) as writer: 
            writer.save(Mix_ch630X9_6hr_X3, dimension_order="CZYX", channel_names=channel_names) 
    else:
        writers.ome_tiff_writer.OmeTiffWriter.save(Mix_ch630X9_6hr_X3, SaveIMG)
    #print("\n\nFinalshape of "+ SaveIMG)
    #print(Mix_ch630X9_6hr_X3.shape)
    return 

def concatenate_tif(signal,target,SaveIMG): 
    if signal.shape != target.shape:
        print('Error: signal shape mismatch with target shape')
        return
    else:
        signal = np.expand_dims(signal, axis=0)
        target = np.expand_dims(target, axis=0)
        #print(signal.dtype)
        #print(target.dtype)
        Mix_ch630X9_6hr_X3 = np.concatenate((signal, target), axis=0)
        #print('concatneated shape is {}'.format(Mix_ch630X9_6hr_X3.shape))
        channel_names = ['brightfield','fluroscense']
        #print(aicsimageio.__version__)
        if int(aicsimageio.__version__.split('.')[0]) < 4:
            with writers.ome_tiff_writer.OmeTiffWriter(SaveIMG, overwrite_file=True) as writer: 
                writer.save(Mix_ch630X9_6hr_X3, dimension_order="CZYX", channel_names=channel_names) 
        else:
            # Create a new OME-TIFF file.
            writers.ome_tiff_writer.OmeTiffWriter.save(Mix_ch630X9_6hr_X3,uri=SaveIMG,dim_order="CZYX",channel_names=channel_names)

def downsample(stack,downsample_factor=0.5):
    list_of_slices = []
    for z in range(0,stack.shape[0]):
        slice_image = stack[z]
        scaled_image =  cv2.GaussianBlur(slice_image,(5,5),1)
        scaled_image = cv2.resize(scaled_image,(1024,1024),downsample_factor,downsample_factor,cv2.INTER_CUBIC)
        list_of_slices.append(scaled_image)
    new_stack = np.stack(list_of_slices)
    return new_stack
