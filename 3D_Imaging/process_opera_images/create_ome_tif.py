
# This script is for creating two channels ome tiff images for training Fnet AllenModel 

import re
import os
import argparse
from util import concatenate_tif,downsample
import sys
from tqdm import tqdm
from skimage import io
import tifffile
from aicsimageio import AICSImage, imread
'''
scikit-image==0.16.2
numpy==1.20.0
imageio==2.9.0
aicsimageio==3.0.7

'''



def createOMETiff(path2Data,path2Save,part_of_cell):
    if not os.path.exists(path2Save):
        os.makedirs(path2Save)

    if not part_of_cell.lower() in ['nucleus','mito','cytoplasm']:
        print('Error: -type must be one of: nucleus,mito,cytoplasm')
        sys.exit()
    print('Started creating ome tif for Opera Images, please wait ...')
    for folder in tqdm(os.listdir(path2Data)):
        if not os.path.isdir(os.path.join(path2Data,folder)):
            continue 
        directory_path = os.path.join(path2Data,folder)
        target = None
        signal = None
        #print(folder)
        for file in os.listdir(directory_path):
            if '.t.tiff' in file:
                target = tifffile.imread(os.path.join(directory_path,file))
            elif '.s.tiff' in file:
                signal = tifffile.imread(os.path.join(directory_path,file))

        if target is not None and signal is not None:
            path2SaveFile = os.path.join(path2Save,folder)
            signal = downsample(signal,downsample_factor=0.474) #0.474 because we want to reduce opera images from 2160x2160  to 1024x1024
            target = downsample(target,downsample_factor=0.474)
            #io.imsave('signal_downsampled.tiff',signal)
            concatenate_tif(signal,target,path2SaveFile+'.ome.tiff')
        elif target is None and signal is not None:
            path2SaveFile = os.path.join(path2Save,folder)
            signal = downsample(signal,downsample_factor=0.474) #0.474 because we want to reduce opera images from 2160x2160  to 1024x1024
            #io.imsave('signal_downsampled.tiff',signal)
            concatenate_tif(signal,signal,path2SaveFile+'.ome.tiff')
        elif target is None and signal is None:
            continue
        
        

        
