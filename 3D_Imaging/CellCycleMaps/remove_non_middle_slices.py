

import os
import sys
import pandas as pd
from tqdm import tqdm
import shutil as sh
import argparse

#path2Data = "/Volumes/Expansion/Collaboration/Moffitt_Noemi/BioinformaticsPaper/data/NCI-N87/A07_patches"

def remove_slices(path2Data,middleSlice):
    slices_to_consider = ['slice'+str(i) for i in range(35-int(middleSlice/2)+1,35+int(middleSlice/2)+1)]
    print('slices to keep are {}'.format(slices_to_consider))
    print('now removing other slices ...')
    for FoF in tqdm(os.listdir(path2Data)):
        if not FoF.startswith('FoF'):
            continue 
        for cellCycle in os.listdir(os.path.join(path2Data,FoF)):
            if not cellCycle.startswith('cellCycle'):
                continue 
            for image_name in os.listdir(os.path.join(path2Data,FoF,cellCycle,'images_padded')):
                if image_name.startswith('._') or not image_name.endswith('.tif'):
                    continue
                if image_name.split('_')[1] in slices_to_consider:
                    continue
                slice = image_name.split('_')[1]
                if slice in slices_to_consider:
                    continue 
                else:
                    #print('Removing {}'.format(image_name))
                    os.remove(os.path.join(path2Data,FoF,cellCycle,'images_padded',image_name))
                    #sys.exit()

def main():
    parser = argparse.ArgumentParser(description="This script removes non middle slices")
    
    # Adding arguments
    parser.add_argument("-path2FoFs", type=str, help="path to FoFs")
    parser.add_argument("-num_slices", type=int, help="How many slices to consider (e.g. 10)", default=10,required=False)

    # Parse arguments
    args = parser.parse_args()
    remove_slices(args.path2FoFs,args.num_slices)

if __name__=="__main__":
    main()
