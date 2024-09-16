# Process Opera Images....
#   --------------------------------
# This script processes raw Opera images, and 1) rename the images, 2) create OME Tiff files, 3) create csv files



import argparse
import sys
from tqdm import tqdm
import sys
import os 
from renameOperaImages import renameOpera
from create_ome_tif import createOMETiff

def process_arguments(args):
    if args.only_renameOpera:
        print('You have selected to run this script for renaming opera images')
        print('Start processing...., please wait...')
        path2Save = renameOpera(args.dir,args.saveDir,args.type)
    elif args.only_createOmeTiff:
        print('You have selected to run this script for creating ome tif images')
        print('Start processing...., please wait...')
        createOMETiff(args.dir,args.saveOmeTiff,args.type.lower())
    else:
        print('Running this script for renaming opera images, and creating ome tif images')
        print('Start processing...., please wait...')
        path2Save = renameOpera(args.dir,args.saveDir,args.type)
        createOMETiff(path2Save,args.saveOmeTiff,args.type.lower())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Getting argument for running renaming and createing OME tiff script for Opera Images")
    parser.add_argument('-dir',required=True, help="Path to the main folder of the data")
    parser.add_argument('-type',required=True,help='either nucleus or mito or cytoplasm')
    parser.add_argument('-saveDir',required=False,help='Optional: path to dir where converted images will be saved',default='.')
    parser.add_argument('-saveOmeTiff',required=False,help='Optional: path to dir where OME tiff files will be saved', default='./opera_ome_tif')
    parser.add_argument('--only_renameOpera',required=False,help='Optional: only running the script for renaming the Opera images',default=False,action='store_true')
    parser.add_argument('--only_createOmeTiff',required=False,help='Optional: only running the script for creating opera files',default=False,action='store_true')
    args = parser.parse_args()
    process_arguments(args)
