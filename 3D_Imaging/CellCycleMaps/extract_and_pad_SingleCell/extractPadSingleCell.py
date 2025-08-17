import os
import pandas as pd
import numpy as np
import tifffile as tifffile
import cv2
from util import extract_patches, get_coordinates_of_cells, create_csv, process_FoF_one_cell_per_time,generate_mask
import glob as glob
from pad_image import get_max_slice, pad_images

class extractPadSingleCell:  
    def __init__(self, path_to_images, path_to_masks, output_path,input_type='.csv'):
        self.path_to_images = path_to_images
        self.path_to_masks = path_to_masks
        self.output_path = output_path
        self.input_type = input_type # input type of '.csv' is for masks that are originated\
        #as csv files, but '.tif' is for files that are already in tif format

    def check_image_dtype(self, image):
        if image.dtype != np.float32:
            print(f"Warning: input image of type {image.dtype}, but the code requires images of type np.float32")
            image = image.astype(np.float32)
        return image

    def generate_mask(self, compartments_loc, cell_id):
        nucleus_mask = np.zeros((70, 1024, 1024))
        mito_mask = np.zeros((70, 1024, 1024))
        cyto_mask = np.zeros((70, 1024, 1024))

        nucleus_cell_loc = pd.read_csv(compartments_loc["nucleus"])
        for x, y, z in zip(nucleus_cell_loc['x'], nucleus_cell_loc['y'], nucleus_cell_loc['z']):
            nucleus_mask[int(z - 1), int(y - 1), int(x - 1)] = int(cell_id)

        if os.path.exists(compartments_loc["mito"]):
            mito_cell_loc = pd.read_csv(compartments_loc["mito"])
            for x, y, z in zip(mito_cell_loc['x'], mito_cell_loc['y'], mito_cell_loc['z']):
                mito_mask[int(z - 1), int(y - 1), int(x - 1)] = int(cell_id)

        if os.path.exists(compartments_loc["cytoplasm"]):
            cyto_cell_loc = pd.read_csv(compartments_loc["cytoplasm"])
            for x, y, z in zip(cyto_cell_loc['x'], cyto_cell_loc['y'], cyto_cell_loc['z']):
                cyto_mask[int(z - 1), int(y - 1), int(x - 1)] = int(cell_id)

        return nucleus_mask, mito_mask, cyto_mask

    def save_image(self, image, file_name):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        tifffile.imwrite(
            os.path.join(self.output_path, file_name),
            image,
            photometric='minisblack',
            metadata={'axes': 'ZCYX'}, # Specify the axes order as ZCYX for OME-TIFF
            ome = True
        )
    # This script loops through the masks and extract the cells coordinates for each mask. 
    def get_cells_coordinates(self):

        for mask_name in os.listdir(self.path_to_masks):
            if mask_name.startswith('._') or not mask_name.endswith('nucleus.p.tif'):
                continue 
            mask = tifffile.imread(os.path.join(self.path_to_masks,mask_name))
            print(mask.shape)
            
            list_of_cells_centers = get_coordinates_of_cells(self.path_to_masks,mask.astype(np.int8),mask_name)
            #Save the centers coordinates
            print('Writing the coordinates for each image ....')
            f_output = mask_name.split('.tif')[0]
            create_csv(self.path_to_masks,list_of_cells_centers,f_output)

    def process_images(self):
        if self.input_type.lower() == '.csv':
            # this part is for 'csv' file masks 
            compartments_dict = process_FoF_one_cell_per_time(self.path_to_images,self.path_to_masks)
            for compartment_loc, cell_id, cell_cycle, name in zip(compartments_dict['compartments'],compartments_dict['cell_ids'],
                                                                  compartments_dict['cell_cycles'],compartments_dict['names']):
                
                nucleus_image, mito_image, cyto_image, nucleus_mask, mito_mask, cyto_mask = generate_mask(compartment_loc, self.path_to_images,cell_id)

                patches_list_nucleus = extract_patches(nucleus_mask, cell_id, raw_image = nucleus_image)
                patches_list_mito = extract_patches(mito_mask, cell_id, raw_image = mito_image)
                patches_list_cyto = extract_patches(cyto_mask, cell_id, raw_image = cyto_image)

                padded_images_nucelus = pad_images(patches_list_nucleus) # max_shape is default (64,64)
                padded_images_mito = pad_images(patches_list_mito) # max_shape is default (64,64)
                padded_images_cyto = pad_images(patches_list_cyto) # max_shape is default (64,64)

                nucleus_padded = np.stack(padded_images_nucelus)
                mito_padded = np.stack(padded_images_mito)
                cyto_padded = np.stack(padded_images_cyto)
                padded_image = np.stack([nucleus_padded,mito_padded,cyto_padded],axis=1)
                self.save_image(np.stack([nucleus_padded,mito_padded,cyto_padded],axis=1), file_name='cell_'+str(cell_id)+'.tif')

        elif self.input_type.lower() == '.tif':
            masks = glob.glob(os.path.join(self.path_to_masks,'*_cellpose_masks.tif'))
            for mask_path in masks:
                mask = tifffile.imread(os.path.join(self.path_to_masks,mask_path))
                image = tifffile.imread(mask_path.replace('_cellpose_masks.tif','.nucleus.p.tif'))
                max_cell_index = mask.max()
                for index in range(1,max_cell_index+1):
                   patches_list = extract_patches(mask, index, raw_image = image)
                   max_slice_shape = get_max_slice(patches_list)
                   padded_images = pad_images(patches_list,max_slice_shape)
                   self.save_image(np.stack(padded_images), file_name='cell_'+str(index)+'.tif')

