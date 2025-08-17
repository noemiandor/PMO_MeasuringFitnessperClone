#util.py

import numpy as np
import pandas as pd 
import sys
import os
import cv2
import tifffile as tifffile
from glob import glob
from tqdm import tqdm

def extract_patches(image_volume, cell_id, raw_image = None):
    image_volume_for_cell_id = (image_volume == int(cell_id)) * 255
    image_volume_for_cell_id = image_volume_for_cell_id * raw_image
    image_volume_for_cell_id = image_volume_for_cell_id.astype(np.float64)
    #tifffile.imwrite('test.tif',image_volume_for_cell_id)
    list_of_patches = []
    for slice in range(0,image_volume_for_cell_id.shape[0]):
        # TBD, before getting the contour multiply the slice with the corresponding image (allen model prediction)
        contours_1, _ = cv2.findContours(image_volume_for_cell_id[slice,:,:].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours_1) > 0:
            image_extracted_n = get_cropped_cell(contours_1,image_volume_for_cell_id[slice,:,:],0,is_nucelus=False)
        else:
            image_extracted_n = np.zeros((64,64))
        list_of_patches.append(image_extracted_n)
    return list_of_patches

def process_FoF_one_cell_per_time(path2FoF,path2CellMasks):

    compartments_loc_dict = {}
    csv_file_list = []
    compartments_loc_list = []
    cell_id_list = []
    cell_cycle_list = []

    csv_files = glob(os.path.join(path2CellMasks, "nucleus.p*.csv"))
    path2Images = os.path.join(path2FoF)
    print('Generating patches, please wait ...')
    for csv_path in tqdm(csv_files):
        csv_file = os.path.basename(csv_path)
        if csv_file.startswith('._*'):
            continue # skip temp file 
        cell_id = csv_file.split('cell_')[1].split('_')[0]
        nucleus_path = csv_path
        mito_path = csv_path.replace('nucleus.p','mito.p')
        cyto_path = csv_path.replace('nucleus.p','cytoplasm.p')
        if not os.path.exists(mito_path) and not os.path.exists(cyto_path):
            #print('Skipping ... No mito or cyto...')
            mito_path = None
            cyto_path = None
        #print('cell_id: {}'.format(cell_id))
        df = pd.read_csv(csv_path)
        if 'cellCycle' not in df.columns:
            df['cellCycle'] = 0
            print('No cell cycle label provided. Setting to zero.')
        #print(df.shape)
        if df.shape[0] != 0:
            compartments_loc = {'nucleus':nucleus_path,'cytoplasm':cyto_path,'mito':mito_path}
            cell_Cycle = df['cellCycle'].iloc[0]
        compartments_loc_list.append(compartments_loc)
        cell_id_list.append(cell_id)
        cell_cycle_list.append(cell_Cycle)
        csv_file_list.append(csv_file)
    compartments_loc_dict['compartments'] = compartments_loc_list
    compartments_loc_dict['cell_ids'] = cell_id_list
    compartments_loc_dict['cell_cycles'] = cell_cycle_list
    compartments_loc_dict['names'] = csv_file_list
    return compartments_loc_dict
def check_image_dtype(image):
    if image.dtype != np.float32:
        print('Warning: input image of type {}, but the code require images of type {}'.format(image.dtype,np.float32))
        image.astype(np.float32)
    return image 

def generate_mask(compartments_loc,path2Images,cell_id):
    image_tif = tifffile.imread(os.path.join(path2Images,'nucleus.p.tif'))
    image_tif = check_image_dtype(image_tif)
    image_tif_shape = image_tif.shape
    Flag_mito = False
    Flag_cyto = False
    nucleus_mask = np.zeros((70,1024,1024))
    
    mito_tif = np.zeros((70,1024,1024))
    cyto_tif = np.zeros((70,1024,1024))
    mito_mask = np.zeros((70,1024,1024))
    cyto_mask = np.zeros((70,1024,1024))

    nucleus_cell_loc = pd.read_csv(compartments_loc["nucleus"])

    #nucleus_mask = np.zeros((70,1024,1024))
    for x,y,z in zip(nucleus_cell_loc['x'],nucleus_cell_loc['y'],nucleus_cell_loc['z']):
        nucleus_mask[int(z-1),int(y-1),int(x-1)] = int(cell_id)

    if os.path.exists(os.path.join(path2Images,'mito.p.tif')) and compartments_loc["mito"] is not None:
        mito_tif = tifffile.imread(os.path.join(path2Images,'mito.p.tif'))
        mito_tif = check_image_dtype(mito_tif)
        mito_mask = np.zeros((70,1024,1024))
        Flag_mito = True
        mito_cell_loc = pd.read_csv(compartments_loc["mito"])
        #print('processing cyto')
        #mito_mask = np.zeros((70,1024,1024))
        for x,y,z in zip(mito_cell_loc['x'],mito_cell_loc['y'],mito_cell_loc['z']):
            mito_mask[int(z-1),int(y-1),int(x-1)] = int(cell_id)

    if os.path.exists(os.path.join(path2Images,'cytoplasm.p.tif')) and compartments_loc["cytoplasm"] is not None:
        cyto_tif = tifffile.imread(os.path.join(path2Images,'cytoplasm.p.tif'))
        cyto_tif = check_image_dtype(cyto_tif)
        cyto_mask = np.zeros((70,1024,1024))
        Flag_cyto = True 
        cyto_cell_loc = pd.read_csv(compartments_loc["cytoplasm"])
        #print('processing mito')
        #cyto_mask = np.zeros((70,1024,1024))
        for x,y,z in zip(cyto_cell_loc['x'],cyto_cell_loc['y'],cyto_cell_loc['z']):
            cyto_mask[int(z-1),int(y-1),int(x-1)] = int(cell_id)
    return image_tif, mito_tif, cyto_tif, nucleus_mask, mito_mask, cyto_mask

def get_global_bounding_box(contours):
    all_x = []
    all_y = []
    all_w = []
    all_h = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        all_x.append(x)
        all_y.append(y)
        all_w.append(x + w)
        all_h.append(y + h)

    x_min = min(all_x)
    y_min = min(all_y)
    x_max = max(all_w)
    y_max = max(all_h)

    global_w = x_max - x_min
    global_h = y_max - y_min

    return (x_min, y_min, global_w, global_h)

def get_cropped_cell(contours,image_masked,index,is_nucelus=False):
    if len(contours) < 1:
        return 0
    #merged_contours = merge_contours_by_drawing(contours,image_masked[index,:,:])
    if len(contours) == 1:
        x, y, w, h = cv2.boundingRect(contours[0])
    else:
        x, y, w, h = get_global_bounding_box(contours)

    if w * h < 10 and is_nucelus: # skip small cells. 
        return np.zeros((64,64))
    elif (y - 10) < 1 or (x - 10) < 1:
        return np.zeros((64,64))
    elif (y + h + 10) >= image_masked.shape[0] or (x + w + 10) >= image_masked.shape[1]:
        return np.zeros((64,64))
    cell_extracted = image_masked[y - 10:y + h + 10, x - 10:x + w + 10]
    return cell_extracted



def get_center(maski):
  maski = maski.astype(np.uint8)
  M = cv2.moments(maski)
  try:
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
  except:
    cX = 0 
    cY = 0
  return (cX,cY)

# Save the cell centers in a file named with the mask name in the folder (Cells_center_coordinates)
def create_csv(images_path,locations_list,f):
  x = [i[0] for i in locations_list]
  y = [i[1] for i in locations_list]
  z = [i[2] for i in locations_list]

  df = pd.DataFrame()
  df['x'] = x
  df['y'] = y
  df['z'] = z

  if not os.path.isdir(os.path.join(images_path,'Cells_center_coordinates')):
    os.makedirs(os.path.join(images_path,'Cells_center_coordinates'))
  df.to_csv(os.path.join(images_path,'Cells_center_coordinates',f+'_Cells_Centers.csv'),index=False)

def euclidean_distance(p1,p2):
  distance = np.sqrt(np.power(p1[0] - p2[0],2) + np.power(p1[1] - p2[1],2))
  return distance
#Get the coordinates for all the cells. 
#Each cell coordinates will be written to a file with the mask name followed by the cell number in the folder (All_Cells_coordinates)
def get_all_coordinates(img,slice):
  indices = np.argwhere(img > 0)
  xyz_loc = []
  if len(indices > 0):
    indices=indices.tolist()
    xyz_loc = [j+[slice] for j in indices]
  return xyz_loc

def get_coordinates_of_cells(path2Images,mask,f):
        #Get center for each cell 
    print('Getting the coordinates of the image ....')
    if not os.path.isdir(os.path.join(path2Images,'All_Cells_coordinates')):
        os.makedirs(os.path.join(path2Images,'All_Cells_coordinates'))
    All_images_locations = []
    List_of_cell_Locations = []
    List_of_cell_Locations2 = []
    print('total cells is {}'.format(int(np.max(mask))))
    for i in range(1, int(np.max(mask))):
        maski = (mask == i)*255
        List_of_Images = [maski[i,:,:] for i in range(0,mask.shape[0])]
        list_of_centers = []
        xyz_loc_accum = []
        for ind,img in enumerate(List_of_Images):
            xy = get_center(img)
            xyz_loc = get_all_coordinates(img,ind)  #slice index (ind)
            if len(xyz_loc) > 0:
                if len(xyz_loc_accum) == 0:
                    xyz_loc_accum = xyz_loc 
                else:
                    xyz_loc_accum = xyz_loc_accum + xyz_loc 
            
            if xy[0] ==0 and xy[1] == 0:
                continue
            else:
                list_of_centers.append(xy)
        #print(len(list_of_centers))
        #Save all the location of a cell
        df = pd.DataFrame(xyz_loc_accum,columns=['y','x','z'])
        f_output = f.split('/')[-1].split('.tif')[0]
        if not os.path.exists(os.path.join(path2Images,'All_Cells_coordinates',f_output)):
            os.makedirs(os.path.join(path2Images,'All_Cells_coordinates',f_output))
        df.to_csv(os.path.join(path2Images,'All_Cells_coordinates',f_output,f_output+'_cell_'+str(i)+'_coordinates.csv'),index=False)

        center_index = np.floor(len(list_of_centers)/2)
        #print(center_index)
        center = list_of_centers[int(center_index)]
        center = tuple([center[0],center[1],int(center_index)])
        #  #print(center)
        List_of_cell_Locations.append(center)
    return List_of_cell_Locations