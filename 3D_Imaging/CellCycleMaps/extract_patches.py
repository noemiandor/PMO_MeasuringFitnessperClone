import os
import sys 
import pandas as pd 
import cv2 
import numpy as np
import tifffile as tifffile
from tqdm import tqdm
from glob import glob
import argparse
import shutil

def normalize8(I):
  mn = I.min()
  mx = I.max()

  mx -= mn

  I = ((I - mn)/mx) * 255
  return I.astype(np.uint8)


#Merge contours. 
def merge_contours(contours):
    # Combine all contours into a single array of points
    if len(contours) > 1:
        all_points = np.vstack(contours)
    else:
        return contours
    # Optional: get the convex hull of the merged contour for a clean outer shape
    merged_contour = cv2.convexHull(all_points)
    # If you want it in a format similar to contours[i], wrap it in a list
    contours_merged = [merged_contour]  
    return contours_merged
#get a cropped version of the cells based on merged contours. 
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

def get_cropped_cell(contours,image_masked,index):
    if len(contours) < 1:
        return 0
    #merged_contours = merge_contours_by_drawing(contours,image_masked[index,:,:])
    if len(contours) == 1:
        x, y, w, h = cv2.boundingRect(contours[0])
    else:
        x, y, w, h = get_global_bounding_box(contours)

    if w * h < 10: # skip small cells. 
        return 0
    elif (y - 10) < 1 or (x - 10) < 1:
        return 0 
    elif (y + h + 10) >= image_masked.shape[1] or (x + w + 10) >= image_masked.shape[2]:
        return 0
    image_extracted = image_masked[index,y - 10:y + h + 10, x - 10:x + w + 10]
    return image_extracted,image_extracted.shape[0],image_extracted.shape[1]

def resize_cropped_images(cropped,w,h):
    cropped_resized = cv2.resize(cropped,(h,w),interpolation=cv2.INTER_CUBIC)
    return cropped_resized

#extract_patchs_3ch(data_dict,slice_mask,cell_id,cell_Cycle)
def extract_patchs_3ch(data_dict,slice_mask,cell_id,cell_Cycle,z,path2Save,FoF):
    # Find unique pixels using numpy
    unique_values = np.unique(slice_mask)
    #print(unique_values)
    nucleus_tif = data_dict['nucleus']
    image_masked = np.zeros_like(slice_mask)
    if 'cyto' in data_dict and 'mito' in data_dict:
        cyto_tif = data_dict['cyto']
        mito_tif = data_dict['mito']
        image_masked[0,:,:] = slice_mask[0,:,:].astype(nucleus_tif.dtype) * nucleus_tif[z,:,:]
        image_masked[1,:,:] = slice_mask[1,:,:].astype(nucleus_tif.dtype) * cyto_tif[z,:,:]
        image_masked[2,:,:] = slice_mask[2,:,:].astype(nucleus_tif.dtype) * mito_tif[z,:,:]
        #tifffile.imwrite('image_masked.tif',image_masked,photometric='minisblack')
        #tifffile.imwrite('nucleus_tif.tif',nucleus_tif)
    else:
        if len(nucleus_tif.shape) == 4:
            nucleus_tif_temp = cv2.cvtColor(nucleus_tif[z,:,:],cv2.COLOR_BGR2GRAY)
            image_masked = slice_mask.astype(nucleus_tif.dtype) * nucleus_tif_temp 
            cyto_tif = nucleus_tif
            mito_tif = nucleus_tif
            #if image_masked.dtype != nucleus_tif.dtype:
            #    image_masked = image_masked.astype(np.float32)
            image_masked = np.expand_dims(image_masked,0)
        else:
            image_masked = slice_mask.astype(nucleus_tif.dtype) * nucleus_tif[z,:,:] 
            cyto_tif = nucleus_tif
            mito_tif = nucleus_tif
            if image_masked.dtype != nucleus_tif.dtype:
                image_masked = image_masked.astype(np.float32)
            image_masked = np.expand_dims(image_masked,0)
    ################################
    cell_mask = (slice_mask > 0)* 255 
    mask_temp = np.zeros(cell_mask.shape, dtype=np.uint8)
    cell_mask = cell_mask.astype(np.uint8)
    try:
        wn, hn, wc, hc, wm, hm = 0, 0, 0, 0, 0, 0
        if 'cyto' in data_dict and 'mito' in data_dict:
            contours_1, _ = cv2.findContours(cell_mask[0,:,:], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_2, _ = cv2.findContours(cell_mask[1,:,:], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_3, _ = cv2.findContours(cell_mask[2,:,:], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #print(len(contours_1),len(contours_2),len(contours_3))

            if len(contours_2) < 1 or len(contours_3) < 1:
                return
            image_extracted_n,wn,hn = get_cropped_cell(contours_1,image_masked,0)
            image_extracted_c,wc,hc = get_cropped_cell(contours_2,image_masked,1)
            image_extracted_m,wm,hm = get_cropped_cell(contours_3,image_masked,2)
            w_max = max([wn,wc,wm])
            h_max = max([hn,hc,hm])
            image_extracted_n = resize_cropped_images(image_extracted_n,w_max,h_max)
            image_extracted_c = resize_cropped_images(image_extracted_c,w_max,h_max)
            image_extracted_m = resize_cropped_images(image_extracted_m,w_max,h_max)

            image_extracted = np.stack([image_extracted_n, image_extracted_c,image_extracted_m])
        else:
            contours_1, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            image_extracted_n = get_cropped_cell(contours_1,image_masked,0)

            image_extracted = image_extracted_n
    
        if not os.path.exists(os.path.join(path2Save,'cellCycle'+str(cell_Cycle),'images')):
            os.makedirs(os.path.join(path2Save,'cellCycle'+str(cell_Cycle),'images'))
        tifffile.imwrite(os.path.join(path2Save,'cellCycle'+str(cell_Cycle),'images',FoF+'_slice'+str(z)+'_cellCycle'+str(cell_Cycle)+'_cell'+str(cell_id)+'.tif'),image_extracted,photometric='minisblack')
    except Exception as e:
        print(f"An error occurred: cell is very small or close to the image edge: {e}")
        #print('Error for cell_id: {} and CellCycle: {}'.format(cell_id,cell_Cycle))
        return
""" 
This function checks for the input image dtype if the code of not of type np.float32
The function convert the images to type of np.float32 bits. 
"""
def check_image_dtype(image):
    if image.dtype != np.float32:
        print('Warning: input image of type {}, but the code require images of type {}'.format(image.dtype,np.float32))
        image.astype(np.float32)
    return image 

# This function finds the count of cells per each slice, and return a dictionary with the index=slice number, and the value is the count 
# Return: max_index, and dictionary of the count of cells per slice. 
def count_cells_per_slice(nucleus):
    slice_count = {}
    max_index = 0
    max_value = 0
    for i in range(0,nucleus.shape[0]):
        slice_image = nucleus[i,:,:]
        unique_values = np.unique(slice_image)
        slice_count[i] = len(unique_values) - 1
        if max_value < (len(unique_values) - 1):
            max_value =  len(unique_values) - 1
            max_index = i 
    return max_index,max_value
     
def generate_mask(compartments_loc,path2Images,cell_id,FoF,cell_Cycle,path2Save,nucleus_mask,mito_mask,cyto_mask):
    image_tif = tifffile.imread(os.path.join(path2Images,'nucleus.p.tif'))
    image_tif = check_image_dtype(image_tif)
    image_tif_shape = image_tif.shape
    Flag_mito = False
    Flag_cyto = False
    nucleus_cell_loc = pd.read_csv(compartments_loc["nucleus"])

    #nucleus_mask = np.zeros((70,1024,1024))
    for x,y,z in zip(nucleus_cell_loc['x'],nucleus_cell_loc['y'],nucleus_cell_loc['z']):
        nucleus_mask[int(z-1),int(y-1),int(x-1)] = int(cell_id)

    if os.path.exists(os.path.join(path2Images,'mito.p.tif')) and compartments_loc["mito"] is not None:
        mito_tif = tifffile.imread(os.path.join(path2Images,'mito.p.tif'))
        mito_tif = check_image_dtype(mito_tif)

        Flag_mito = True
        mito_cell_loc = pd.read_csv(compartments_loc["mito"])
        #print('processing cyto')
        #mito_mask = np.zeros((70,1024,1024))
        for x,y,z in zip(mito_cell_loc['x'],mito_cell_loc['y'],mito_cell_loc['z']):
            mito_mask[int(z-1),int(y-1),int(x-1)] = int(cell_id)

    if os.path.exists(os.path.join(path2Images,'cytoplasm.p.tif')) and compartments_loc["cytoplasm"] is not None:
        cyto_tif = tifffile.imread(os.path.join(path2Images,'cytoplasm.p.tif'))
        cyto_tif = check_image_dtype(cyto_tif)
        Flag_cyto = True 
        cyto_cell_loc = pd.read_csv(compartments_loc["cytoplasm"])
        #print('processing mito')
        #cyto_mask = np.zeros((70,1024,1024))
        for x,y,z in zip(cyto_cell_loc['x'],cyto_cell_loc['y'],cyto_cell_loc['z']):
            cyto_mask[int(z-1),int(y-1),int(x-1)] = int(cell_id)
    return nucleus_mask,mito_mask,cyto_mask

def extract_patches(compartments_loc,path2Images,cell_id,FoF,cell_Cycle,path2Save,upper_slice_limit,lower_slice_limit):
    image_tif = tifffile.imread(os.path.join(path2Images,'nucleus.p.tif'))
    image_tif = check_image_dtype(image_tif)
    image_tif_shape = image_tif.shape
    Flag_mito = False
    Flag_cyto = False
    nucleus_cell_loc = pd.read_csv(compartments_loc["nucleus"])

    nucleus_mask = np.zeros((70,1024,1024))
    for x,y,z in zip(nucleus_cell_loc['x'],nucleus_cell_loc['y'],nucleus_cell_loc['z']):
        nucleus_mask[int(z-1),int(y-1),int(x-1)] = int(cell_id)

    if os.path.exists(os.path.join(path2Images,'mito.p.tif')) and compartments_loc["mito"] is not None:
        mito_tif = tifffile.imread(os.path.join(path2Images,'mito.p.tif'))
        mito_tif = check_image_dtype(mito_tif)

        Flag_mito = True
        mito_cell_loc = pd.read_csv(compartments_loc["mito"])
        #print('processing cyto')
        mito_mask = np.zeros((70,1024,1024))
        for x,y,z in zip(mito_cell_loc['x'],mito_cell_loc['y'],mito_cell_loc['z']):
            mito_mask[int(z-1),int(y-1),int(x-1)] = int(cell_id)

    if os.path.exists(os.path.join(path2Images,'cytoplasm.p.tif')) and compartments_loc["cytoplasm"] is not None:
        cyto_tif = tifffile.imread(os.path.join(path2Images,'cytoplasm.p.tif'))
        cyto_tif = check_image_dtype(cyto_tif)
        Flag_cyto = True 
        cyto_cell_loc = pd.read_csv(compartments_loc["cytoplasm"])
        #print('processing mito')
        cyto_mask = np.zeros((70,1024,1024))
        for x,y,z in zip(cyto_cell_loc['x'],cyto_cell_loc['y'],cyto_cell_loc['z']):
            cyto_mask[int(z-1),int(y-1),int(x-1)] = int(cell_id)
    # get max index and dict of slice:count for each slice.
    for z in range(0,nucleus_mask.shape[0]):
        if z <= upper_slice_limit or z > lower_slice_limit: # Only taking the mid slices. 
            continue
        mask_n = nucleus_mask[z,:,:]
        data_dict = {'nucleus':image_tif}
        if not mask_n.max() > 0:
            continue
        if Flag_mito and Flag_cyto:
            mask_c = cyto_mask[z,:,:]
            mask_m = mito_mask[z,:,:]
            data_dict = {'nucleus':image_tif,'cyto':cyto_tif,'mito':mito_tif}
            slice_mask = np.stack([mask_n,mask_m,mask_c])
        else:
            data_dict = {'nucleus':image_tif}
            slice_mask = mask_n
        extract_patchs_3ch(data_dict,slice_mask,cell_id,cell_Cycle,z,path2Save,FoF)

""" 
def extract_patches(path2Images,cell_id,cell_Cycle,compartments_loc,nucleus_mask,mito_mask,cyto_mask,FoF,path2Save):
    
    image_tif = tifffile.imread(os.path.join(path2Images,'nucleus.p.tif'))
    image_tif = check_image_dtype(image_tif)
    image_tif_shape = image_tif.shape
    Flag_mito = False
    Flag_cyto = False

    if os.path.exists(os.path.join(path2Images,'mito.p.tif')) and compartments_loc["mito"] is not None:
        mito_tif = tifffile.imread(os.path.join(path2Images,'mito.p.tif'))
        mito_tif = check_image_dtype(mito_tif)
        Flag_mito = True

    if os.path.exists(os.path.join(path2Images,'cytoplasm.p.tif')) and compartments_loc["cytoplasm"] is not None:
        cyto_tif = tifffile.imread(os.path.join(path2Images,'cytoplasm.p.tif'))
        cyto_tif = check_image_dtype(cyto_tif)
        Flag_cyto = True 
    
    # get max index and dict of slice:count for each slice.
    max_index, max_value = count_cells_per_slice(nucleus_mask)
    upper_slice_limit = max_index - 5 
    lower_slice_limit = max_index + 5 
    for z in range(0,nucleus_mask.shape[0]):
        if z <= upper_slice_limit or z > lower_slice_limit: # Only taking the mid slices. 
            continue
        #tifffile.imwrite('./nucleus_mask.tif',nucleus_mask[z,:,:])
        #print(np.unique(nucleus_mask[z,:,:]))
        for i in np.unique(nucleus_mask[z,:,:]):
            if i == 0:
                continue
            mask_n = nucleus_mask[z,:,:]
            mask_temp = (mask_n == i)* 1 
            mask_n = mask_n * mask_temp
            #tifffile.imwrite('./nucleus_mask_singleCell.tif',mask_n)
            data_dict = {'nucleus':image_tif}
            if not mask_n.max() > 0:
                continue
            if Flag_mito and Flag_cyto:
                mask_c = cyto_mask[z,:,:]
                mask_m = mito_mask[z,:,:]

                mask_temp = (mask_c == i) * 1
                mask_c = mask_c * mask_temp
                mask_temp = (mask_m == i) * 1
                mask_m = mask_m * mask_temp

                data_dict = {'nucleus':image_tif,'cyto':cyto_tif,'mito':mito_tif}
                slice_mask = np.stack([mask_n,mask_m,mask_c])
            else:
                data_dict = {'nucleus':image_tif}
                slice_mask = mask_n
            extract_patchs_3ch(data_dict,slice_mask,int(i),cell_Cycle,z,path2Save,FoF)
    return max_index,max_value
"""
"""
This function loops over the FoFs and for each csv file, it generate the masks by calling generate_masks()
"""

def process_FoF_one_cell_per_time(path2FoF,path2CellMasks,path2Save,upper_slice_limit,lower_slice_limit):

    csv_files = glob(os.path.join(path2CellMasks, "nucleus.p*.csv"))
    path2Images = os.path.join(path2FoF)
    nucelus = np.zeros((70,1024,1024))
    mito = np.zeros((70,1024,1024))
    cyto = np.zeros((70,1024,1024))
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
            #print('Serving Cell_id: {}  CellCycle: {}'.format(cell_id,cell_Cycle))
            # print('cell_Cycle:'.format(cell_Cycle))
            nucelus,mito,cyto = generate_mask(compartments_loc,path2Images,cell_id,"FoF",cell_Cycle,path2Save,nucelus,mito,cyto)
            extract_patches(compartments_loc,path2Images,cell_id,"FoF",cell_Cycle,path2Save,upper_slice_limit,lower_slice_limit)
def process_FoF(path2FoF,path2CellMasks,path2Save):
    # deleting the path2Save if it is already exists. 
    try:
        if os.path.exists(path2Save):
            shutil.rmtree(path2Save)
    except:
        print('Error in removing previous output dir, skipping ...')
    # for FoF in os.listdir(path2FoF):
    #     if not os.path.isdir(os.path.join(path2FoF,FoF)):
    #         continue 
    csv_files = glob(os.path.join(path2CellMasks, "nucleus.p*.csv"))
    path2Images = os.path.join(path2FoF)
    nucelus = np.zeros((70,1024,1024))
    mito = np.zeros((70,1024,1024))
    cyto = np.zeros((70,1024,1024))
    print('Counting number of cells per slice to find slice with max cells...')
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
            #print('Serving Cell_id: {}  CellCycle: {}'.format(cell_id,cell_Cycle))
            # print('cell_Cycle:'.format(cell_Cycle))
            nucelus,mito,cyto = generate_mask(compartments_loc,path2Images,cell_id,"FoF",cell_Cycle,path2Save,nucelus,mito,cyto)

    max_index, max_value = count_cells_per_slice(nucelus)
    upper_slice_limit = max_index - 5 
    lower_slice_limit = max_index + 5
    process_FoF_one_cell_per_time(path2FoF,path2CellMasks,path2Save,upper_slice_limit,lower_slice_limit) 
    
    #max_index, max_value = extract_patches(path2Images,cell_id,cell_Cycle,compartments_loc,nucelus,mito,cyto,"FoF",path2Save)
    return max_index,max_value

if __name__ == '__main__':
    #path2FoF="/Volumes/Expansion/Collaboration/Moffitt_Noemi/BioinformaticsPaper/data/NCI-N87-Dataset2/A04_CellposeOutput/FoF1_231005_fluorescent.nucleus/"
    #path2CellMasks="/Volumes/Expansion/Collaboration/Moffitt_Noemi/BioinformaticsPaper/data/NCI-N87-Dataset2/A06_multiSignals_Linked/FoF1_231005_fluorescent.nucleus/"
    #path2Save="/Volumes/Expansion/Collaboration/Moffitt_Noemi/BioinformaticsPaper/data/NCI-N87-Dataset2/A07_patches/FoF1_231005_fluorescent.nucleus/"
    parser = argparse.ArgumentParser(description="Process Fields of View (FoF)")
    parser.add_argument('path2FoF', type=str, help="Path to the folder of interest (FoF)")
    parser.add_argument('path2CellMasks', type=str, help="Path to the Cell_Masks folder")
    parser.add_argument('path2Save', type=str, help="Path to save the output Cell_patches")
    args = parser.parse_args()
    path2FoF = args.path2FoF
    path2CellMasks = args.path2CellMasks
    path2Save = args.path2Save
    print(f"path2FoF: {path2FoF}")
    print(f"path2CellMasks: {path2CellMasks}")
    print(f"path2Save: {path2Save}")
    max_index,max_value = process_FoF(path2FoF, path2CellMasks, path2Save)
