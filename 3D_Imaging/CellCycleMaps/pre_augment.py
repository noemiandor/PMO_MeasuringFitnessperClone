import os
import cv2
import tifffile 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd

class AugmentImages():
    def __init__(self,path2Data,path2Save):
        self.path2Data = path2Data 
        self.path2Save = path2Save
        self.augmentor = ImageDataGenerator(
            rotation_range=90,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            rescale=1./255,
            fill_mode='nearest')
    def iterate(self):
        for folder in os.listdir(self.path2Data):
            if os.path.isfile(os.path.join(self.path2Data, folder)):
                continue 
            print('processing folder {}'.format(folder))
            for file in tqdm(os.listdir(os.path.join(self.path2Data,folder,'images_padded'))):
                if not file.endswith('.tif'):
                    continue 
                try:
                    image = tifffile.imread(os.path.join(self.path2Data,folder,'images_padded',file))
                    #image = self.normalize8(image)
                    #self.check_cell_roundness(image)
                    label = folder
                    image_name = file
                    #print(image_name)
                    image_rgb = np.stack([image,image,image], axis = -1)
                    aug_image,labels = self.random_augmentation(image_rgb,label,image_name)
                    self.save_aug_images(aug_image,labels,image_name)   
                except Exception as e:
                    print("Error in image {}  {}.".format(file,e))

    def iterate_df(self):
        print('Please wait... applying pre-augmentation')
        df_train = pd.read_csv(os.path.join(self.path2Data,'train_data.csv'))
        for filePath,label in tqdm(zip(df_train['image'].tolist(),df_train['label'].tolist())):
            try:
                #print(filePath)
                if filePath.endswith('.png') or filePath.endswith('jpg'):
                    image = cv2.imread(os.path.join(self.path2Data,filePath))
                elif filePath.endswith('tif') or filePath.endwith('tiff'):
                    image = tifffile.imread(os.path.join(self.path2Data,filePath))
                else:
                    print('unexpected image extention')
                image_name = os.path.basename(filePath)
                dir_name = os.path.dirname(filePath)
                label_name = image_name.split('_')[-2]
                if image.shape[0] == 3:
                    image_rgb = image.transpose(1 , 2 , 0)
                else:
                    image_rgb = np.stack([image,image,image], axis = -1)
                aug_image,labels = self.random_augmentation(image_rgb,label_name,image_name)
                self.save_aug_images(aug_image,labels,dir_name,image_name)
            except Exception as e:
                print("Error in image {}  {}.".format(image_name,e))


    def normalize8(self,I):
        #print(I.dtype)
        #print(type(I))
        if isinstance(I, tf.Tensor):
            I = I.numpy()
        mn = I.min()
        mx = I.max()
        mx -= mn
        I = ((I - mn)/mx) * 255
        return I.astype(np.uint8)

    def save_aug_images(self,aug_images,labels,dir_name,image_name=None):
        index = 0
        for image,folder in zip(aug_images,labels):
            if not os.path.exists(os.path.join(self.path2Save,dir_name)):
                os.makedirs(os.path.join(self.path2Save,dir_name))
            aug_image_name = image_name.split('.tif')[0] + '_aug_'+str(index)+'.tif'
            image = self.normalize8(image)
            image = image.transpose(2 , 0 , 1)
            #print(image.shape)

            tifffile.imwrite(os.path.join(self.path2Save,dir_name,aug_image_name),image,photometric='minisblack')
            index = index + 1
    def random_augmentation(self,image,y1,img_name):
            #print(image.shape) 
            aug_list = []
            aug_list.append(image)
            #Rotation  
            for i in range(5,360,5):
                #print('rotation angle {}'.format(i))
                aug_image = self.augmentor.apply_transform(image,{'theta':i})
                aug_list.append(aug_image)
            for i in range(40): # generate 40x random images. 
                aug_image = tf.image.random_brightness(image, max_delta=0.7)  # Random brightness
                aug_list.append(aug_image)
                aug_image = tf.image.random_contrast(image, lower=0.8, upper=2.2)  # Random contrast
                aug_list.append(aug_image)
            #flipping 
            aug_image = self.augmentor.apply_transform(image,{'flip_horizontal':True})
            aug_list.append(aug_image)

            aug_image = self.augmentor.apply_transform(image,{'flip_vertical':True})
            aug_list.append(aug_image)
            
            y1_list = [y1 for i in range(0,len(aug_list))]
            return aug_list,y1_list

#path2Data = '/Users/saeedalahmari/Library/CloudStorage/GoogleDrive-aalahmari.saeed@gmail.com/My Drive/Research/Collaboration/Noemi_Moffitt/BioInformaticsPaper/data'

#dpath2Save = '/Users/saeedalahmari/Library/CloudStorage/GoogleDrive-aalahmari.saeed@gmail.com/My Drive/Research/Collaboration/Noemi_Moffitt/BioInformaticsPaper/data-aug'


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Augment the images before training NN')
    argparser.add_argument('-dataDir',required=True,type=str,default ='./data2')
    argparser.add_argument('-saveDir',required=True,type=str,default ='./data-aug2')
    args = argparser.parse_args()
    Image_Augmentor = AugmentImages(args.dataDir,args.saveDir)
    Image_Augmentor.iterate()

