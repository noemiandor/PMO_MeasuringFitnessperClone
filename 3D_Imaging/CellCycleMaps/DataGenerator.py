import numpy as np
import tensorflow.keras as keras 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import tensorflow.keras as K
import sys
import os
import tensorflow as tf
import tifffile as tifffile
from pre_augment import AugmentImages
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,data_path,samples,num_classes, batch_size=32, dim=(64,64), n_channels=1, shuffle=True,single_channel=False, augment=False,n_labels=1):
        'Initialization'
        self.data_path = data_path
        self.dim = dim
        self.augment = augment
        self.batch_size = batch_size
        self.samples = samples
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.n_labels= n_labels
        self.num_classes = num_classes
        self.single_channel = single_channel
        self.indexes = range(0,self.samples.shape[0])
        self.apply_augmentation = AugmentImages(self.data_path,self.data_path + '-aug')
        self.augmentor = ImageDataGenerator(
            rotation_range=90,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            rescale=1./255,
            fill_mode='nearest')
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.samples.shape[0] / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.samples.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def random_augmentation(self,image,y1,img_name):
        #print(image.shape) 
        aug_list = []
        aug_list.append(image)
        #Rotation        
        for i in range(5,360,5):
            #print('rotation angle {}'.format(i))
            aug_image = self.augmentor.apply_transform(image,{'theta':i})
            aug_list.append(aug_image)
        for i in range(20): # generate 10x random images. 
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
        return aug_list,y1_list,
    

    def save_augmented_images(self,listOfImages):
        if os.path.exists(os.path.join('.','savedImages')):
            os.makedirs(os.path.join('.','savedImages'))
        for i,img in enumerate(listOfImages):
            cv2.imwrite(os.path.join('.','savedImages',str(i)+'.png'),img)

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
    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # Initialise X_train and y_train arrays for this batch
        X_train = []
        y1_train = []
        # For each example
        for ind in indexes: #interpolation=cv2.INTER_CUBIC
            # Load image (X) and label (y)
            #print(batch_sample)
            img_name = self.samples.iloc[ind]['image']
            #print(img_name)
            label1 = self.samples.iloc[ind]['label']
            if img_name.endswith('.png') or img_name.endswith('.jpg'): 
                img = cv2.imread(os.path.join(self.data_path,img_name),-1)
            else:
                img = tifffile.imread(os.path.join(self.data_path,img_name))
            #Just to check if three channels impact the performance 
            #img_temp = img[0,:,:]
            #img = img_temp
            # apply any kind of preprocessing
            if self.single_channel and img.shape[0] == 3:
                img = img[0,:,:] # take only the nuclues
            if not img.shape[0] == 3:
                img = cv2.resize(img,(64,64),interpolation=cv2.INTER_CUBIC)
            else:
                img_resized = np.zeros((3,64,64))
                img_resized[0,:,:] = cv2.resize(img[0,:,:],(64,64),interpolation=cv2.INTER_CUBIC)
                img_resized[1,:,:] = cv2.resize(img[1,:,:],(64,64),interpolation=cv2.INTER_CUBIC)
                img_resized[2,:,:] = cv2.resize(img[2,:,:],(64,64),interpolation=cv2.INTER_CUBIC)
                img_transposed = img_resized.transpose(1, 2, 0)
                img = img_transposed
            if 'cell' in img_name and self.augment:
                label2 = 0
                if not img.dtype == np.uint8:
                    img = self.normalize8(img)
                if not img.shape[-1] == 3:
                    #img  = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    img = np.stack([img,img,img], axis = -1)
                #aug_imgs,y1 =  self.random_augmentation(img,label1,img_name)
                aug_imgs,y1 =  self.apply_augmentation.random_augmentation(img,label1,img_name)
                X_train = X_train + aug_imgs 
                y1_train = y1_train + y1 
                #y2_train = y2_train + y2
            else:
                if not img.dtype == np.uint8:
                    img = self.normalize8(img)
                if not img.shape[-1] == 3:
                    img = np.stack([img,img,img],axis = -1)
                    #img  = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                X_train.append(img)
                y1_train.append(label1)

        # Make sure they're numpy arrays (as opposed to lists)
        X_train = np.array(X_train)
        X_train = X_train.astype(np.float32)
        X_train = X_train / 255.0
        #print(X_train.max())
        #print(X_train.min())
        #X_train = X_train.astype(np.float32)
        y1_train = np.array(y1_train)
        # The generator-y part: yield the next training batch
        #tf.keras.utils.to_categorical( y, num_classes=None, dtype='float32')
        return X_train, K.utils.to_categorical(y1_train,self.num_classes)
