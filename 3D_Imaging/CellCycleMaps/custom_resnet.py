# ResNet 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras as k
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import regularizers
def res_block1(x, filters):
    #print('Input',x.shape)
    #https://github.com/alinarw/ResNet/blob/master/ResNet.ipynb
    bn1 = layers.BatchNormalization()(x)
    act1 = layers.Activation('relu')(bn1)
    #act1 = keras.layers.LeakyReLU(0.3)(bn1)
    conv1 = layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same')(act1)
    
    #print('conv1.shape', conv1.shape)
    bn2 = layers.BatchNormalization()(conv1)
    act2 = layers.Activation('relu')(bn2)
    #act2 = keras.layers.LeakyReLU(0.3)(bn2)
    conv2 = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(act2)
  
    #print('conv2.shape', conv2.shape)
    bn3 = layers.BatchNormalization()(conv2)
    act3 = layers.Activation('relu')(bn3)
    #act3 = keras.layers.LeakyReLU(0.3)(bn3)
    conv3 = layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same')(act3)
    #x = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='valid')(x)
    
    
    residual = layers.Conv2D(1, (1, 1), strides=(1, 1))(x)
    #print('residual.shape', x.shape)
    out = layers.Add()([conv3, residual])
    return out

def res_block2(x, filters):
    #print('Input',x.shape)
    #https://github.com/alinarw/ResNet/blob/master/ResNet.ipynb
    bn1 = layers.BatchNormalization()(x)
    act1 = layers.Activation('relu')(bn1)
    #act1 = keras.layers.LeakyReLU(0.3)(bn1)
    conv1 = layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same')(act1)
    
    #print('conv1.shape', conv1.shape)
    bn2 = layers.BatchNormalization()(conv1)
    act2 = layers.Activation('relu')(bn2)
    #act2 = keras.layers.LeakyReLU(0.3)(bn2)
    conv2 = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(act2)
  
    #print('conv2.shape', conv2.shape)
    bn3 = layers.BatchNormalization()(conv2)
    act3 = layers.Activation('relu')(bn3)
    #act3 = keras.layers.LeakyReLU(0.3)(bn3)
    conv3 = layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same')(act3)
    #x = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='valid')(x)
    
    #print('residual.shape', x.shape)
    out = layers.Add()([conv3, x])
    return out

def create_resnet(input_shape=(64, 64, 3), num_classes=4):
  """Creates a ResNet model.

  Args:
    input_shape: Shape of the input tensor.
    num_classes: Number of classes for the classification task.

  Returns:
    A Keras model.
  """
  input_tensor = keras.Input(shape=input_shape)
  x = layers.BatchNormalization()(input_tensor)
  x = layers.Conv2D(32, (3, 3), strides=1, padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation('relu')(x)
  x = layers.MaxPooling2D((3, 3), strides=1, padding='same')(x)
  
  # Stack residual modules
  x = res_block1(x, 32)
  x = res_block2(x, 32)
  x = res_block2(x, 128)
  
  x = res_block2(x, 64)
  x = res_block2(x, 64)
  x = res_block2(x, 256)
 

  x = res_block2(x, 128)
  x = res_block2(x, 128)
  x = res_block2(x, 512)

  x = layers.GlobalAveragePooling2D()(x)
  #x = layers.AveragePooling2D((3, 3), strides=1, padding='valid')(x)
  #x = layers.BatchNormalization()(x)
  #x = layers.Flatten()(x)
  #x = layers.Dropout(0.3)(x)
  
  out = layers.Dense(num_classes, activation='softmax',name='output')(x)
  #Model with input and output.
  model = keras.Model(inputs=input_tensor, outputs=out)
  model.compile(loss=categorical_crossentropy,optimizer=k.optimizers.Adam(learning_rate=0.01),metrics=['accuracy'])
  return model


def create_resnet_small(input_shape=(64, 64, 3), num_classes=4):
  """Creates a ResNet model.

  Args:
    input_shape: Shape of the input tensor.
    num_classes: Number of classes for the classification task.

  Returns:
    A Keras model.
  """
  input_tensor = keras.Input(shape=input_shape)
  x = layers.BatchNormalization()(input_tensor)
  x = layers.Conv2D(32, (3, 3), strides=1, padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation('relu')(x)
  x = layers.MaxPooling2D((3, 3), strides=1, padding='same')(x)

  # Stack residual modules
  x = res_block1(x, 32)
  x = res_block2(x, 32)
  x = res_block2(x, 128)

  x = res_block2(x, 64)
  x = res_block2(x, 64)
  x = res_block2(x, 256)


  x = res_block2(x, 128)
  x = res_block2(x, 128)
  x = res_block2(x, 512)

  #x = layers.GlobalAveragePooling2D()(x)
  x = layers.AveragePooling2D((3, 3), strides=1, padding='valid')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Flatten()(x)
  #x = layers.Dense(128, activation='relu')(x)
  x = layers.Dropout(0.3)(x)
  out = layers.Dense(num_classes, activation='softmax',name='output')(x)
  #Model with input and output.
  model = keras.Model(inputs=input_tensor, outputs=out)
  model.compile(loss=categorical_crossentropy,optimizer=k.optimizers.RMSprop(learning_rate=0.001),metrics=['accuracy'])
  return model
# Create the ResNet model
#model = create_resnet()

# Print the model summary
#model.summary()
