# implementation of the paper Robust classification of cell cycle phase and biological feature extraction by image-based deep learning
# https://github.com/dtakao-lab/Nagao2020/blob/master/Code/4_FunctionalModel.ipynb
from keras.models import Model
from keras.layers import Input, Conv2D,MaxPooling2D,Dense, Flatten,Dropout,BatchNormalization
from tensorflow.keras import regularizers
import tensorflow.keras as k

# architecture of CNN
# we refer to this method as custom_cnn 
def buildModel(neurons,drop,hidden_layers,nb_classes):  
  

    neurons=int(neurons)
    hidden_layers=int(hidden_layers)

    # input layer
    inputs = Input(shape=(64,64,3))

    x = Conv2D(neurons, (3, 3), padding='same', activation='relu',kernel_regularizer=regularizers.l2(0.01))(inputs) #Saeed: Added l2 regularizer to overcome overfitting issue
    x = BatchNormalization()(x) # Saeed: Added this to overcome overfitting problems
    x = MaxPooling2D((2, 2), padding='same')(x)
    # hidden layers
    #print('Number of hidden layers is {}'.format(hidden_layers))
    if hidden_layers !=0:
        for i in range(1,hidden_layers+1):
            #print(i)
            x = Conv2D(neurons*(2**i), (3, 3), padding='same', activation='relu',kernel_regularizer=regularizers.l2(0.01))(x) #Saeed: Added l2 regularizer to overcome overfitting issue
            x = BatchNormalization()(x) # Saeed: Added this to overcome overfitting problem.
            #print('conv hidden {}'.format(x.shape))
            x = MaxPooling2D((2, 2), padding='same')(x)
            #print('maxpool hidden {}'.format(x.shape))
    x = Flatten()(x)
    x = Dense(neurons*(2**(hidden_layers+1)), activation='relu',kernel_regularizer=regularizers.l2(0.01))(x) #Saeed: Added l2 regularizer to overcome overfitting issue
    #x = Dropout(drop)(x)
    x = BatchNormalization()(x)
    # output
    predictions = Dense(nb_classes, activation='softmax')(x)

    # modeling
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=k.optimizers.RMSprop(learning_rate=0.001),#optimizer='rmsprop'
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
    return model

'''
# hyperparameters
drop = 0.3 # dropout rate
hidden_layers = 2 # number of hidden convolutional/max pooling layer sets
batch = 89 # batch size
neurons = 28 # number of convolution kernels

'''
