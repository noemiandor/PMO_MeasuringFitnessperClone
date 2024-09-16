import pandas as pd
import os
import sys
import glob

path2Data = '/data/saeed3/Noemi_Moffitt/renameOperaImages/B01_Opera_OME_TIFF_train'

files = glob.glob(path2Data+'/*.tiff')
df = pd.DataFrame()
df['path_tiff'] = files
df['channel_signal'] = [0 for i in files]
df['channel_target'] = [1 for i in files]


#train = df.sample(frac=0.8,random_state=200)
#test = df.drop(train.index)
train = df
print(train.shape)
#print(test.shape)

train.to_csv('./n_train_opera_B01_finetune.csv',index=False)
#test.to_csv('./n_test_opera_finetune_new_test2.csv',index=False)
