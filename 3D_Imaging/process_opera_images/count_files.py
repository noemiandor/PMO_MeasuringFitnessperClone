import os
import sys


path2Dataset = '../A01_OperaPhenix_230804'


count_signal_target = 0
count_target = 0
count_signal = 0

for folder in os.listdir(path2Dataset):
    if folder.startswith('.'):
        continue
    if os.path.exists(os.path.join(path2Dataset,folder,'nucleus.t.tiff')) and os.path.exists(os.path.join(path2Dataset,folder,'nucleus.s.tiff')):
        count_signal_target = count_signal_target + 1
    elif os.path.exists(os.path.join(path2Dataset,folder,'nucleus.t.tiff')):
        count_target = count_target + 1
    elif os.path.exits(os.path.join(path2Dataset,folder,'nucleus.s.tiff')):
        count_signal = count_signal + 1

print('total number of folders with signal and targets files together are {}'.format(count_signal_target))
print('total number of folders with signal only files is {}'.format(count_signal))
print('total number of folders with target only files is {}'.format(count_target))

