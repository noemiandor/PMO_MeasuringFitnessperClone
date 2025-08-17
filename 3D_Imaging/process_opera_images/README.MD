# How to run the script.

First, run the following command to learn about the input arguments

```shell
python process_images.py -h
```
usage: process_images.py [-h] -dir DIR -type TYPE [-saveDir SAVEDIR] [-saveOmeTiff SAVEOMETIFF] [--only_renameOpera]
                         [--only_createOmeTiff]

Getting argument for running renaming and creating OME tiff script for Opera Images

optional arguments:
  -h, --help            show this help message and exit
  -dir DIR              Path to the main folder of the data
  -type TYPE            either nucleus or mito or cytoplasm
  -saveDir SAVEDIR      Optional: path to dir where converted images will be saved
  -saveOmeTiff SAVEOMETIFF
                        Optional: path to dir where OME tiff files will be saved
  --only_renameOpera    Optional: only running the script for renaming the Opera images
  --only_createOmeTiff  Optional: only running the script for creating opera files

#Note the images must be inside a folder called images as follows: <br/>
B01_OperaPhenix_230804 <br/>
└── images

# To run the script for renaming Opera Images and Creating OME TIFF files
```shell
python process_images.py -dir ../rawImages/B01_OperaPhenix_230804/ -type nucleus -saveDir . -saveOmeTiff ./opera_ome_tif_check2 
```
# To run the script for renaming Opera Images use the argument 
```shell
python process_images.py -dir ../rawImages/B01_OperaPhenix_230804/ -type nucleus -saveDir . --only_renameOpera
```
# To run the script for creating OME TIFF images use the argument 
```shell
python process_images.py -dir ./B01_OperaPhenixTrain_230804/ -type nucleus -saveDir . -saveOmeTiff ./opera_ome_tif_check2 --only_createOmeTiff
```


