# Cell Cycle Prediction with CNN
These scripts predict the cell cycle assignment of specially formatted imaging data where each file contains a 3D brightfield image of a cell  

## 1) Install requirements (tested with conda package manager, python 3.10)
```bash
conda env create -f environment.yml
```
or run in docker to ensure stable environment
```bash
docker run -it -v DATADIR:/data ghcr.io/zsdqui/cellcycle_cnn:v0.1
```
DATADIR is the path containing the images fields of view with each image lables inferred from the file name i.e.  
prefix_CellCycle[label]_cellID.tiff  
The label is the character before the final underscore  

## 2) (optional) train model:
```bash
python runModel.py -m train -epochs N_epochs -dataDir DATADIR -epochs NUM_EPOCHS -arch [custom_cnn/custom_resnet] -exp EXP_NAME
```
At the first run, the script will split the dataset into train/val/test splits and corresponding csvs with fovs linked to cell cycle labels will be generated.    

## 3) test model/generate bottleneck features (reproduce results in Alahmari et al.):
```bash
python runModel.py -m test -dataDir DATADIR -testCSV ./models/test_1ch.csv -arch custom_cnn -exp CNN_exp1_merged_data_ch1
```
Note: when using this to infer results on a different dataset, replace the test_1ch.csv with your own csv file.
A new results folder should be created with accuracy results and features csv file, containing a true cell cycle label, predicted label, and 244 features

