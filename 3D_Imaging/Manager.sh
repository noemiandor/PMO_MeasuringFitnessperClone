code='/Users/saeedalahmari/Downloads/BioInformaticsPaper/PMO_MeasuringFitnessperClone/3D_Imaging'
root='/Volumes/Expansion/Collaboration/Moffitt_Noemi/BioinformaticsPaper/data/NCI-N87'
A03=$root/A03_allenModel/

# ## Renome data
# cd /raid/crdlab/ix2/andrew/PMO_MeasuringFitnessperClone/3D_Imaging/Project_Cyto/Data_Images/210818_Cyto/Images
# renameRawData=$code/renameRawData.sh
# date='210928'
# organelle='Mito'
# source='t'
# channel='00' #fluorescense
# $renameRawData $date $organelle $source $channel
# source='s'
# channel='01' #brightfield
# $renameRawData $date $organelle $source $channel

#uncomment the following section to run it on a single FoF
"""
##################
####Single FoF####

# FoF='FoF1001_220407_brightfield'
FoF='FoF9_211110_fluorescent.nucleus'
# FoF='FoF13_220228_fluorescent.cytoplasm'

## @TODO ome.tiff conversion

## AllenModel
$code/allenModelPredict.sh $FoF nucleus &&
$code/allenModelPredict.sh $FoF mito 

## CellPose
$code/cellPoseSegment.sh $FoF
"""
##################
#### All FoFs ####
cd /raid/crdlab/ix1/Projects/M005_MeasuringFitnessPerClone_2019/data/GastricCancerCLs/3Dbrightfield/NCI-N87/A02_ometiffconversion

## ome.tiff conversion
# python $code/stack2ometiff_conversion.py

declare -a dates=(231005 240918) #211104 210818 220228 210928 211027 
for date in "${dates[@]}"; do
  f=`ls -d *$date*ome.tif`

  ## AllenModel
  for x in $f; do 
    FoF="$(basename -- $x .ome.tif)"
    echo $FoF
    $code/allenModelPredict.sh $FoF nucleus
    $code/allenModelPredict.sh $FoF mito
    $code/allenModelPredict.sh $FoF cytoplasm
    # $code/cellPoseSegment.sh $FoF
  done
done


## CellPose
while read FoF; do
   ls -d $A03/$FoF
   $code/cellPoseSegment.sh $FoF
done <~/Downloads/fois_nuc

# for x in $f; do
#   FoF="$(basename -- $x .ome.tif)"
#   $code/cellPoseSegment.sh $FoF
# done
