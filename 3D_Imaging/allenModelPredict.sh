#!/bin/sh

FoF=$1
organelle=$2
cellLine='NCI-N87'
if [ "$#" -eq 3 ]; then
   cellLine=$3
fi

A02="/Volumes/Expansion/Collaboration/Moffitt_Noemi/BioinformaticsPaper/$cellLine/A02_ometiffconversion"
A03="/Volumes/Expansion/Collaboration/Moffitt_Noemi/BioinformaticsPaper/$cellLine/A03_allenModel/"$FoF
json=$A02/$FoF.$organelle".json"
csv=$A02/$FoF".csv"
target=${FoF##*.}

# Prepare json and csv file
cp /Users/saeedalahmari/Downloads/BioInformaticsPaper/PMO_MeasuringFitnessperClone/3D_Imaging/allenModel/Template.json $json
cp /Users/saeedalahmari/Downloads/BioInformaticsPaper/PMO_MeasuringFitnessperClone/3D_Imaging/allenModel/Template.csv $csv
sed -i '' "s/organelleXX/$organelle/g" "$json"
sed -i '' "s/FoFXX/$FoF/g" "$json"
sed -i '' "s/FoFXX/$FoF/g" "$csv"
sed -i '' "s/cellLineXX/$cellLine/g" "$json"
sed -i '' "s/cellLineXX/$cellLine/g" "$csv"

# Run the predict function. The test images path must be present in $csv file. 
fnet predict --json $json --gpu_ids 1

# Rename Allen Model output
mv $A03/tifs/0_prediction_c0.$organelle".p.tif" $A03/$organelle".p.tif" 
# Only if input includes fluorescence signal:
if [ "$FoF" != "$target" ]; then
	mv $A03/tifs/0_target.tif $A03/$target".t.tif"
fi

# Writing permissions to output FoF folder
chmod -R a+w $A03
chmod a+w $A03/tifs/*
chmod a+w $A03/*
