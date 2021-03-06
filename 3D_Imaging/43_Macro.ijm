run("Image Sequence...", "open=[C:/Users/tablet/Desktop/moffit document/Moffitt reserch/Moffitt dataset/Sequence/Sequence.tif] starting=40 convert sort");
selectWindow("Sequence");
close();
setOption("ScaleConversions", true);
run("Image Sequence...", "open=[C:/Users/tablet/Desktop/moffit document/Moffitt reserch/Moffitt dataset/Sequence/0_prediction_c0.model0005.tif]");
setOption("ScaleConversions", true);
selectWindow("Sequence");
run("Duplicate...", "use");
run("Median...", "radius=2");
run("Subtract Background...");
run("Duplicate...", " ");
run("Invert");
run("Subtract Background...");
//run("Brightness/Contrast...");
run("Enhance Contrast", "saturated=0.35");
run("Enhance Contrast", "saturated=0.35");
run("Enhance Contrast", "saturated=0.35");
resetMinAndMax();
run("Enhance Contrast", "saturated=0.35");
run("Enhance Contrast", "saturated=0.35");
run("Enhance Contrast", "saturated=0.35");
run("Enhance Contrast", "saturated=0.35");
run("Apply LUT");
run("Close");
setAutoThreshold("Default");
//run("Threshold...");
setAutoThreshold("Huang");
resetThreshold();
setAutoThreshold("Huang");
//setThreshold(0, 148);
setOption("BlackBackground", true);
run("Convert to Mask");
run("Close");
selectWindow("4_target0023");
run("Remove Outliers...");
run("Despeckle");
run("Gamma...");
run("Enhance Contrast...", "saturated=0.3 normalize equalize");
run("Remove Outliers...");
run("Image Calculator...");
run("Subtract Background...", "rolling=35");
run("Despeckle");
run("Remove Outliers...");
run("Duplicate...", " ");
run("Remove Outliers...");
run("Remove Outliers...", "radius=70 threshold=5 which=Bright");
run("Smooth");
run("Convert to Mask");
run("Open");
run("Open");
run("Fill Holes");
run("Undo");
run("Erode");
run("Duplicate...", " ");
run("Open");
run("Fill Holes");
selectWindow("4_target0023-1");
selectWindow("4_target0023-3");
run("Dilate");
selectWindow("Sequence");
run("Duplicate...", "use");
selectWindow("4_target0023-3");
selectWindow("4_target0023");
imageCalculator("Add create", "4_target0023","4_target0023-3");
selectWindow("Result of 4_target0023");
saveAs("Tiff", "C:/Users/tablet/Desktop/moffit document/Moffitt reserch/Moffitt dataset/Sequence/80percent.tif");
selectWindow("4_target0023");
selectWindow("4_target0023-3");
close();
saveAs("Tiff", "C:/Users/tablet/Desktop/moffit document/Moffitt reserch/Moffitt dataset/Sequence/mask43.tif");
close();
selectWindow("Sequence");
