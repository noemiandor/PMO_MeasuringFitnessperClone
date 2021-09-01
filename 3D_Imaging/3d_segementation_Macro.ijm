//run("Brightness/Contrast...");
run("Enhance Contrast", "saturated=0.35");
run("Enhance Contrast", "saturated=0.35");
run("Apply LUT", "stack");
run("Gamma...", "value=1.12 stack");
run("Enhance Contrast...", "saturated=0.3 normalize equalize process_all");
run("Bandpass Filter...", "filter_large=160 filter_small=3 suppress=None tolerance=5 autoscale saturate process");

setAutoThreshold("MaxEntropy dark");
//run("Threshold...");
setOption("BlackBackground", true);
run("Convert to Mask", "method=MaxEntropy background=Dark calculate black");
run("Open", "stack");
run("Remove Outliers...", "radius=15 threshold=50 which=Bright stack");
run("Fill Holes", "stack");
run("Remove Outliers...", "radius=20 threshold=50 which=Bright stack");
run("Distance Transform Watershed 3D", "distances=[Borgefors (3,4,5)] output=[16 bits] normalize dynamic=2 connectivity=6");
run("3D Manager");
run("3D Manager");
Ext.Manager3D_AddImage();
// do some measurements, save measurements and close window
Ext.Manager3D_Measure();
Ext.Manager3D_SaveResult("M","C:/Users/Rudy/Desktop/Results3D.csv");
Ext.Manager3D_CloseResult("M");

