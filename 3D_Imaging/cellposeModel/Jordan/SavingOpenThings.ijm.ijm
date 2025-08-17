
folder = "C:/Users/80027908/Desktop/OHSU-G10";
name = File.nameWithoutExtension;
folder2 = folder + File.separator + name;
File.makeDirectory(folder2);

run("Stack to Images");

for (i=0;i<nImages;i++) {
	selectImage(i+1); 
    title = getTitle; 
    print(title);
    saveAs("tiff", folder2 + File.separator + title); 
}
print("Done");
close("*");

// Or..

input = "C:/Users/80027908/Desktop/K00_GemcitabineExposure_033023/F6_1/HALO Markup";
output = input + File.separator + "output";
File.makeDirectory(output);

list = getFileList(input);

setBatchMode(true);
for (i = 0; i < list.length; i++){
	saveSelected(input, output, list[i]);
}

function saveSelected(input, output, filename) {
	open(input + File.separator + filename);
	filename_pure = File.nameWithoutExtension;
	run("RGB Color", "slices");
	run("Stack to Images");
	selectImage(filename_pure + "-0001");
	close("\\Others");
	run("Scale...", "x=.2854 y=.2854 width=1468 height=1100 interpolation=Bilinear average create");
    saveAs("tiff", output + File.separator + filename);
    print(filename);
	close("*");
}


// Or..

dir = getDirectory("Choose a Directory"); 
for (i=0;i<nImages;i++) {
	selectImage(i+1); 
    title = getTitle; 
    print(title);
    saveAs("tiff",  dir + title); 
}
print("Done");
close("*");