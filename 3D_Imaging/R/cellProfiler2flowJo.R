
# setwd("/Users/4470246/Downloads/NCI-N87-dataset1and2-CellProfilerFeatures 3")
# fucci=read.csv("A02_3D_10object.csv")
# ii=grep("231005",fucci$FileName_bright)
# write.csv(fucci[ii,],file="~/Projects/PMO/MeasuringFitnessPerClone/data/GastricCancerCLs/3Dbrightfield/NCI-N87/I08_3DCellProfiler_FUCCI/FoFX_231005_fluorescent.nucleus/object.csv", row.names = F)
# ii=grep("240918",fucci$FileName_bright)
# write.csv(fucci[ii,],file="~/Projects/PMO/MeasuringFitnessPerClone/data/GastricCancerCLs/3Dbrightfield/NCI-N87/I08_3DCellProfiler_FUCCI/FoFX_240918_fluorescent.nucleus/object.csv", row.names = F)
# 
# setwd("/Users/4470246/Downloads/NCI-N87-dataset3-CellProfilerFeatures 3")
# fucci=read.csv("A02_3D_10object.csv")
# write.csv(fucci,file="~/Projects/PMO/MeasuringFitnessPerClone/data/GastricCancerCLs/3Dbrightfield/NCI-N87/I08_3DCellProfiler_FUCCI/FoFX_2410_fluorescent.nucleus/object.csv", row.names = F)




setwd("~/Projects/PMO/MeasuringFitnessPerClone/data/GastricCancerCLs/3Dbrightfield/NCI-N87/I08_3DCellProfiler_FUCCI/FoFX_240918_fluorescent.nucleus")
library(flowCore)
library(Biobase)
# fucci <- read.table("~/Downloads/231005_fluorescent.nucleus_fucci.txt", header = T)
# fucci <- read.table("~/Downloads/240918_fluorescent.nucleus_fucci.txt", header = T)
fucci <- read.csv(file = "object.csv")
fucci$FileName_bright=gsub(".ome.tif","",gsub("stk_0001_","",fucci$FileName_bright))
colnames(fucci)=gsub("Location_Center_","", colnames(fucci))
fucci$FileName_bright=gsub("_ch1","",fucci$FileName_bright)
fucci$FileName_bright=gsub("_ch01","",fucci$FileName_bright)
colnames(fucci) = gsub("fluor_1","green",colnames(fucci)) ## fluor_1 = green
colnames(fucci) = gsub("fluor_2","red",colnames(fucci)) ## fluor_2 = red

## test coordinates are right:
ii=grep("FoF1_",fucci$FileName_bright)
plot(fucci$X[ii],fucci$Y[ii])

tmp=strsplit(fucci$FileName_bright,"_")
fucci$FoF =as.numeric( gsub("FoF","",sapply(tmp,"[[",1)))
fucci$Date=as.numeric(sapply(tmp,"[[",2))
fucci$CellID=1:nrow(fucci)
rownames(fucci)=as.character(fucci$CellID)

fucci_=fucci[,c("CellID","FoF","Date",grep("MeanIntensity_",colnames(fucci),value=T))]
metadata <- data.frame(
  name = colnames(fucci_),
  desc = colnames(fucci_),
  range = apply(fucci_, 2, max),
  minRange = apply(fucci_, 2, min),
  maxRange = apply(fucci_, 2, max)
)
data_ff <- flowFrame(as.matrix(fucci_), parameters = AnnotatedDataFrame(metadata))

write.FCS(data_ff, paste0("~/Downloads/",fileparts(getwd())$name,".nucleus_fucci.fcs"))
# Ananlyze in Flow Jo --> export settings:
# select population --> export populations --> Concatenate tab --> format = csv scale values; include header: checked; Parameter: checked

plot(fucci_$Intensity_MeanIntensity_green,fucci_$Intensity_MeanIntensity_red, log='xy', pch=20)

cc=read.csv("~/Desktop/concat_1.csv")
cc=cc[cc$CellID %in% fucci$CellID,]
cc=cc[cc$SampleID!=1,]
cc$SampleID = cc$SampleID-1
plot(cc$Intensity_MeanIntensity_green,cc$Intensity_MeanIntensity_red, log='xy', pch=20, col=cc$SampleID)
legend("topright",as.character(unique(cc$SampleID)),fill=col)

fucci=cbind(fucci[as.character(cc$CellID),],as.matrix(cc$SampleID))
colnames(fucci)[ncol(fucci)] = "cellCycle"
fucci$cellCycle[fucci$cellCycle==4]="S"
fucci$cellCycle[fucci$cellCycle==1]="G1"
fucci$cellCycle[fucci$cellCycle==2]="G1S"
fucci$cellCycle[fucci$cellCycle==3]="G2M"

col=1:4
names(col)=unique(fucci$cellCycle)
plot(fucci$Intensity_MeanIntensity_green,fucci$Intensity_MeanIntensity_red, log='xy', pch=20, col=col[fucci$cellCycle])
legend("topright",names(col),fill=col)

fr=plyr::count(fucci$cellCycle)
fr$freq=fr$freq/sum(fr$freq)
write.csv(fucci,file = "object_cellCycle.csv",row.names = F, quote = F)


