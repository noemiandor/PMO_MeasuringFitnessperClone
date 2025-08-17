options(java.parameters = "-Xmx9g")
rm(list=c("svmFeatures"))
library(xlsx)
library(dyno)
library(caret)
library(tidyverse)
library(matlab)
library(RColorBrewer)
library(flexclust)
library(ggplot2)
library(e1071)
library(slingshot)
library(umap)
devtools::source_url("https://github.com/noemiandor/Utils/blob/master/grpstats.R?raw=TRUE")
devtools::source_url("https://github.com/noemiandor/Utils/blob/master/Pathways/getAllPathways.R?raw=TRUE")
setwd("~/Projects/PMO/MeasuringFitnessPerClone/code/3D_Imaging/R")
source("CorrectCellposeSegmentation.R")
source("assignCompartment2Nucleus.R")
source("compareCells.R")
source("generateImageMask.R")
source("Utils.R")
asDataset<-function(imgStats, imgStats_raw, FoF=NULL, coi){
  ii=1:nrow(imgStats)
  if(!is.null(FoF)){
    ii=which(imgStats$FoF==FoF)
  }
  tmp=apply(imgStats[ii,coi],2,as.numeric)
  tmp = tmp + min(tmp[tmp>0])*0.1
  tmp2=apply(imgStats_raw[ii,coi],2,as.numeric)
  rownames(tmp)<- rownames(tmp2) <- rownames(imgStats)[ii]
  dataset <- wrap_expression(
    expression = log2(tmp),
    counts = tmp2
  )
  return(dataset)
}

pseudotimeDirectionality <- function(pseudotime_img, fucci_raw, cellcyclecolumn="cellCycle"){
  pseudotime_img_=sapply(unique(pseudotime_img$FoF), function(x) pseudotime_img[pseudotime_img$FoF==x,"pseudotime"],simplify = F)
  pseudotime_fucci_=sapply(unique(fucci_raw$FoF), function(x) as.matrix(fucci_raw[fucci_raw$FoF==x,cellcyclecolumn]),simplify = F)
  te=sapply(names(pseudotime_img_), function(x) cor.test(pseudotime_img_[[x]],pseudotime_fucci_[[x]]), simplify = F)
  par(mfrow=c(2,2))
  sapply(names(pseudotime_img_), function(x) boxplot(pseudotime_img_[[x]]~pseudotime_fucci_[[x]], main=x), simplify = F)
  
  timeline=1:4
  la=sapply(names(pseudotime_img_), function(x) grpstats(as.matrix(pseudotime_img_[[x]]), pseudotime_fucci_[[x]], "mean")$mean[as.character(timeline),])
  te=list()
  for(shift in 1:length(timeline)){
    if(shift==1){
      timeline_=timeline
    }else{
      timeline_ =timeline[c(shift:length(timeline), 1:(shift-1))]
    }
    te[[shift]]=apply(la,2, function(x) cor.test(x,timeline_)$estimate)
  }
  te=do.call(rbind,te)
  ## First identify which pseudotimes must be reversed:
  toReverse=colnames(te)[apply(te,2, function(x) x[which.max(abs(x))]<0)]
  tmp=sapply(pseudotime_img_[toReverse], function(x)  x * -1, simplify = F)
  tmp=sapply(tmp, function(x) x-min(x), simplify = F)
  pseudotime_img_[toReverse]=tmp
  te[,toReverse] = te[,toReverse]*-1
  
  ##sort all again 
  idx = sapply(pseudotime_img_, order)
  pseudotime_img_ = sapply(names(pseudotime_img_), function(x) pseudotime_img_[[x]][idx[[x]]] )
  pseudotime_fucci_ = sapply(names(pseudotime_fucci_), function(x) pseudotime_fucci_[[x]][idx[[x]]] )
  
  ## Next find max of circular cross correlation to shift pseudotime
  cMax=apply(abs(te),2,which.max)
  cMax=cMax[cMax!=1]
  for(FoF in names(cMax)){
    if(!isempty(cMax)){
      x=order(pseudotime_img_[[FoF]])
      idx=which(x==round(mean(x[pseudotime_fucci_[[FoF]]>=cMax[FoF]])) )
      x=x[c(idx:length(x), 1:(idx-1))]
      pseudotime_img_[[FoF]]=pseudotime_img_[[FoF]][x]
    }
  }
  ##plot after
  par(mfrow=c(2,2))
  sapply(names(pseudotime_img_), function(x) boxplot(pseudotime_img_[[x]]~pseudotime_fucci_[[x]], main=x), simplify = F)
  
  return(pseudotime_img_)
  
} 

classifyCellCyclePhase <- function(x, y, main="", svmfit=NULL){
  dat = data.frame(as.matrix(x), y = as.factor(y))
  if(is.null(svmfit)){
    svmfit = svm(y ~ ., data = dat, kernel = "radial", cost = 10, scale = F)
    # print(svmfit)
    try(plot(svmfit, dat,area_nucleus.p~area_mito.p, main=main),silent = T)
  }
  out=predict(svmfit, dat)
  confMat=caret::confusionMatrix(dat$y, out)$byClass
  print(confMat)
  return(list(svmfit=svmfit, out=out, confusionMatrix=confMat))
}

setwd("~/Projects/PMO/MeasuringFitnessPerClone/code/SingleCellSequencing")
ROOT="~/Projects/PMO/MeasuringFitnessPerClone/data/GastricCancerCLs/3Dbrightfield/NCI-N87"
A01=paste0(ROOT,filesep,"A01_rawData")
A04=paste0(ROOT,filesep,"A04_CellposeOutput")
A05=paste0(ROOT,filesep,"A05_PostProcessCellposeOutput")
INSTATS=paste0(ROOT,filesep,"A07_LinkedSignals_Stats")
OUTPSEUDOTIME=paste0(ROOT,filesep,"A08_Pseudotime")
INPCELLIMAGES=paste0(ROOT,filesep,"A06_multiSignals_Linked")
dirCreate(OUTPSEUDOTIME,permission = "a+w")
FROMILASTIK=paste0(ROOT,filesep,"G07_IlastikOutput")
FUCCIDIR=paste0(ROOT,filesep,"I08_3DCellProfiler_FUCCI")
DATA4PAPERDIR=paste0("~/Projects/PMO/MeasuringFitnessPerClone/code/3D_Imaging/R/data4paper",filesep,fileparts(OUTPSEUDOTIME)$name)
xyz=c("x","y","z")
K=15 ## neighbors
N=30; ## cells
MINNUCVOL=7^3
MAXNUCVOL=2000
custom.settings = umap.defaults
custom.settings$n_neighbors = 8
custom.settings$negative_sample_rate = 3
custom.settings$local_connectivity = 1
xydim = 255
xmlfiles=list.files('../../data/GastricCancerCLs/3Dbrightfield/NCI-N87/A01_rawData/',pattern=".xml",full.names=T)
thedate=strsplit(as.character(Sys.time())," ")[[1]][1]

## read imaging stats (if timeseries, FoFs must be sorted in ascending temporal order)
# FoFs=grep("002005", gsub("_stats.txt","",list.files(INSTATS,pattern = "_221018_brightfield")), value=T)
# FoFs=paste0("FoF",1:5,"003_220721_brightfield")
colors=list()
date="_231005_fluorescent.nucleus"
FoVs =  c("FoFX_231005_fluorescent.nucleus")
# date="_221018_brightfield"
# FoVs =  c("FoFX002005","FoFX001003")
FoFs=c()
for(FoV in FoVs){
  print(FoV)
  FoFs=c(FoFs,grep(gsub("FoFX","",FoV), gsub("_stats.txt","",list.files(INSTATS,pattern = date)), value=T))
}
# z-axis shift for FoF3002005 renders mitochondrial features unusable:
# FoFs = grep("FoF300",FoFs,invert = T, value = T)
fucciStats <- imgStats <- list()
for(FoF in FoFs){
  imgStats_=read.table(paste0(INSTATS,filesep,FoF,"_stats.txt"),sep="\t",check.names = F,stringsAsFactors = F,header = T)
  print(paste(FoF,"# cells before any filtering:",nrow(imgStats_)))
  imgStats_=imgStats_[apply(!is.na(imgStats_),1,all),]; ## exclude cells whose volume could not be estimated
  imgStats_=imgStats_[,apply(!is.na(imgStats_),2,all)]; ## exclude features with NA vals
  imgStats_=imgStats_[imgStats_$vol_nucleus.p > MINNUCVOL,]
  # imgStats_=imgStats_[imgStats_$vol_nucleus.p < MAXNUCVOL,]
  # imgStats_=as.data.frame(collectTensorsAsVectors(FoF))
  imgStats_$FoF= FoF
  ## which frame (timepoint) is this? Used for mapping to Ilastik output:
  imgStats_$frame=which(FoF==FoFs)-1;  
  imgStats_$ID=rownames(imgStats_)
  imgStats_$png=paste0(INPCELLIMAGES,filesep,FoF,filesep, "cell_",imgStats_$ID,".png")
  imgStats[[FoF]]=imgStats_[!imgStats_$segmentationError,]
  # list.files(paste0(INPCELLIMAGES,filesep,FoF),pattern="png", full.names = T)
  
  fucciStats[[FoF]]=read.table(paste0(INPCELLIMAGES,filesep,FoF, filesep ,FoF, "_fucci.txt"),sep="\t",check.names = F,stringsAsFactors = F,header = T)
  fucciStats[[FoF]]$FoF=FoF
}
print(paste(FoF,"# cells after filtering:",sapply(imgStats,nrow)))
fucciStats=do.call(rbind, fucciStats)
imgStats=do.call(rbind, imgStats)
imgStats=imgStats[,which(apply(imgStats, 2, function(x) !all(x==0 | is.na(x))))]
## Columns of interest for slingshot
coi=setdiff(colnames(imgStats),c("FoF","frame","ID","count_nucleus.p","png","segmentationError","smv_cell","smv_nucleus.p","smv_mito.p","smv_cytoplasm.p",xyz[1:2])); #
## Exclude feature subset
dub=sapply(c("Dist","cell","vol_mito","area_mito"), function(x) grep(x,coi, value=T), simplify = F)
# coi=setdiff(coi, dub$Dist)
# coi=setdiff(coi, dub$cell)
# coi=setdiff(coi, dub$vol_mito)
# coi=setdiff(coi, dub$area_mito)
imgStats_raw=imgStats;
imgStats[,coi]=sweep(imgStats[,coi], 2, STATS = apply(imgStats[,coi],2,median, na.rm=T),FUN = "/")
print(paste("Found",nrow(imgStats),"cells across",length(FoFs),"images"))
rownames(imgStats) = paste0(imgStats$FoF,"_cell",imgStats$ID)
rownames(fucciStats) = paste0(fucciStats$FoF,"_cell",fucciStats$ID)
fucciStats = fucciStats[rownames(imgStats),]

## Now deal with FUCCI data
coi_fucci=colnames(fucciStats)[-c(1:16)]
coi_fucci=grep("^Intensity_", coi_fucci,value=T)
# coi_fucci=grep("_bright$", coi_fucci,value=T, invert = T)
# coi_fucci=grep("Edge_", coi_fucci,value=T, invert = F)
fucci_raw = fucciStats
fucciStats[,coi_fucci]=apply(fucci_raw[,coi_fucci],2,as.numeric)


## Fucci: Ad-hoc cell cycle assignment
MINGREEN = 700
MINRED = 500
# fucci_raw$cellCycle=log(1+50*(fucci_raw$Intensity_MeanIntensityEdge_green+fucci_raw$Intensity_MeanIntensityEdge_red))
plot(fucci_raw$Intensity_MeanIntensityEdge_green, fucci_raw$Intensity_MeanIntensityEdge_red)
plot(fucci_raw$Intensity_IntegratedIntensity_green, fucci_raw$Intensity_IntegratedIntensity_red)
fucci_raw$cellCycle = 2
fucci_raw$cellCycle[fucci_raw$Intensity_IntegratedIntensity_green>MINGREEN & fucci_raw$Intensity_IntegratedIntensity_red<MINRED] = 1
fucci_raw$cellCycle[fucci_raw$Intensity_IntegratedIntensity_green<MINGREEN & fucci_raw$Intensity_IntegratedIntensity_red>MINRED] = 3
fucci_raw$cellCycle[fucci_raw$Intensity_IntegratedIntensity_green>MINGREEN & fucci_raw$Intensity_IntegratedIntensity_red>MINRED] = 4
plyr::count(fucci_raw$cellCycle)
# In brief, green cells are in G1 phase, red in S phase and double positive (yellow) in G2/M.
# You might see some cells with no color, these are possibly in late G1, G1/S transition or early S phase.
# green = 1 (G1)
# no color = 2 (G1/S transition)
# red = 3 (S)
# yellow = 4 (G2/M)


# run pseudotime inference on all frames: fucci as well as allen model features 
if(exists("svmFeatures")){
  coi_=svmFeatures
}else{
  coi_=coi
}
fucciAndImgStats_=sapply(unique(imgStats$FoF), function(x) asDataset(imgStats, imgStats_raw, x, coi_), simplify = F)
tmp=sapply(unique(imgStats$FoF), function(x) asDataset(fucciStats, fucci_raw, x, coi_fucci), simplify = F)
names(tmp) = paste0("Fucci_",names(tmp))
fucciAndImgStats_= c(fucciAndImgStats_, tmp)
sapply(fucciAndImgStats_, function(x) dim(x$expression))
fucciAndImgStats_$All=asDataset(imgStats, imgStats_raw, FoF = NULL, coi_)
fucciAndImgStats_$Fucci_All=asDataset(fucciStats, fucci_raw, FoF = NULL, coi_fucci)
model=list()
for(x in names(fucciAndImgStats_)){
  dataset=fucciAndImgStats_[[x]]
  print(x)
  print(dim(dataset$expression))
  nonFinite=sum(!apply(is.finite(dataset$expression), 1, all))
  if(nonFinite>0){
    print("Non-finite entries found in dataset. Aborting.")
    next;
  }
  # dataset <- add_prior_information(
  #   dataset,
  #   start_id = rownames(tmp)[1]
  # )
  guidelines <- guidelines(
    dataset,
    answers = answer_questions(
      dataset,
      multiple_disconnected = FALSE,
      expect_topology = TRUE,
      expected_topology = "cycle"
    )
  )
  model_ <- infer_trajectory(dataset, ti_angle(dimred = "pca"))
  # model_ <- infer_trajectory(dataset, ti_slingshot())
  # model_ <- infer_trajectory(dataset, ti_paga(filter_features = F,n_comps = 10))
  # model_ <- infer_trajectory(dataset, ti_slice())
  # model_ <- infer_trajectory(dataset, ti_scorpius())
  # model_ <- infer_trajectory(dataset, ti_tscan());
  # model_ <- infer_trajectory(dataset, ti_embeddr());
  ## model_ <- infer_trajectory(dataset, ti_elpicycle())
  ## model_ <- infer_trajectory(dataset, ti_raceid_stemid())
  ## model_ <- infer_trajectory(dataset, ti_mst());
  ## model_ <- infer_trajectory(dataset, ti_dpt())
  ## model_ <- infer_trajectory(dataset, ti_monocle_ddrtree())
  if(! "pseudotime" %in% names(model_)){
    tmp=plot_dimred(model_,color_cells = "pseudotime")
    model_$pseudotime=tmp$data$pseudotime
    names(model_$pseudotime)=tmp$data$cell_id
  }
  model[[x]] =model_
}
save(file = paste0(OUTPSEUDOTIME, filesep,"pseudotime_",thedate,".RObj"), list = "model")
model_All=model[grep("All", names(model))]
model=model[grep("All", names(model), invert = T)]
## compare img and fucci derived pseudotime
pdf(paste0(OUTPSEUDOTIME, filesep,"All_img_vs_fucci-derivedPseudotime_",thedate,".pdf") )
par(mfrow=c(2,2))
ii=names(model_All$Fucci_All$pseudotime)
boxplot(model_All$Fucci_All$pseudotime~fucci_raw[ii,"cellCycle"], main=FoF,cex.main=0.6)
ii=names(model_All$All$pseudotime)
boxplot(model_All$All$pseudotime ~ fucci_raw[ii,"cellCycle"], main=FoF,cex.main=0.6)
te=cor.test(model_All$All$pseudotime, model_All$Fucci_All$pseudotime[ii])
plot(model_All$All$pseudotime, model_All$Fucci_All$pseudotime[ii], main=paste("r=",round(te$estimate,2),"P=",te$p.value))
dev.off()
## Plot trajectories:
tmp=sapply(model, function(x) plot_dimred(x,color_cells = "pseudotime"), simplify = F)
sapply(names(tmp), function(x) ggsave(paste0("~/Downloads/",x,".png"), plot=tmp[[x]]))
## Split fucci from allen model pseudotime
tmp=list()
for(invert in c(T,F)){
  ii = grep("Fucci", names(model), invert = invert, value=T)
  pseudotime_img=as.data.frame( do.call(c, sapply(model[ii], function(x) x$pseudotime)) *20)
  rownames(pseudotime_img) = do.call(c,sapply(model[ii], function(x) names(x$pseudotime)))
  colnames( pseudotime_img)="pseudotime"
  tmp[[length(tmp)+1]] = pseudotime_img
}
pseudotime_img = tmp[[1]]
fucci_raw$pseudotime=NA
fucci_raw[rownames(tmp[[2]]),]$pseudotime = tmp[[2]]$pseudotime
# pseudotime_img = pseudotime_img[rownames(pseudotime_img) %in% rownames(imgStats),,drop=F]
## Save for comparison with live-cell imaging data
pseudotime_img[,c(xyz,"FoF","frame","cellID","png")]=imgStats[rownames(pseudotime_img),c(xyz,"FoF","frame","ID","png")]
# write.table(pseudotime_img, file=paste0(OUTPSEUDOTIME,filesep,FoV,".txt"),sep="\t",row.names = F,quote = F)
# write.csv(pseudotime_img,   file=paste0(OUTPSEUDOTIME,filesep,FoV,date,".csv"),row.names = F)
pseudotime_img_median=grpstats(as.matrix(pseudotime_img[,c("pseudotime",xyz,"frame")]),pseudotime_img$frame,"median")$median
write.csv(pseudotime_img_median,   file=paste0(OUTPSEUDOTIME,filesep,FoV,date,".csv"),row.names = F)
## Sort by pseudotime
pseudotime_img = pseudotime_img[order(pseudotime_img$pseudotime),]
fucci_raw = fucci_raw[rownames(pseudotime_img),]
write.table(fucci_raw, paste0(OUTPSEUDOTIME,filesep,"Fucci_stats.txt"),sep="\t",quote = F, row.names = T)
write.table(imgStats, paste0(OUTPSEUDOTIME,filesep,"LabelFree_stats.txt"),sep="\t",quote = F, row.names = T)

## compare img and fucci derived pseudotime
pdf(paste0(OUTPSEUDOTIME, filesep,"img_vs_fucci-derivedPseudotime_",thedate,".pdf") )
par(mfrow=c(2,4))
svmFeatures <- anovaresults <- list()
for(FoF in FoFs){
  ii = rownames(fucci_raw)[fucci_raw$FoF==FoF]
  boxplot(pseudotime_img[ii,"pseudotime"]~fucci_raw[ii,"cellCycle"], main=FoF,cex.main=0.6)
  boxplot(fucci_raw[ii,"pseudotime"]~fucci_raw[ii,"cellCycle"], main=FoF,cex.main=0.6)
  print(FoF)
  print(cor.test(pseudotime_img[ii,"pseudotime"],fucci_raw[ii,"cellCycle"]))
  ##feature selection for svm classification
  anovaresults[[FoF]]=list()
  for(x in coi){
    te=anova(lm(imgStats[ii,x]~fucci_raw[ii,"cellCycle"]))
    anovaresults[[FoF]][[x]]=te$`Pr(>F)`[1]
  }
  svmFeatures_=sort(unlist(anovaresults[[FoF]]))
  svmFeatures_=svmFeatures_[svmFeatures_<0.01]
  print(svmFeatures_)
  svmFeatures[[FoF]]=names(svmFeatures_)
}
#repeat pseudotime inference with these refined coi
fr=plyr::count(unlist(svmFeatures))
whichFoF=sapply(fr$x, function(x) sapply(svmFeatures, function(f) x %in% f))
fr$FoF=apply(whichFoF, 2, function(x) paste(rownames(whichFoF)[x], collapse = ", "))
print(plyr::count(fr$FoF))
svmFeatures=fr[fr$freq>=4,]$x
# svmFeatures=fr$x[fr$FoF=="FoF2_231005_fluorescent.nucleus, FoF3_231005_fluorescent.nucleus, FoF4_231005_fluorescent.nucleus"]
##Look at top features across FoFs:
par(mfrow=c(3,3))
for(x in names(sort(unlist(anovaresults))[1:9])){
  FoF=names(unlist(sapply(FoFs, grep, x)))
  ii = rownames(fucci_raw)[fucci_raw$FoF==FoF]
  x_=gsub(paste0(FoF,"."),"",x)
  boxplot(imgStats[ii,x_]~fucci_raw[ii,"cellCycle"],ylab=x_,main=x,cex.main=0.6)
}
dev.off(); 


## Visualize pseudotime spatial distribution -- compare with FUCCI
zslice=35
pdf(paste0(OUTPSEUDOTIME, filesep,"pseudotimeSpatialDistribution_",thedate,".pdf") )
par(mfrow=c(1+length(FoFs),2),bg="gray",mai=c(0.1,0.5,0.1,0.5))
pseudotime_img[,"pseudotimeCol"]=round(pseudotime_img[,"pseudotime"]*10)
fucci_raw[,"pseudotimeCol"]=round(fucci_raw[,"pseudotime"]*10)
pseudotime_img[,"randCol"]=sample(pseudotime_img[,"pseudotimeCol"])
col=fliplr(heat.colors(max(pseudotime_img[,"pseudotimeCol"])))
fuccicol=fliplr(heat.colors(max(fucci_raw[,"pseudotimeCol"])))
# fuccicol=fliplr(rainbow(max(fucci$cellCycle)*1.2)[1:max(fucci$cellCycle)])
pseudotime_img$cellCycleFucci=NA
for(FoF in unique(pseudotime_img$FoF)){
  pseudotime_img_=pseudotime_img[pseudotime_img$FoF==FoF,]
  rownames(pseudotime_img_)=as.character(pseudotime_img_$cellID)
  fucci_=fucci_raw[fucci_raw$FoF==FoF,]
  
  OUTLINKED_=paste0(ROOT,filesep,OUTLINKED,filesep,FoF,filesep)
  coord_=readOrganelleCoordinates(signals_per_id[[FoF]], "nucleus.p", OUTLINKED_)
  coord_ = coord_[coord_$z == zslice*z_interval, ]
  # ## plot brightfield
  # TIF=list.files(paste0("../../data/GastricCancerCLs/3Dbrightfield/NCI-N87/A01_rawData",filesep,FoF), pattern=paste0("_z",zslice), full.names = T)
  # img=sapply(TIF, function(x) bioimagetools::readTIF(x), simplify = F)
  # img=sapply(img, function(x) EBImage::rotate(x,-180), simplify = F)
  # bioimagetools::img(resize4Ilastik(img[[2]], xydim = xydim)[,,1]);
  # bioimagetools::img(resize4Ilastik(img[[1]]+img[[3]], xydim = xydim)[,,1]);
  ## plot pseudotime colorcoded
  plot(coord_$x,coord_$y,col=col[pseudotime_img_[as.character(coord_$id),"pseudotimeCol"]],xaxt='n',yaxt='n', ann=FALSE,main=paste(FoF,"img"))
  ## plot random color code
  # plot(coord_$x,coord_$y,col=col[pseudotime_img_[as.character(coord_$id),"randCol"]],main=FoF,xaxt='n',yaxt='n', ann=FALSE)
  ## plot FUCCI
  plot(fucci_$x,fucci_$y, col=fuccicol[fucci_$pseudotimeCol],xaxt='n',yaxt='n', ann=FALSE,main=paste(FoF,"fucci"))
  # plot(fucci_$x,fucci_$y,col=fuccicol[fucci_$cellCycle],main=FoF,xaxt='n',yaxt='n', ann=FALSE)
}
color.bar(unique(col),min=1,max = length(col),nticks = length(unique(col)),title = "pseudo order")
dev.off()
## zoom in:
# ii=which(slice$x>0.5*max(slice$x) )
# plot(-slice$x[ii],slice$y[ii],col=col[pseudotime_img_[as.character(slice$ID[ii]),"pseudotimeCol"]],main=FoF,xaxt='n',yaxt='n', ann=FALSE,ylim=quantile(slice$y,c(0,1))*c(1,0.75))
# file.copy(TIF,"~/Downloads/")
# print(paste("open",TIF))


#### pseudotime vs FUCCI cell cycle time
pseudotime_img_=sapply(FoFs, function(x) pseudotime_img[pseudotime_img$FoF==x,], simplify = F)
pseudotime_fucci_=sapply(FoFs, function(x) fucci_raw[fucci_raw$FoF==x,], simplify = F)
# FoF="FoF3_231005_fluorescent.nucleus"
# x=fucciAndImgStats_[[FoF]]$expression[rownames(pseudotime_fucci_[[FoF]]),svmFeatures]; 
# y=pseudotime_fucci_[[FoF]]$cellCycle
trainCells=sapply(fucciAndImgStats_[FoFs], function(x) sample(rownames(x$expression), 45), simplify = F)
testCells=sapply(FoFs, function(x) setdiff(rownames(fucciAndImgStats_[[x]]$expression),trainCells[[x]]), simplify = F)
x=sapply(FoFs, function(x) fucciAndImgStats_[[x]]$expression[trainCells[[x]],svmFeatures] )
x=do.call(rbind, x)
y=sapply(FoFs, function(x) pseudotime_fucci_[[x]][trainCells[[x]],]$cellCycle, simplify = F)
y=do.call(c, y)

# train SVM to classify cells to cell cycle phase based on allen Model features:
svm=classifyCellCyclePhase(x, y, main="", svmfit=NULL)
out=sapply(names(pseudotime_fucci_), function(x) classifyCellCyclePhase(fucciAndImgStats_[[x]]$expression[testCells[[x]], svmFeatures], pseudotime_fucci_[[x]][testCells[[x]],"cellCycle"], main="", svmfit=svm$svmfit), simplify=F)
# use SVM predictions to correct directionality of pseudotime:
# la=pseudotimeDirectionality(pseudotime_img, fucci_raw,cellcyclecolumn = "cellCycle")
## Save output:
svm=list(svmfit=svm$svmfit, svmFeatures=svmFeatures, training = trainCells, testing=testCells)
save(file = paste0(OUTPSEUDOTIME, filesep,"cellCyclePredictionFromImgFeatures_svm_",thedate,".RObj"), list = "svm")
confMat=sapply(out, function(x) x$confusionMatrix, simplify = F, USE.NAMES = T)
# sapply(names(confMat), function(x) write.table(confMat[[x]], paste0(OUTPSEUDOTIME, filesep,x,"_cellCyclePrediction_svm_",thedate,".txt"), sep="\t" , quote = F))
confMat= as.data.frame(do.call(rbind,confMat))
tmp=as.data.frame(sapply(c("Class..1","Class..2","Class..3","Class..4"), function(x) apply(confMat[grep(x, rownames(confMat)),], 2, median, na.rm=T )))
tmp["FoF",]="Median"
confMat$FoF=unlist(sapply(names(out), rep, 1,4, simplify = F))
confMat=rbind(confMat,t(tmp))
write.xlsx(confMat, paste0(OUTPSEUDOTIME, filesep,"cellCyclePrediction_svm_",thedate,".xlsx"))
print(confMat)

##@TODO next: -- try next! HALO to improve 3D nuclei segmentation -- ask Mahmoud
##@TODO next: cont here -- increase pixel threshold when segmenting mito via DBSCAN -- can clustering be improved? Same for cytoplasm 
##@TODO next: calculate intensity based features for mito & cyto
##@TODO next: check instances where pseudotime_fucci_ and pseudotime_img_ don't match --> visualize these cells!
##@TODO next: segment nucleolus & calc its features
##@TODO next: finetune exclusion of segmentation errors
##@TODO next: try various pseudotime algorithms once supervised classificationa accuracy is >0.75 across all classes and FoFs
##@TODO next: ask Ana & Stan if gates for discrete fucci based classification are ok


## read seq stats
mydb = cloneid::connect2DB();
stmt = "select distinct cloneID,size,origin from Perspective where whichPerspective='TranscriptomePerspective' and origin like 'NCI-N87%' and size>0.1";
rs = suppressWarnings(dbSendQuery(mydb, stmt))
origin=fetch(rs, n=-1)
robjname=paste0(OUTPSEUDOTIME,filesep,"scRNAseq_pseudotimeInput_",thedate,".RObj");
if(!file.exists(robjname)){
  p=cloneid::getSubProfiles(cloneID_or_sampleName = 119967, whichP = "TranscriptomePerspective")
  save(file = robjname, list = "p")
}
load(robjname)
scRNA=t(p[grep(":",rownames(p),invert = T,fixed = T),])
# seqStats=read.table("../../data/GastricCancerCLs/RNAsequencing/B02_220112_seqStats/NCI-N87/Clone_0.244347_ID119967.txt",sep="\t",check.names = F,stringsAsFactors = F)
# load('~/Projects/PMO/MeasuringFitnessPerClone/data/GastricCancerCLs/RNAsequencing/B01_220112_pathwayActivity/NCI-N87/Clone_0.244347_ID119967.RObj')
# ccState=sapply(colnames(pq), function(x) cloneid::getAttribute(x,"TranscriptomePerspective","state"))
# seqStats=t(pq)
scRNA_raw=scRNA;
scRNA=sweep(scRNA, 2, STATS = apply(scRNA,2,mean, na.rm=T),FUN = "/")

## pathwayActivity
gs=getAllPathways(include_genes=T, loadPresaved = T);     
gs=gs[sapply(gs, length)>=5]
load('~/Projects/PMO/MeasuringFitnessPerClone/data/GastricCancerCLs/RNAsequencing/B01_220112_pathwayActivity/NCI-N87/Clone_0.244347_ID119967.RObj')
gs=gs[rownames(pq)]
goi=unique(unlist(gs))
pq <- gsva(t(scRNA[,colnames(scRNA) %in% goi]), gs, kcdf="Poisson", mx.diff=T, verbose=FALSE, parallel.sz=2, min.sz=5)
pq <- as.data.frame(t(pq))

# pseudotime inference sequencing
poi=grep("cycle", names(gs),ignore.case = T, value=T)
poi=union(poi, grep("G1", names(gs),ignore.case = T, value=T))
poi=union(poi, grep("G2", names(gs),ignore.case = T, value=T))
poi=union(poi, grep("mitos", names(gs),ignore.case = T, value=T))
goi=intersect(colnames(scRNA), as.character(unlist(gs[poi])))
# TOPN=500; # top variable genes
# goi=goi[order(apply(scRNA[,goi],2,var),decreasing = T)[1:TOPN]]
dataset = asDataset(scRNA, scRNA_raw, FoF=NULL, goi)
guidelines <- guidelines( dataset, answers = answer_questions(dataset,    multiple_disconnected = FALSE, expect_topology = TRUE, expected_topology = "cycle"))
model_ <- infer_trajectory(dataset, ti_angle(dimred = "pca"))
plot_dimred(model_,color_cells = "pseudotime")
pq$pseudotime =model_$pseudotime[rownames(pq)]
pq$FoF=origin$origin[1]
plot(sort(pq$pseudotime));
points(sort(model$All$pseudotime),col="red")
legend("bottomleft", c("seq","img"), fill=c("black","red"))
save(file = paste0(OUTPSEUDOTIME, filesep,"scRNAdataset_",thedate,".RObj"), list = "pq")
model$Seq = model_
save(file = paste0(OUTPSEUDOTIME, filesep,"pseudotime_",thedate,".RObj"), list = "model")

## Same number of sequenced and imaged cells:
# pq=pq[sample(nrow(pq),size = nrow(imgStats)),]
# g1cells_seq=rownames(pq)[ccState[rownames(pq)]=="G0G1"]



# ############### ############### ############### ############### ############### 
# ############## Figures for paper # ############### ############### ##############
# ############### ############### ############### ############### ##############
## cp all required input to data4paper folder as input to continuousCCprediction_figures4paper.Rmd
file.copy(paste0(OUTPSEUDOTIME,filesep,"Fucci_stats.txt"),DATA4PAPERDIR, overwrite = T)
file.copy(paste0(OUTPSEUDOTIME,filesep,"LabelFree_stats.txt"),DATA4PAPERDIR, overwrite = T)
file.copy(paste0(OUTPSEUDOTIME, filesep,"pseudotime_",thedate,".RObj"),DATA4PAPERDIR, overwrite = T)
file.copy(paste0(OUTPSEUDOTIME, filesep,"cellCyclePredictionFromImgFeatures_svm_",thedate,".RObj"),DATA4PAPERDIR, overwrite = T)
file.copy(paste0(OUTPSEUDOTIME, filesep,"cellCyclePrediction_svm_",thedate,".xlsx"),DATA4PAPERDIR, overwrite = T)
file.copy(paste0(OUTPSEUDOTIME, filesep,"scRNAdataset_",thedate,".RObj"),DATA4PAPERDIR, overwrite = T)



# #############################################################
# ## Compare pseudotime with LCI-derived time since division ##
# #############################################################
# realtime_img=read.table(file=paste0(FROMILASTIK,filesep,FoF,"_DeltaDivision.txt"),sep="\t",check.names = T,stringsAsFactors = T,header = T)
# ## Coordinate based mapping of cells between pseudotime and Ilastik output per each frame:
# jointTimes=list()
# par(mfrow=c(3,2))
# for(time in unique(pseudotime_img$frame)){
#   print(paste("merging time",time,"..."))
#   pseudotime_img_ = pseudotime_img[pseudotime_img$frame==time,]
#   realtime_img_ = realtime_img[realtime_img$frame==time,]
#   d=dist2(pseudotime_img_[,xyz], realtime_img_[,xyz])
#   ii=apply(d,2,which.min)
#   realtime_img_$pseudotime= pseudotime_img_$pseudotime[ii] 
#   ##@TODO: test -- there should be a unique arg minimum for each cell
#   print(apply(pseudotime_img[,xyz],2,quantile))
#   print(apply(realtime_img_[,xyz],2,quantile))
#   # heatmap.2(d,trace="n")
#   hist(plyr::count(ii)$freq,30,main=paste("time",time))
#   # - focus on just cells with assigned parent track ID
#   ii=!is.na(realtime_img_$time_since_division) | !is.na(realtime_img_$time_until_division)
#   jointTimes[[as.character(time)]]=realtime_img_[ii,,drop=F]
# }
# jointTimes=do.call(rbind, jointTimes)
# write.csv(jointTimes,file=paste0(OUTPSEUDOTIME,filesep,FoV,".csv"),row.names = F)
# # # run in Matlab: 
# #   # calc circular cross correlation btw. the "pseudotime timepoints" and the estimates of "time_since_division"
# #   circularCrossCorr(jointTimes$time_since_division,jointTimes$pseudotime)
# ## Read in data after shifting pseudotime by cross correlation
# f=list.files(OUTPSEUDOTIME, pattern="_matlabOut.csv", full.names = T)
# pdf("~/Downloads/realtime_vs_pseudotime.pdf",height = 4.5,width = 9)
# par(mfrow=c(1,2))
# for(FoV in c("FoFX002005","FoFX001003")){
#   jointTimes=read.csv(file=grep(FoV,f,value=T),check.names = F,stringsAsFactors = F)
#   # te=cor.test(jointTimes$time_since_division,jointTimes$pseudotime_shifted,method = "spearman")
#   te=cor.test(jointTimes$frame,jointTimes$pseudotime_shifted,method = "spearman")
#   plot(jointTimes$frame,jointTimes$pseudotime_shifted,pch=20,cex=2, xlab="frame", ylab="shifted pseudotime", main=paste0(FoV, ": R=",round(te$estimate,2),"; p=", round(te$p.value,10)))
#   fr=grpstats(jointTimes[,"pseudotime",drop=F], round(jointTimes$frame), "median")$median
#   
#   vioplot::vioplot(jointTimes$pseudotime_shifted ~ round(jointTimes$frame),  xlab="frame", ylab="shifted pseudotime", main=paste0(FoV, ": R=",round(te$estimate,2),"; p=", round(te$p.value,10)))
# }
# dev.off()
