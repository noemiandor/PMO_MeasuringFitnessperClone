# conda activate r_env
# setwd("/raid/crdlab/ix1/Projects/M005_MeasuringFitnessPerClone_2019/code/R")
setwd("~/Projects/PMO/MeasuringFitnessPerClone/code/3D_Imaging/R")
source("CorrectCellposeSegmentation.R")
source("assignCompartment2Nucleus.R")
source("compareCells.R")
source("generateImageMask.R")
source("Utils.R")
source("visualizeSingleCells.R")
eps=read.table('../dbscan/eps.txt')
library(abind)
library(matlab)
library(rgl)
library(ijtiff)
library(geometry)
library(flexclust)
devtools::source_url("https://github.com/noemiandor/Utils/blob/master/grpstats.R?raw=TRUE")


## Constants, Settings, Input and output folders:
# ROOT="/raid/crdlab/ix1/Projects/M005_MeasuringFitnessPerClone_2019/data/GastricCancerCLs/3Dbrightfield/NCI-N87"
ROOT="~/Projects/PMO/MeasuringFitnessPerClone/data/GastricCancerCLs/3Dbrightfield/NCI-N87"
# ROOT="~/Projects/PMO/MeasuringFitnessPerClone/data/GastricCancerCLs/3Dbrightfield/SUM-159"
setwd(ROOT)
EPS=10; #6
MINPTS=3; #4
xydim = 255
pixelsize_xy = 0.232 # um 
z_interval = 0.29 #  um 
# r3dDefaults$windowRect=c(0,50, 1600, 800) 
r3dDefaults$windowRect=c(0,50, 500,500)
INDIR="A04_CellposeOutput"
OUTCORRECTED="A05_PostProcessCellposeOutput"
dirCreate(OUTCORRECTED, permission = "a+w")
# ILASTIKINPUT="G06_IlastikInput"
# dirCreate(ILASTIKINPUT, permission = "a+w")
OUTLINKED="A06_multiSignals_Linked"
OUTSTATS="A07_LinkedSignals_Stats"
dirCreate(OUTLINKED, permission = "a+w")
dirCreate(OUTSTATS, permission = "a+w")
xmlfiles=list.files('A01_rawData/',pattern=".xml",full.names=T)


## Local helper functions
correctSegmentations<-function(FoF, signals, eps){
  sapply(names(signals), function(x) CorrectCellposeSegmentation(FoF,signal=x,INDIR,OUTCORRECTED,doplot=F,eps=eps[FoF,x]))
}
readOrganelleCoordinates<-function(signals_per_id, signals, IN){
  coord=c();
  for(cell in signals_per_id$x){
    for(s in signals){
      x=paste0(s,"_cell_",cell,"_coordinates.csv")
      a=read.csv(paste0(IN,filesep,x))
      a$z=a$z*z_interval
      a$x=a$x*pixelsize_xy
      a$y=a$y*pixelsize_xy
      id=strsplit(fileparts(x)$name,"_")[[1]]
      a$organelle = a[,ncol(a)] 
      a$signal=s
      id=id[length(id)-1]
      a$id=id
      coord=rbind(coord,a[,c("y", "x", "z", "organelle", "id","signal")])
    }
  }
  coord$id=as.numeric(coord$id)
  return(coord)
}


###############################################
######Allen model performance evaluation#######
###############################################

## Input and output:
# FoFs=paste0("FoF",1:5,"007_220523_brightfield")
# FoFs=paste0("FoF",1:5,"001_220721_brightfield")
# FoFs=list.files(INDIR, pattern="001003_221018_brightfield")
# FoFs=list.files(INDIR, pattern="001005_221018_brightfield")
FoFs=list.files(INDIR, pattern="002005_221018_brightfield")
# FoFs=c(list.files(INDIR, pattern="FoF20020"), list.files(INDIR, pattern="FoF40020")); FoFs=grep("221018_brightfield",FoFs, value = T)
signals=list(nucleus.p="nucleus.p_Cells_Centers.csv",mito.p="mito.p_Cells_Centers.csv")
# signals=list(nucleus.p="nucleus.p_Cells_Centers.csv"); #nucleus.t="nucleus.t_Cells_Centers.csv",
# signals=list(nucleus.t="nucleus.t_Cells_Centers.csv"); 
stats=list()
mfrow3d(nr = 1, nc = 4, sharedMouse = TRUE)  
rawimges <- images <- list()
ncells=list()
for(FoF in FoFs){
  print(FoF)
  setwd(ROOT)
  # unlink(paste0(OUTCORRECTED,filesep,FoF),recursive=T)
  # 
  # ###############################################
  # ###### Correcting Cellpose Segmentation #######
  # ###############################################
  # ## For linking organelles:
  # CorrectCellposeSegmentation(FoF,signal=names(signals)[1],INDIR,OUTCORRECTED,doplot=0,eps=EPS,minPts=MINPTS,IMPORTALLORGANELLES=T)
  # ncells[[FoF]]= length(list.files(paste0(OUTCORRECTED,filesep,FoF,filesep,"All_Cells_coordinates"),pattern = "nucleus"))
  # ## For live-cell tracking:
  # # # CorrectCellposeSegmentation(FoF,signal=names(signals),INDIR,OUTCORRECTED,doplot=F,eps=EPS,minPts=MINPTS,IMPORTALLORGANELLES=F)
  # # # rgl.snapshot("~/Downloads/Brightfield_Timeseries.png")
  # # images[[FoF]]=generateImageMask(FoF, INDIR=OUTCORRECTED, OUTDIR=OUTCORRECTED,root = ROOT, xydim = xydim)
  # images[[FoF]]=try(generateImageMask(FoF, INDIR=OUTCORRECTED, OUTDIR=OUTCORRECTED,root = ROOT, signal = "nucleus.p"))
  # # img=bioimagetools::readTIF(paste0(INDIR,filesep,FoF,filesep,names(signals),".tif"))
  # # img=img[fliplr(1:nrow(img)),,1:dim(images[[FoF]])[3]]
  # # img=EBImage::rotate(img,-90)
  # # rawimges[[FoF]]=resize4Ilastik(img, xydim = xydim)
  # 
  # 
  # ## Next link each predicted nucleus to its closest target nucleus
  OUTLINKED_=paste0(getwd(),filesep,OUTLINKED,filesep,FoF,filesep)
  # unlink(OUTLINKED_,recursive=T)
  # setwd(paste0(OUTCORRECTED,filesep,FoF,filesep,"Cells_center_coordinates"))
  # # # assignCompartment2Nucleus(signals$nucleus.p, signals$nucleus.t, OUTLINKED_)
  # assignCompartment2Nucleus(signals$mito.p, signals$nucleus.p, OUTLINKED_, save_cell_gif=T)
  # setwd(ROOT)
  
  ## Visualize cells
  cells=unique(sapply(strsplit(list.files(OUTLINKED_,pattern = "nucleus.p"),"_"),"[[",3))
  tmp=sapply(cells, function(i) visualizeSingleCells(i, signals$mito.p, signals$nucleus.p, OUTLINKED_))
   
  # ## Compare each predicted to its linked target nucleus
  # stats[[FoF]]=compareCells(signals$nucleus.t, signals$nucleus.p, OUTLINKED_)
}
barplot(unlist(ncells), names=names(ncells))
save(file="~/Downloads/stats.RObj","stats")
## save as h5 for Ilastik
saveAsH5<-function(images, H5OUT, binary=F, dotrim=F){
  h5=do.call(abind,c(images,along=4))
  # h5=aperm(h5,c(4,1,2,3))
  if(binary){
    h5[h5>0]=1
  }
  file.remove(H5OUT)
  h5createFile(H5OUT)  
  if(dotrim){
    h5=255*h5
    h5=h5-min(h5)
  }
  h5=round(h5)
  h5createDataset(H5OUT, dataset = fileparts(H5OUT)$name, dims=dim(h5), H5type="H5T_NATIVE_UINT32")
  h5write(h5, file = H5OUT, fileparts(H5OUT)$name)
  h5closeAll()
  return(h5)
}
tmp=gsub(substr(FoF,1,4), "FoFX",FoF)
h5=saveAsH5(rawimges,paste0(ILASTIKINPUT,filesep,tmp,".h5"),dotrim=T)
h5=saveAsH5(images,paste0(ILASTIKINPUT,filesep,tmp,"_mask.h5"))


##Plot stats for first FoF
stats_=stats[[1]]
minmax=quantile(unlist(stats_[,1:2]), c(0,1))
par(mfrow=c(2,2))
plot(stats_$nucleus.t_NumPixels,stats_$nucleus.p_NumPixels,pch=20,log="xy",xlim=minmax,ylim=minmax)
hist(stats_$nucleus.t_IntersectingPixels,col="cyan")
hist(stats_$nucleus.p_IntersectingPixels,col="cyan")

## for Saeed: Test mask as pseudo label to learn to classify mitotic cells 
mask=generateImageMask("FoF2002006_221018_brightfield", INDIR=OUTCORRECTED, OUTDIR=OUTCORRECTED,root = ROOT, signal = "nucleus.p")

#################################
###### Linking organelles #######
#################################
# 2005, 2006, 1005, 1003
# 1 = unsynchronized, 2 = synchronized
FoFs=list.files(OUTLINKED, pattern="001003_221018_brightfield")
# FoFs=list.files(OUTLINKED, pattern="002005_221018_brightfield")
# FoFs=list.files(OUTLINKED, pattern="001005_221018_brightfield")
signals=list(nucleus.p="nucleus.p_Cells_Centers.csv",mito.p="mito.p_Cells_Centers.csv"); #,cytoplasm.t="cytoplasm.t_Cells_Centers.csv")
# FoFs="FoF13_220228_fluorescent.cytoplasm"
# signals=list(nucleus.p="nucleus.p_Cells_Centers.csv",mito.p="mito.p_Cells_Centers.csv",cytoplasm.t="cytoplasm.t_Cells_Centers.csv")
signals_per_id=list()
## Input and output:
for(FoF in FoFs){
  OUTLINKED_=paste0(getwd(),filesep,OUTLINKED,filesep,FoF,filesep)
  
  # ## link each predicted nucleus to its closest target nucleus
  # setwd(paste0(OUTCORRECTED,filesep,FoF,filesep,"Cells_center_coordinates"))
  # assignCompartment2Nucleus(signals$mito.p, signals$nucleus.p, OUTLINKED_)
  # # assignCompartment2Nucleus(signals$cytoplasm.t, signals$nucleus.p, OUTLINKED_)
  # setwd(ROOT)
  
  ## keep only cells with all three signals:
  f=list.files(OUTLINKED_,full.names = T, pattern = ".csv")
  signals_per_id_=plyr::count(sapply(strsplit(f,"_"), function(x) x[length(x)-1]))
  toRM=signals_per_id_$x[signals_per_id_$freq<length(signals)]
  for(x in toRM){
    y=list.files(OUTLINKED_,full.names = T,pattern = paste0("_",x,"_"))
    file.remove(y)
  }
  signals_per_id[[FoF]]=signals_per_id_[!signals_per_id_$x %in% toRM,]
}
barplot(sapply(signals_per_id,nrow), names=names(signals_per_id))


###########################
## Calculate image stats ##
###########################
for(FoF in names(signals_per_id)){
  print(FoF)
  OUTLINKED_=paste0(getwd(),filesep,OUTLINKED,filesep,FoF,filesep)
  
  ## Read in linked organelles
  coord_=readOrganelleCoordinates(signals_per_id[[FoF]], names(signals), OUTLINKED_)
  save(file=paste0("~/Downloads/stats_",FoF,".RObj"),"coord_")
  coord=coord_; ## Backup
  
  ## Gather stats
  cells=unique(coord_$id)
  thesignals=unique(coord_$signal)
  imgStats=as.data.frame(matrix(NA,length(cells),3+4*length(thesignals)))
  rownames(imgStats)=as.character(cells)
  colnames(imgStats)=c(sapply(c("vol_","area_","pixels_","count_"),paste0,thesignals),"x","y","z")
  for(id in cells){
    # print(paste("cell",id))
    a=coord_[coord_$id==id,]
    imgStats[as.character(id),c("x","y","z")]=apply(a[a$signal=="nucleus.p",c("x","y","z")],2,median)
    ## shotcut to get convex hull stats
    getstats<-function(a_){
      hull <- try(convhulln(a_[,c("x","y","z")], options = "FA", return.non.triangulated.facets=T),silent = T)
      # hull=try(Plot_ConcaveHull(a_$x, a_$y, a_$z, lcolor =1, alpha=0.15,add = F))
      if(class(hull)=="try-error"){
        hull=list(area=NA, vol=NA, maxAreaSlice=NA)
      }
      ## includes nucleus coordinates
      if(!isempty(grep("nucleus", a_$signal))){
        ## Central slice area
        slice <- try(convhulln(a_[,c("x","y")], options = "FA"),silent = T)
        hull$maxAreaSlice=slice$area
      }
      hull$pixels=nrow(a_)
      return(hull[c("area","vol","pixels","maxAreaSlice")])
    }
    ## stats for entire cell (all organelles together)
    stats_=getstats(a)
    stats_$count=1
    imgStats[as.character(id),paste0(names(stats_),"_cell")]=sapply(stats_,sum,na.rm=T)
    ## iterate through signals: mito, cyto, nucleus
    for(i in 1:length(thesignals)){
      signal=thesignals[i]
      if(isempty(grep("nucleus",signal))){
        stats_=list(area=NA,vol=NA,pixels=NA, maxAreaSlice=NA)
        for(organelle in unique(a$organelle)){
          a_=a[a$organelle==organelle & a$signal==signal,]
          hull=getstats(a_)
          stats_$area=c(stats_$area,hull$area)
          stats_$vol=c(stats_$vol,hull$vol)
          stats_$pixels=c(stats_$pixels,hull$pixels)
        }
        stats_$count=length(stats_$area)-1
      }else{
        a_=a[a$signal==signal,]
        stats_=getstats(a_)
        stats_$count=1
      }
      imgStats[as.character(id),paste0(names(stats_),"_",signal)]=sapply(stats_,sum,na.rm=T)
      
      ## distance to other organelles
      a_=a[a$signal==thesignals[i],]
      a_= t(as.matrix(apply(a_[,c("x","y","z")], 2, median)))
      for(j in i:length(thesignals)){
        if(i==j){
          next
        }
        b_=a[a$signal==thesignals[j],]
        b_=grpstats(b_[,c("x","y","z")], b_$organelle,"median")$median
        d=dist2(a_,b_)
        imgStats[as.character(id), paste0(c("Min","Max","Median"),"Dist_",thesignals[i],"_",thesignals[j])]=quantile(d,c(0,1,0.5))
      }
    }
  }
  
  ## Add more stats and save 
  imgStats$pixel_per_mito_avg=imgStats$pixels_mito.p/imgStats$count_mito.p
  imgStats$pixel_per_volume_mito=imgStats$pixels_mito.p/imgStats$vol_mito.p
  for(organelle in thesignals){
    imgStats[,paste0("pixels_per_volume_",organelle)]=imgStats[,paste0("pixels_",organelle)]/imgStats[,paste0("vol_",organelle)]
  }
  if("cytoplasm.t" %in% names(signals)){
    imgStats$nuc_to_mito_plus_cyto = imgStats$vol_nucleus.p/(imgStats$vol_mito.p+imgStats$vol_cytoplasm.t)
    imgStats$nuc_to_cyto = imgStats$vol_nucleus.p/imgStats$vol_cytoplasm.t
    imgStats$cyto_to_mito = imgStats$vol_cytoplasm.t/imgStats$vol_mito.p
  }
  imgStats$nuc_to_mito = imgStats$vol_nucleus.p/imgStats$vol_mito.p
  imgStats$nuc_vol_to_area=imgStats$vol_nucleus.p/imgStats$area_nucleus.p
  write.table(imgStats,file=paste0(OUTSTATS,filesep,FoF,"_stats.txt"),sep="\t",quote=F,row.names = T)
  
  
  ## Visualize stats
  imgStats=read.table(paste0(OUTSTATS,filesep,FoF,"_stats.txt"),header = T,sep="\t")
  ## Z-score
  imgStats=apply(imgStats, 2, function(x) (x - mean(x,na.rm=T))/sd(x,na.rm=T))
  tmp=as.matrix(imgStats)
  tmp[!is.finite(tmp)]=NA
  # hm = gplots::heatmap.2(tmp,trace = "none", margins = c(13, 6), symm = F)
}
plot(sapply(signals_per_id,nrow))



#################################################################################
### Visualize segmentation of synchronized and unsynchronized cells over time ###
#################################################################################
setwd(ROOT)
FoFs=c("2005","1003")
dateID="_221018_brightfield"
f=sapply(FoFs, function(FoF) list.files("A03_allenModel", pattern=paste0(FoF,dateID), full.names = T))
f_cellpose=sapply(FoFs, function(FoF) list.files("A06_multiSignals_Linked/",pattern = paste0(FoF,dateID), full.names = T))
# f_cellpose=sapply(FoFs, function(FoF) list.files("A04_CellposeOutput/",pattern = paste0(FoF,dateID), full.names = T))
# f_cellpose=sapply(FoFs, function(FoF) list.files("A05_PostProcessCellposeOutput/",pattern = paste0(FoF,dateID), full.names = T))
slice=27
ncells=as.data.frame(f)
ncells[T]=NA
rownames(ncells) =getTimeStampsFromMetaData(sapply(f[,1], function(x) fileparts(x)$name), root="A01_rawData", xmlfiles)
pdf(paste0("~/Downloads/3D_NCI-N87_slice_",slice,".pdf"),width = 6,height = 12);
par(mfrow=c(5,2),mai=c(0.05,0.25,0.05,0.25))
for(i in 1:nrow(f)){
  img=sapply(f[i,], function(x) bioimagetools::readTIF(paste0(x,"/nucleus.p.tif")), simplify = F)
  for(j in 1:ncol(f)){
    f_cells=list.files(f_cellpose[i,j], full.names = T, pattern = "nucleus")
    # f_cells=list.files(paste0(f_cellpose[i,j],filesep,"All_Cells_coordinates"), full.names = T, pattern = "nucleus.p_")
    ncells[i,j]=length(f_cells)
    bioimagetools::img(t(fliplr(img[[j]][,,slice])))
    # for(f_cell in f_cells){
    #   dm=read.csv(f_cell)
    #   dm=dm[abs(dm$z-slice)<=5,]
    #   dm =dm[sample(nrow(dm),nrow(dm)/220),]
    #   points(dm$y,dm$x,pch=20, cex=0.02, col=which(f_cell==f_cells))
    # }
  }
}
ncells=sweep(ncells,2,STATS = apply(ncells,2,min), FUN = "/")
ncells$hour=as.numeric(rownames(ncells))
ncells$synchronized=1
ncells$unsynchronized=0
ncells=as.matrix(ncells)
dev.off()
## Compare growth dynamics for synchronized vs unsynchronized cells
ncells=as.data.frame(rbind(ncells[,c(1,3,4)], ncells[,c(2,3,5)]))
ncells$synchronized=as.factor(ncells$synchronized)
colnames(ncells)[1]="cells"
p=ggplot(ncells, aes(x=hour, y=cells)) + 
  geom_line(aes(colour=synchronized, group=synchronized)) +
  geom_point(aes(colour=synchronized),size=3)    
ggsave("~/Downloads/NCI-N87_synchronized_vs_unsynchronized_cellCount.png",p,width = 4,height = 3)


##########################
## Visualize organelles ##
##########################
tmp = quantile(coord_$z,c(0,1))
space=tmp[2]-tmp[1]
zlim=c(tmp[1]-space/2, tmp[2]+space/2)
col=rainbow(length(unique(coord_$signal)))
names(col)=as.character(unique(coord_$signal))
## Color by organelle
rgl::close3d()
# rgl::plot3d(coord_$x, coord_$y, coord_$z, pch3d=20, zlim=zlim, size=2, axes=F, xlab="",ylab="", zlab="",col="white",alpha=0.4)
alpha=list(cytoplasm.p=0.01,cytoplasm.t=0.01,nucleus.t=1,nucleus.p=1,mito.t=0.1,mito.p=0.1)
for(s in names(signals)){
  X=coord_[coord_$signal==s,]
  if(s=="nucleus.p"){
    rgl::plot3d(X$x, X$y, X$z, pch3d=20, zlim=zlim, size=2, axes=F, xlab="",ylab="", zlab="",col=col[X$signal],alpha=alpha[[s]], add=T)
  }else{
    rgl::points3d(X$x, X$y, X$z, pch3d=20, col=col[X$signal],alpha=alpha[[s]], add=T)
  }
}
## Color by cell
rgl::close3d()
col=rainbow(length(unique(coord_$id)))
names(col)=as.character(unique(coord_$id))
rgl::plot3d(coord_$x, coord_$y, coord_$z, pch=20, zlim=zlim, size=2, axes=F, xlab="",ylab="", zlab="",col=col[as.character(coord_$id)],add=T)
## Save as gif
# rgl::movie3d(
#   movie=paste0("CellPose3D_output_",FoF),
#   rgl::spin3d( axis = c(1, 1, 1), rpm = 8),
#   duration = 1,
#   dir = "~/Downloads/",
#   type = "gif",
#   clean = TRUE
# )



## Visualize segmentation stats for various DBSCAN runs
library(ggplot2)
library(patchwork)
Y=list()
for(COI in c("zslices_nucleus.t","vol_nucleus.t")){
  FoF="FoF12_211110_fluorescent.nucleus"
  setwd("~/Projects/PMO/MeasuringFitnessPerClone/data/GastricCancerCLs/3Dbrightfield/NCI-N87/D06_Stats")
  f=list.files()
  X=lapply(f, function(x) read.csv(paste0(x,filesep,FoF,"_stats.csv")))
  X_=sapply(X, function(x) x[x[,COI]>0, COI])
  names(X_)=f
  ## Sort by epsilon
  eps=gsub("EPS","",sapply(strsplit(names(X_),"_"),"[[",1))
  Y[[COI]]=X_[order(as.numeric(eps))]
}
## params & stats
med=as.data.frame(sapply(Y, function(X_) sapply(X_, median)))
plot(med$zslices_nucleus.t, med$vol_nucleus.t, pch=21, cex=2)
text(jitter(med$zslices_nucleus.t-1,5), jitter(med$vol_nucleus.t+5,50),gsub("MINPTS","M",names(Y[[1]])),cex=0.7)

X_=Y[[2]]
p=lapply(names(X_), function(x) ggplot(data.frame(X_[[x]]), aes(X_[[x]])) +
           geom_histogram(bins = 32) + ggtitle(x)
         + scale_x_continuous(trans = "log")         )

print(sapply(X_,median))
p[[1]]+p[[2]]+p[[3]]+p[[4]]+p[[5]]+p[[6]]+p[[7]]+p[[8]]+p[[9]]
p[[10]]+p[[11]]+p[[12]]+p[[13]]+p[[14]]+p[[15]]+p[[16]]+p[[17]]+p[[18]]
p[[19]]+p[[20]]+p[[21]]+p[[22]]+p[[23]]+p[[24]]+p[[25]]+p[[26]]+p[[27]]
p[[28]]+p[[29]]+p[[30]]

