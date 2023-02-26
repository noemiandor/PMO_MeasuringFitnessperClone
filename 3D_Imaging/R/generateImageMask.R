# source("R/generateImageMask.R")
# FoF="FoF1001_220407_brightfield";
# generateImageMask(FoF)
generateImageMask <- function(FoF, INDIR="A05_PostProcessCellposeOutput", OUTDIR="B06_OrganelleMasks", root="/raid/crdlab/ix1/Projects/M005_MeasuringFitnessPerClone_2019/data/GastricCancerCLs/3Dbrightfield/NCI-N87", targetcellids=NULL, xydim=1024, signal=""){
  library(matlab)
  library("rhdf5")
  idim = 1024
  xydim = min(idim, xydim)
  OUTDIR=paste0(root,filesep,OUTDIR)
  H5OUT=paste0(OUTDIR,filesep,FoF,".h5")
  olddir=getwd()
  success=try(setwd(paste0(root,filesep,INDIR,"/",FoF,"/All_Cells_coordinates")),silent = T)
  if(class(success)=="try-error"){
    print(paste("No output found for",FoF,"under",INDIR,". Postprocess segmentation first."))
    return()
  }
  dirCreate(paste0(OUTDIR,filesep,FoF), permission = "a+w")
  
  tmp=list.files(pattern=signal)
  cells=sample(length(tmp),length(tmp))
  names(cells)=tmp
  ## mark filtered cells if applicable
  if(!is.null(targetcellids)){
    allids=sapply(strsplit(names(cells),"cell_"),"[[",2)
    allids=sapply(strsplit(allids,"_"),"[[",1)
    cells[!allids %in% targetcellids]=NA
  }
  
  zstack=69
  images <- h5 <- list()
  for(z in 1:zstack){
    img=matrix(0,idim,idim)
    for(cell in names(cells)){
      csv=read.csv(cell)
      for(i in which(csv$z==z)){
        img[csv$x[i],csv$y[i]]=cells[cell]
      }
    }
    
    ## write image
    z_=z
    if(z<10){
      z_=paste0("0",z)
    }
    iout=paste0(OUTDIR,filesep,FoF,filesep,FoF,"_z",z_)
    png(paste0(iout,".png"),width = idim, height = idim)
    par(mai=c(0,0,0,0)); 
    bioimagetools::img(t(img),col=rainbow(max(img,na.rm=T)), mask=!is.na(t(img)))
    write.table(t(img), file=paste0(iout,".txt"),sep="\t",quote=F, row.names = F, col.names = F)
    # image(img,frame.plot=F,axes=F)
    dev.off()
    ## save for gif
    images[[z]]=try(magick::image_read(iout),silent = T)
    img[is.na(img)]=0
    h5[[z]]=EBImage::resize(img,h = xydim, w=xydim)
  }
  ## save for h5
  file.remove(H5OUT)
  h5createFile(H5OUT)
  h5=do.call(abind,c(h5,along=3))
  h5write(h5, file = H5OUT, 'foo')
  h5closeAll()
  
  ## animate at 2 frames per second
  prefix=paste0(OUTDIR,filesep,FoF)
  cmd=paste0("convert -delay 20 -loop 0 ",prefix,filesep,"*.png ",prefix,".gif ")
  system(cmd)
  ## Clean up:
  try(sapply(list.files(prefix, pattern=".png", full.names=T), file.remove))
  
  setwd(olddir)
  # if(all(sapply(images,class)!="try-error")){
  #   img_joined <- image_join(images)
  #   img_animated <- image_animate(img_joined, fps = 1)
  #   image_write(image = img_animated, path = paste0(OUTDIR,filesep,FoF,".gif"))
  # }else{
  #   print("Magick not installed. No .gif output generated")
  # }
  return(h5)
}
