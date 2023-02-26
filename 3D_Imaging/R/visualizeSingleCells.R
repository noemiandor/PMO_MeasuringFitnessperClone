visualizeSingleCells<-function(i, other_center_coord, nuc_center_coord,OUTD){
  
  print(paste("cell",i))
  other_center_coord=strsplit(other_center_coord,"_Cells_")[[1]][1]
  nuc_center_coord=strsplit(nuc_center_coord,"_Cells_")[[1]][1]
  
  dm=read.csv(paste0(OUTD,other_center_coord,"_cell_",i,"_coordinates.csv"), header = T)
  a=read.csv(paste0(OUTD,nuc_center_coord,"_cell_",i,"_coordinates.csv"),header=T)
  a$what="nucleus"
  dm$what="mito"
  dm=rbind(a[,c("x","y","z","what")],dm[,c("x","y","z","what")])
  dm[,c("x","y")] = round(dm[,c("x","y")] * pixelsize_xy)
  dm$z = round(dm$z * z_interval)
  dm$x=1+dm$x-min(dm$x)
  dm$y=1+dm$y-min(dm$y)
  dm$z=1+dm$z-min(dm$z)
  a=dm[dm$what=="nucleus",]
  b=dm[dm$what=="mito",]
  ## Lims
  xlim<-ylim<- ceil(c(0,110)*pixelsize_xy)
  zlim=ceil(c(0,70)*z_interval)
  b$x[b$x>xlim[2]]=xlim[2]
  b$y[b$y>ylim[2]]=ylim[2]
  ##Save plot
  hull=try(Plot_ConcaveHull(a$x, a$y, a$z, lcolor =1, alpha=0.15,add = F,xlim=xlim,ylim=ylim,zlim=zlim, other=b[,c("x","y","z")], png=paste0(OUTD,filesep,"cell_",i,".png")))
  if(class(hull)=="try-error"){
    return();
  }
  # rgl::movie3d(movie=paste0("cell_",i),
  #              rgl::spin3d( axis = c(1, 0, 1), rpm = 8),
  #              duration = 4, dir = OUTD, type = "gif", clean = TRUE)
  # rgl::close3d()
  saveRDS(hull$nucleus, paste0(OUTD,filesep,"nucleus_",i,".rds")); ## write 3D array
  saveRDS(hull$other, paste0(OUTD,filesep,"mito_",i,".rds")); ## write 3D array
  ## Save as multi channel tiff
  tif_=array(NA, c(dim(hull$nucleus),2))
  tif_[,,,1]=hull$nucleus
  tif_[,,,1]=tif_[,,,1]/max(tif_[,,,1])
  tif_[,,,2]=hull$other*0.3
  
  write_tif(tif_, paste0(OUTD,filesep,"nucleus_",i,".ome.tiff"), overwrite=T)
  # tif=stack()
  # for(j in 1:dim(tif_)[3]){
  #   tif=addLayer(tif,raster(tif_[,,j]))
  # }
  # writeRaster(tif,file=paste0(OUTD,filesep,"nucleus_",i,".ome.tiff"), overwrite=T)
  # bioimagetools::writeTIF(tif_, file=paste0(OUTD,filesep,"nucleus_",i,".ome.tiff"))
  
}
