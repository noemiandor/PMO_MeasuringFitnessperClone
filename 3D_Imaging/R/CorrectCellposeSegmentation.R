CorrectCellposeSegmentation<- function(ID,signal,INDIR,OUTDIR,doplot=0, eps=2.5, minPts = 2, IMPORTALLORGANELLES=T, MINZSLICES=8, doplotcentercoord=c(200,500),ZSTACK_DISTANCE=0.29){
  library(matlab)
  library(geometry)
  library(misc3d)
  library(rgl)
  r3dDefaults$windowRect=c(0,50, 800, 800) 
  
  ## Create output folders
  sapply(c("Cells_center_coordinates","All_Cells_coordinates"), function(x) dir.create(paste0(OUTDIR,filesep,ID,filesep,x), recursive = TRUE))
  
  ############################################################
  ## Correct segmentation: merge ids belonging to same cell ##
  coord_=read.csv(paste0("./",INDIR,filesep,ID, "/Cells_center_coordinates/", signal, "_Cells_Centers.csv"))
  coord_$z=coord_$z*ZSTACK_DISTANCE
  o=dbscan::dbscan(coord_[,c("x","y", "z")],eps = eps, minPts = minPts)
  Sys.sleep(10)
  if (file.exists("Rplots.pdf")){
    file.remove("Rplots.pdf")
  }
  coord_$id=o$cluster
  fr=plyr::count(o$cluster)
  colnames(fr)[1]="newCellID"
  fr=fr[order(fr$freq,decreasing = T),]
  fr=fr[fr$newCellID!=0,]; ##remove noise
  
  fr$datapoints=NA
  newCellCoord=list()
  allNewCellIDs = fr$newCellID
  for(newCell in fr$newCellID){
    ##Gather all coordinates associated with new cell
    oldCells=which(newCell==coord_$id)
    tmp=c()
    for(oldid in oldCells){
      dm_=read.csv(paste0("./",INDIR,filesep,ID, "/All_Cells_coordinates/", signal, "_cell_",oldid,"_coordinates.csv"))
      dm_$oldid=oldid
      tmp=rbind(tmp,dm_)
    }
    ## Filter by zstack representation
    if(length(round(unique(tmp$z),1)) < MINZSLICES){
      allNewCellIDs =setdiff(allNewCellIDs, newCell)
      next
    }
    newCellCoord[[as.character(newCell)]]=tmp
    ## Record # of coordinates:
    fr$datapoints[fr$newCellID==newCell]=nrow(tmp)
  }
  if(isempty(allNewCellIDs)){
    print("None of the cells have sufficient zstack representation. No cells saved")
    return()
  }
  fr = fr[fr$newCellID %in% allNewCellIDs, ]
  ## Stats
  centroids=t(sapply(newCellCoord, function(x) apply(x[,c("x","y","z")],2,mean)))
  colnames(centroids)=c("x","y","z")
  fr=fr[order(fr$datapoints),]
  # hist(fr$freq)
  
  ## Save output
  tmp=sapply(1:length(newCellCoord), function(id) write.csv(newCellCoord[[id]],file=paste0("./",OUTDIR,filesep,ID, "/All_Cells_coordinates/", signal, "_cell_",id,"_coordinates.csv"), row.names = F, quote = F))
  write.csv(centroids,file=paste0("./",OUTDIR,filesep,ID, "/Cells_center_coordinates/", signal, "_Cells_Centers.csv"),quote = F, row.names = F)
  print(paste(nrow(coord_),"cell IDs merged into",length(newCellCoord),"unique IDs"))
  
  ## Diagnsotic plots
  if(doplot==1){
    # coordinates for cells of interest
    o2=flexclust::dist2(centroids[,c("x","y")],doplotcentercoord)
    coi=rownames(centroids)[order(o2)[1:5]]
    plot(centroids[,"x"],-centroids[,"y"],pch=20,col=1+(rownames(centroids) %in% coi))
    # coi=fr$newCellID[seq(1,nrow(fr),by=20)]
    # rgl::open3d()
    add = F
    for(cell in coi){
      a=newCellCoord[[as.character(cell)]]
      hull=Plot_ConcaveHull(a$x, -a$y, a$z, lcolor =which(cell==coi), alpha=0.5,add = add)
      add = T
    }
    view3d( theta = 0, phi = 0, fov=0)
  }else if(doplot==2){
    ## overlay segmentation onto predicted nucleus
    png(file=paste0("~/Downloads/",ID,"_3D_NCI-N87_slice%02d.png"),width = 600,height = 600);
    for(slice in 1:70){
      par(mai=c(0.05,0.25,0.05,0.25))
      img=bioimagetools::readTIF(paste0("A03_allenModel",filesep,ID,filesep,"nucleus.p.tif"))
      bioimagetools::img(t(fliplr(img[,,slice])))
      f_cells=list.files(paste0(OUTDIR,filesep,ID,filesep,"All_Cells_coordinates"), full.names = T, pattern = "nucleus.p_")
      for(f_cell in f_cells){
        dm=read.csv(f_cell)
        dm=dm[abs(dm$z-slice)<=0,]
        dm =dm[sample(nrow(dm),nrow(dm)/2),]
        points(dm$y,dm$x,pch=20, cex=0.01, col=which(f_cell==f_cells))
      }
    }
    dev.off()
    system(paste0("convert -delay 80 ~/Downloads/",ID,"_3D_NCI-N87_slice*.png ~/Downloads/",ID,"_3D_NCI-N87.gif"))
    system(paste0("rm ~/Downloads/",ID,"_3D_NCI-N87_slice*.png"))
  }
  
  ## Also copy mitochondria and cytoplasm (if they exist):
  if(IMPORTALLORGANELLES){
    for(other in c("mito","cyto")){
      for(res in c("Cells_center_coordinates","All_Cells_coordinates")){
        sapply(list.files(paste0(INDIR,filesep,ID,"/",res,"/"),pattern = other,full.names = T), function(x) file.copy(x,paste0(OUTDIR,filesep,ID, "/",res,"/",fileparts(x)$name,".csv")))
      }
    }
  }
}
