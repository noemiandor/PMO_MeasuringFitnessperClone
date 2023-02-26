assignCompartment2Nucleus<-function(other_center_coord, nuc_center_coord,OUTD, save_cell_gif=F){
  dirCreate(OUTD, recursive = T, permission = "a+w")
  pixelsize_xy = 0.232 # um 
  z_interval = 0.29 #  um 
  other_coord=read.csv(file=other_center_coord,check.names = F,stringsAsFactors = F)
  nuc_coord=read.csv(file=nuc_center_coord,check.names = F,stringsAsFactors = F)
  ## pixel to um conversion:
  nuc_coord$z=nuc_coord$z*z_interval
  nuc_coord$x=nuc_coord$x*pixelsize_xy
  nuc_coord$y=nuc_coord$y*pixelsize_xy
  other_coord$z=other_coord$z*z_interval
  other_coord$x=other_coord$x*pixelsize_xy
  other_coord$y=other_coord$y*pixelsize_xy
  
  
  ##@TODO: should not be necessary!
  other_center_coord=strsplit(other_center_coord,"_Cells_")[[1]][1]
  nuc_center_coord=strsplit(nuc_center_coord,"_Cells_")[[1]][1]
  
  ## Mito to nucleus assignment: read full coordinates of top 3 closest nuclei centers and calculate distance to all their coord:
  d=flexclust::dist2(other_coord, nuc_coord)
  i_mito=apply(d,1,function(x) order(x)[1:3], simplify = F)
  i_nuc = rep(NA, length(i_mito))
  for(i in 1:nrow(other_coord)){
    top3closestnuc_coord=c()
    for(j in i_mito[[i]]){
      f_nuc=list.files("../All_Cells_coordinates",pattern =paste0(nuc_center_coord,"_cell_",j,"_"),full.names = T )
      coord_=read.csv(file=f_nuc,check.names = F,stringsAsFactors = F)
      coord_$nucleus=j
      top3closestnuc_coord=rbind(top3closestnuc_coord, coord_)
    }
    d_=flexclust::dist2(other_coord[i,,drop=F], top3closestnuc_coord[,colnames(other_coord)])
    ##Assign mitochondria to closest nucleus
    tmp=top3closestnuc_coord[order(d_)[1:10],]
    tmp=plyr::count(tmp$nucleus)
    i_nuc[i]=tmp$x[which.max(tmp$freq)]
  }
  print(paste("Considering closest edge instead of center lead to",100*sum(i_nuc[1:i]!= apply(d,1,which.min)[1:i])/i, "% mito reassigned to a different nucleus"))
  ## Aggregate mitochondria all_cells_coordinates for each nucleus 
  ## and name output file according to nucleus ID
  for(i in unique(i_nuc)){
    print(paste("cell",i))
    if(file.exists(paste0(OUTD,other_center_coord,"_cell_",i,"_coordinates.csv"))){
      next
    }
    i_mito=which(i==i_nuc)
    dm=c()
    for(j in i_mito){
      f_mito=list.files("../All_Cells_coordinates",pattern =paste0(other_center_coord,"_cell_",j,"_"),full.names = T )
      ## @TODO: should not be necessary
      if(isempty(f_mito)){
        next
      }
      coord=read.csv(file=f_mito,check.names = F,stringsAsFactors = F)
      coord[,other_center_coord]=j
      dm=rbind(dm,coord)
    }
    
    #### WRITE LINKED COMPARTMENTS:
    ## Copy other compartment file (e.g. nucleus)
    f_nuc=list.files("../All_Cells_coordinates",pattern =paste0(nuc_center_coord,"_cell_",i,"_"),full.names = T )
    ## @TODO: should not be necessary
    if(isempty(f_nuc) || is.null(dm)){
      next
    }
    file.copy(f_nuc,paste0(OUTD,nuc_center_coord,"_cell_",i,"_coordinates.csv"))
    ## Also write mitochondria file
    write.csv(dm,paste0(OUTD,other_center_coord,"_cell_",i,"_coordinates.csv"), row.names = F, quote = F)
    
    ##Visualize cell
    if(save_cell_gif){
      visualizeSingleCells(i, other_center_coord, nuc_center_coord,OUTD)
    }
  }
  
}

# setwd("~/Projects/PMO/MeasuringFitnessPerClone/data/GastricCancerCL/3Dbrightfield/NCI-N87/I04_PostProcessCellposeOutput/Cells_center_coordinates")
# OUTD="../../I05_multiOrganelles_Linked/"
# nuc_center_coord="nucleus.p_Cells_Centers.csv";
# other_center_coord="mito.p_Cells_Centers.csv"
# assignCompartment2Nucleus(other_center_coord, nuc_center_coord, OUTD)
# other_center_coord="cytoplasm.p_Cells_Centers.csv"
# assignCompartment2Nucleus(other_center_coord, nuc_center_coord, OUTD)
# 
# f=list.files(OUTD,full.names = T)
# signals_per_id=plyr::count(sapply(strsplit(f,"_"), function(x) x[length(x)-1]))
# ## keep only cells with all three signals:
# toRM=signals_per_id$x[signals_per_id$freq<3]
# for(x in toRM){
#   y=list.files(OUTD,full.names = T,pattern = paste0("_",x,"_"))
#   file.remove(y)
# }
