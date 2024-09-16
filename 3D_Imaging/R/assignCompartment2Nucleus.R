assignCompartment2Nucleus<-function(MITOTIF, CYTOTIF,OUTD, nuc_center_coord, save_cell_gif=F){
  dirCreate(OUTD, recursive = T, permission = "a+w")
  # pixelsize_xy = 0.232 # um 
  # z_interval = 0.29 #  um 
  
  xyz=c("x", "y", "z")
  EXPANSIONFACTOR=1; #0.5
  alpha=list(cytoplasm.p=0.05,cytoplasm.t=0.05,nucleus.t=1,nucleus.p=1,mito.t=0.05,mito.p=0.05)
  col_org=alpha
  col_org[names(alpha)]=rainbow(length(alpha))
  
  ## Nuclei
  nuc_coord=read.csv(file=nuc_center_coord,check.names = F,stringsAsFactors = F)
  nuc_center_coord=strsplit(nuc_center_coord,"_Cells_")[[1]][1]
  
  PATH2TIF=list(mito.p=MITOTIF, cytoplasm.p=CYTOTIF)
  assigned_coord<-organelle_coord<-list()
  for(organelle in names(PATH2TIF)){
    # organelle=fileparts(MITOTIF)$name
    img=bioimagetools::readTIF(PATH2TIF[[organelle]]); 
    # img=img[,,35:36]; ##@TODO: remove to include all slices
    MINS=quantile(img,0.90); #0.97
    coord = as.data.frame(which(img> MINS,arr.ind = T))
    colnames(coord) = c("y", "x", "z")
    coord = coord[,xyz]
    coord$signal=img[img> MINS]
    organelle_coord[[organelle]]=coord
    ## Initialize
    assigned_coord[[organelle]]=list()
  }
  # mito_coord$z_adj=mito_coord$z*z_interval
  # mito_coord$x_adj=mito_coord$x*pixelsize_xy
  # mito_coord$y_adj=mito_coord$y*pixelsize_xy
  
  f_nuc=list.files("../All_Cells_coordinates",pattern =paste0(nuc_center_coord,"_cell_"),full.names = T )
  coord_nuc_all=sapply(f_nuc, function(x) read.csv(file=x,check.names = F,stringsAsFactors = F), simplify = F)
  names(coord_nuc_all)=sapply(names(coord_nuc_all), function(x) strsplit(fileparts(x)$name,"_")[[1]][3])
  
  ## cluster organelle pixels in vincinty of each nucleus
  for(j in 1:nrow(nuc_coord)){
    coord_=coord_nuc_all[[as.character(j)]]
    
    rspan= apply(coord_,2,quantile,c(0,1))
    tmp=EXPANSIONFACTOR*(rspan[2,]-rspan[1,])
    rspan[1,] = rspan[1,] - tmp
    rspan[2,] = rspan[2,] + tmp
    
    coord_$signal=1
    coord_$what="nucleus.p"
    coord_$id=rownames(coord_)
    for(organelle in names(organelle_coord)){
      o_coord =organelle_coord[[organelle]]
      ii=which(o_coord$y>rspan[1,"y"] & o_coord$y<rspan[2,"y"] & o_coord$x>rspan[1,"x"] & o_coord$x<rspan[2,"x"] & o_coord$z>rspan[1,"z"] & o_coord$z<rspan[2,"z"])
      o_coord_ = o_coord[ii, c(xyz,"signal")]
      o_coord_$what=organelle
      o_coord_$id=rownames(o_coord_)
      coord_=rbind(coord_[,c(xyz,"what","signal","id")],o_coord_[,c(xyz,"what","signal","id")])
    }
    coord_$cell=j
    clusters=dbscan::dbscan(coord_[,c("x", "y", "z")], eps=1,minPts = 2, weights=coord_$signal)
    # clusters=dbscan::dbscan(mito_coord[,c("x_adj", "y_adj", "z_adj")], eps=z_interval,minPts = 1, weights=mito_coord$signal)
    fr=plyr::count(clusters$cluster)
    print(paste("Found",nrow(fr),"clusters"))
    print(paste(100*fr[1,2]/nrow(mito_coord),"% of datapoints clustered as noise"))
    print(vegan::diversity(fr$freq/sum(fr$freq),"simpson"))
    
    # ## plot
    # try(rgl.close())
    # tmp = quantile(coord_$z,c(0,1))
    # space=tmp[2]-tmp[1]
    # zlim=c(tmp[1]-space/2, tmp[2]+space/2)
    # jj=which(clusters$cluster>0)
    # jj=jj[sample(length(jj),length(jj)*0.25)]
    # rgl::plot3d(coord_$x[jj], coord_$y[jj], coord_$z[jj], pch3d=20, size=2, zlim=zlim,axes=F, xlab="",ylab="", zlab="",col=clusters$cluster[jj]+1,alpha=alpha[coord_$what[jj]], add=F)
    # # rgl::plot3d(coord_$x[jj], coord_$y[jj], coord_$z[jj], pch3d=20, size=2, zlim=zlim,axes=F, xlab="",ylab="", zlab="",col=col_org[coord_$what[jj]],alpha=alpha[coord_$what[jj]], add=F)
    
    ## find out which cluster has nucleus of interest in it:
    fr=plyr::count(clusters$cluster[coord_$what=="nucleus.p"])
    coi=fr$x[which.max(fr$freq)]
    ## Assign mito and cyto coordinates to that nucleus if they are members of same cluster
    for(organelle in names(organelle_coord)){
      ii=which(clusters$cluster==coi & coord_$what==organelle);
      assigned_coord[[organelle]][[as.character(j)]]=coord_[ii,,drop=F]
    }
    assigned_coord[["nucleus.p"]][[as.character(j)]]=coord_[coord_$what=="nucleus.p",,drop=F]
  }
  
  ## correct doubly assigned coordinates
  for(organelle in names(organelle_coord)){
    X=assigned_coord[[organelle]]
    
    fr=sapply(X, function(x) x$id)
    fr=plyr::count(unlist(fr))
    ambg_i = fr$x[fr$freq>1]
    print(paste(100*length(ambg_i)/sum(sapply(X,nrow)),"%", organelle,"pixels have ambiguous cell assignment"))
    ambg=sapply(X, function(x) x[ x$id %in% ambg_i,,drop=F], simplify = F)
    ambg=do.call(rbind, ambg) 
    
    ## exclude ambiguous assignments per each cell
    for(x in unique(ambg$cell)){
      tmp=ambg[ambg$cell==x,]
      i=as.character(x)
      X[[i]] =X[[i]][!X[[i]]$id %in% tmp$id,,drop=F]
    }
    assigned_coord[[organelle]] = X
    
    # ## ambiguous cell assignment per each organelle coordinate: too slow
    # for(x in unique(ambg$id)){
    #   tmp=ambg[ambg$id==x,]
    #   d=sapply(coord_nuc_all[as.character(tmp$cell)], function(x) min(dist2(x[,xyz], tmp[1,xyz, drop=F])) )
    #   cell=names(which.min(d))
    #   
    #   for(i in setdiff(as.character(tmp$cell), cell)){
    #     X[[i]] = X[[i]][X[[i]]$id != tmp$id[1],]
    #   }
    # }
  }
  
  # ## exclude nuclei without any assigned 
  # counts=sapply(names(assigned_coord$nucleus.p), function(x) c(nrow(assigned_coord$mito.p[[x]]), nrow(assigned_coord$cytoplasm.p[[x]])))
  
  ## DBSCAN each type of intra-cell organelles; report summary stats
  for(organelle in names(organelle_coord)){
    X=assigned_coord[[organelle]]
    print(paste(100*sum(sapply(X,nrow))/nrow(organelle_coord[[organelle]]),"%", organelle,"pixels have been assigned to a cell"))
    
    for(i in names(X)){
      coord_ =  X[[i]]
      if(nrow(coord_)>0){
        clusters_=dbscan::dbscan(coord_[,c("x", "y", "z")], eps=1,minPts =2, weights=coord_$signal)
        # clusters_=dbscan::dbscan(coord_[,c("x", "y", "z")], eps=2,minPts =1, weights=coord_$signal)
        coord_[,organelle]=clusters_$cluster+1
        # plyr::count(clusters_$cluster)
        # try(rgl.close())
        # rgl::plot3d(coord_$x, coord_$y, coord_$z, pch3d=20, size=2, axes=F, xlab="",ylab="", zlab="",col=clusters_$cluster+1,add=F)
        
        ## Save output
        write.csv(coord_[,c("x", "y", "z", "signal", organelle)],paste0(OUTD,organelle,"_cell_",i,"_coordinates.csv"), row.names = F, quote = F)
      }
      ## Also save nucleus output
      write.csv(assigned_coord$nucleus.p[[i]][,c("x", "y", "z")],paste0(OUTD,"nucleus.p_cell_",i,"_coordinates.csv"), row.names = F, quote = F)
      ##Visualize cell
      if(save_cell_gif){
        visualizeSingleCells(i, organelle, nuc_center_coord,OUTD)
      }
    }
  }
  
  
}
