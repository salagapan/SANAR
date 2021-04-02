function scaledNeighbors = scaleBeatHeights(neighbors,height,neighborheight)
%function scaledNeighbors = scaleBeatHeights(neighbors,height,width,neighborwidths)


scaleFactor = height./neighborheight ;
%scaledNeighbors = neighbors.*repmat(scaleFactor,size(neighbors,1),1);
scaledNeighbors = neighbors * diag(scaleFactor) ;


