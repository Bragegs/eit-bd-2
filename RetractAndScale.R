RetractAndScale <- function(data, from_index,to_index,
                            scale=TRUE, plot=FALSE,
                            remove=FALSE, logical_remove_vector=NULL, min_length=NULL){
  
  units <- dim(data)[1] #number of cryptocurrencies in data
  n <- to_index - from_index + 1 #number of observations being retracted
  
  return_matrix <- matrix(0, nrow=units, ncol=n)
  for (i in 1:units){
    return_matrix[i, ] <- as.double(data[i, from_index:to_index])
  }
  
  if(is.null(min_length)){
    min_length = n/7
  }
  
  if(!is.null(logical_remove_vector) && length(logical_remove_vector) != units){
    return("Length of logical_remove_vector is wrong")
  }
  
  if(scale==TRUE){
    return_matrix <- t(apply(return_matrix, 1, function(x)(x-min(x))/(max(x)-min(x))))
  }
  
  if(remove==TRUE && is.null(logical_remove_vector)){ 
    remove <- vector("logical", units)
    
    for(j in 1:units){
      
      if(is.na(return_matrix[j,1])){
        remove[j]=TRUE
      }else{
        remove[j]=FALSE
      }
      
      if(remove[j]==FALSE){ #if not NA
        if(length(unique(return_matrix[j,])) < min_length){
          remove[j]=TRUE
        }else{
          remove[j]=FALSE
        } 
      }
      
      
    }
    return_matrix <- return_matrix[remove==FALSE, ]
  }else if(remove==TRUE && !is.null(logical_remove_vector)){
    return_matrix <- return_matrix[logical_remove_vector==FALSE, ]
  }

  if(plot==TRUE){
    matplot(t(return_matrix), type="l", xlab = "", ylab="")
  }
  
  return(list("return"=return_matrix, "remove"=remove))
}
  


EITcluster <- function(data, number_of_clusters = 5, plot=TRUE, prediction = FALSE){
  cluster <- kmeans(data, centers=number_of_clusters)
  
  if(plot==TRUE){
    for(i in 1:number_of_clusters){
      matplot(t(data[cluster$cluster==i,]), type="l", xlab="", ylab=i)
    }
  }
  
  
  return(cluster)
}
  
  
  
  
  
  
  
  
  
