source("RetractAndScale.R")

mydata = read.csv("/Users/magnlila/Downloads/crypto_data (2).csv")
explanation <- read.csv("/Users/magnlila/Downloads/column_explanation (1).csv")

units <- dim(mydata)[1] #number of cryptocurrency in data
names <- vector("character", units)
for (i in 1:units){
  names[i] <- as.character(mydata[i,]$name)
}

variables <- dim(explanation)[1]
a <- as.matrix(explanation)
variable_names <- rownames(explanation)
remove_matrix <- matrix(FALSE, nrow <- variables-1, ncol <- units)
for (i in 2:variables){
  if(variable_names[i]!="name"){
    from <- as.double(a[i,1])+2
    to <- as.double(a[i,2])+2
    b <- RetractAndScale(mydata, from, to, remove=TRUE, plot=TRUE)
    assign(variable_names[i],b$return)
    title(main=variable_names[i]) #labels plot if plot=TRUE in RetractAndScale
    remove_matrix[i-1, ] <- b$remove
  }
}

a <- vector("numeric", variables-1)
for(i in 1:variables-1){
  a[i] <- length(remove_matrix[remove_matrix[i, ]==TRUE, ])
}


