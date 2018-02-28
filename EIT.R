#t(apply(mydata, 1, function(x)(x-min(x))/(max(x)-min(x))))

mydata = read.csv("/Users/magnlila/PycharmProjects/EiTV18/eit-bd-2/crypto_data.csv")

price <- matrix(0, nrow = 25, ncol = 168)
names <- vector("character")
for (i in 1:25){
  price[i, ] <- as.double(mydata[i, 2:169])
  names[i] <- as.character(mydata[i,]$name)
}
matplot(t(price), type="l")
scaledprice <- apply(price, 1, function(x)(x-min(x))/(max(x)-min(x)))

scaledprice2 <- scaledprice[,c(-10,-18)] #remove iota and tether
names2 <- names[c(-10,-18)] #remove from names
matplot(scaledprice2, type="l")

avgsil <- vector("numeric",length=4)

#Number of clusters 2
cluster2 <- pam(t(scaledprice2), k=2)
avgsil[1] <- cluster2$silinfo$avg.width
attach(mtcars)
par(mfrow=c(1,1))
matplot(scaledprice2[,cluster2$clustering==1], type="l")
matplot(scaledprice2[,cluster2$clustering==2], type="l")

#Number of clusters 3
cluster3 <- pam(t(scaledprice2), k=3)
avgsil[2] <- cluster3$silinfo$avg.width
par(mfrow=c(1,1))
matplot(scaledprice2[,cluster3$clustering==1], type="l")
matplot(scaledprice2[,cluster3$clustering==2], type="l")
matplot(scaledprice2[,cluster3$clustering==3], type="l")

#Number of clusters 4
cluster4 <- pam(t(scaledprice2), k=4)
avgsil[3] <- cluster4$silinfo$avg.width
par(mfrow=c(1,1))
matplot(scaledprice2[,cluster4$clustering==1], type="l")
matplot(scaledprice2[,cluster4$clustering==2], type="l")
matplot(scaledprice2[,cluster4$clustering==3], type="l")
matplot(scaledprice2[,cluster4$clustering==4], type="l")

#Number of clusters 5
cluster5 <- pam(t(scaledprice2), k=5)
avgsil[4] <- cluster5$silinfo$avg.width
par(mfrow=c(1,1))
matplot(scaledprice2[,cluster5$clustering==1], type="l")
matplot(scaledprice2[,cluster5$clustering==2], type="l")
matplot(scaledprice2[,cluster5$clustering==3], type="l")
matplot(scaledprice2[,cluster5$clustering==4], type="l")
matplot(scaledprice2[,cluster5$clustering==5], type="l")

plot(2:5, avgsil, type="b")

avgsilcomplete <- vector("numeric", 14)
for (i in 2:15){
  clusterx <- pam(t(scaledprice2), k=i)
  avgsilcomplete[i-1] = clusterx$silinfo$avg.width
}
plot(2:15, avgsilcomplete, type="b")

#clustering with k-means
kmeans5 <- kmeans(t(scaledprice2), centers=5)
matplot(scaledprice2[,kmeans5$cluster==1], type="l")
matplot(scaledprice2[,kmeans5$cluster==2], type="l")
matplot(scaledprice2[,kmeans5$cluster==3], type="l")
matplot(scaledprice2[,kmeans5$cluster==4], type="l")
matplot(scaledprice2[,kmeans5$cluster==5], type="l")
names2[kmeans5$cluster==3]
names2[kmeans5$cluster==2]


#similar approach with percent change
percent <- matrix(0, nrow = 25, ncol = 168)
for (i in 1:25){
  percent[i, ] <- as.double(mydata[i, 170:(170+167)])
}
matplot(t(percent), type="l")

scaledpercent <- apply(percent, 1, function(x)(x-min(x))/(max(x)-min(x))) #scale?
matplot(scaledpercent, type="l")

scaledpercent2 <- scaledpercent[,c(-10,-18)] #remove tether and iota

percent2 <- percent[c(-10,-18),]
matplot(t(percent2), type="l")

#clustering with k-means
kmeans5p <- kmeans(percent2, centers=5)
matplot(scaledprice2[,kmeans5p$cluster==1], type="l")
matplot(scaledprice2[,kmeans5p$cluster==2], type="l")
matplot(scaledprice2[,kmeans5p$cluster==3], type="l")
matplot(scaledprice2[,kmeans5p$cluster==4], type="l")
matplot(scaledprice2[,kmeans5p$cluster==5], type="l")
names2[kmeans5p$cluster==2]
names2[kmeans5p$cluster==5]


#Similar approach with relative change in volume
volume <- matrix(0, nrow = 25, ncol = 167)
for (i in 1:25){
  volume[i, ] <- as.double(mydata[i, 338:(338+166)])
}
matplot(t(volume), type="l")

scaledvolume <- apply(volume, 1, function(x)(x-min(x))/(max(x)-min(x))) #scale?
matplot(scaledvolume, type="l")

scaledvolume2 <- scaledpercent[,c(-10,-18)] #remove tether and iota, perhaps not tether?

#clustering with k-means
kmeans5v <- kmeans(t(scaledvolume2), centers=5)
matplot(scaledprice2[,kmeans5v$cluster==1], type="l")
matplot(scaledprice2[,kmeans5v$cluster==2], type="l")
matplot(scaledprice2[,kmeans5v$cluster==3], type="l")
matplot(scaledprice2[,kmeans5v$cluster==4], type="l")
matplot(scaledprice2[,kmeans5v$cluster==5], type="l")
names2[kmeans5v$cluster==3]
names2[kmeans5v$cluster==1]

#Similar approach with interest in US
m <- 505 #start interest data
n <- 167 #step length
interestUS <- matrix(0, nrow = 25, ncol = 168)
for (i in 1:25){
  interestUS[i, ] <- as.double(mydata[i, m:(m+n)])
}
matplot(t(interestUS), type="l")

scaledinterestUS <- apply(interestUS, 1, function(x)(x-min(x))/(max(x)-min(x))) #scale?
matplot(scaledinterestUS, type="l")

#clustering with k-means
kmeans5US <- kmeans(t(scaledinterestUS), centers=5)
matplot(scaledprice[,kmeans5US$cluster==1], type="l")
matplot(scaledprice[,kmeans5US$cluster==2], type="l")
matplot(scaledprice[,kmeans5US$cluster==3], type="l")
matplot(scaledprice[,kmeans5US$cluster==4], type="l")
matplot(scaledprice[,kmeans5US$cluster==5], type="l")
names[kmeans5US$cluster==1]
names[kmeans5US$cluster==5]


#clustering based all factors
allfactorsscaled <- matrix(0, nrow=25, ncol=671)
for (i in 1:25){
  allfactorsscaled[i,] = c(scaledprice[,i], scaledpercent[,i], scaledvolume[,i], scaledinterestUS[,i])
}

allfactorsscaled2 <- allfactorsscaled[c(-10,-18),] #removing IOTA and tether


matplot(t(allfactorsscaled2), type="l")
kmeans5all <- kmeans(allfactorsscaled2, centers=5)
matplot(scaledprice2[,kmeans5all$cluster==1], type="l")
matplot(scaledprice2[,kmeans5all$cluster==2], type="l")
matplot(scaledprice2[,kmeans5all$cluster==3], type="l")
matplot(scaledprice2[,kmeans5all$cluster==4], type="l")
matplot(scaledprice2[,kmeans5all$cluster==5], type="l")
names2[kmeans5all$cluster==1]
names2[kmeans5all$cluster==5]

#mean and prediction intervals
matplot(scaledprice2, type="l", col="black")
empmean <- rowMeans(scaledprice2)
lines(empmean, col="red")
RowVar <- function(x) {
  rowSums((x - rowMeans(x))^2)/(dim(x)[2] - 1)
}
empvar <- RowVar(scaledprice2)
predlow <- empmean - 1.96*sqrt(empvar)
predhigh <- empmean + 1.96*sqrt(empvar)
lines(predlow, col="green")
lines(predhigh, col="green")

scaledpricec1 <- scaledprice2[,kmeans5all$cluster==1]
scaledpricec2 <- scaledprice2[,kmeans5all$cluster==2]
scaledpricec3 <- scaledprice2[,kmeans5all$cluster==3]
scaledpricec4 <- scaledprice2[,kmeans5all$cluster==4]
scaledpricec5 <- scaledprice2[,kmeans5all$cluster==5]

matplot(scaledpricec1, type="l", col="black")
empmeanc1 <- rowMeans(scaledpricec1)
lines(empmeanc1, col="red")
empvarc1 <- RowVar(scaledpricec1)
predlowc1 <- empmeanc1 - 1.96*sqrt(empvarc1)
predhighc1 <- empmeanc1 + 1.96*sqrt(empvarc1)
lines(predlowc1, col="green")
lines(predhighc1, col="green")

matplot(scaledpricec2, type="l", col="black")
empmeanc2 <- rowMeans(scaledpricec2)
lines(empmeanc2, col="red")
empvarc2 <- RowVar(scaledpricec2)
predlowc2 <- empmeanc2 - 1.96*sqrt(empvarc2)
predhighc2 <- empmeanc2 + 1.96*sqrt(empvarc2)
lines(predlowc2, col="green")
lines(predhighc2, col="green")

matplot(scaledpricec3, type="l", col="black")
empmeanc3 <- rowMeans(scaledpricec3)
lines(empmeanc3, col="red")
empvarc3 <- RowVar(scaledpricec3)
predlowc3 <- empmeanc3 - 1.96*sqrt(empvarc3)
predhighc3 <- empmeanc3 + 1.96*sqrt(empvarc3)
lines(predlowc3, col="green")
lines(predhighc3, col="green")

matplot(scaledpricec4, type="l", col="black")
empmeanc4 <- rowMeans(scaledpricec4)
lines(empmeanc4, col="red")
empvarc4 <- RowVar(scaledpricec4)
predlowc4 <- empmeanc4 - 1.96*sqrt(empvarc4)
predhighc4 <- empmeanc4 + 1.96*sqrt(empvarc4)
lines(predlowc4, col="green")
lines(predhighc4, col="green")

matplot(scaledpricec5, type="l", col="black")
empmeanc5 <- rowMeans(scaledpricec5)
lines(empmeanc5, col="red")
empvarc5 <- RowVar(scaledpricec5)
predlowc5 <- empmeanc5 - 1.96*sqrt(empvarc5)
predhighc5 <- empmeanc5 + 1.96*sqrt(empvarc5)
lines(predlowc5, col="green")
lines(predhighc5, col="green")

plot(empmeanc1, type="l", col="red")
lines(empmeanc2, type="l", col="green")
lines(empmeanc3, type="l", col="blue")
lines(empmeanc4, type="l", col="black")
lines(empmeanc5, type="l", col="orange")

#time series fitting for cluster
seriesc1 <- rowMeans(scaledpricec1)
series1 <- seriesc1[1:150]
#pacf(seriesc1)
#acf(seriesc1, lag.max=100)
#diffseriesc1 <- diff(seriesc1)
#acf(diffseriesc1, lag.max=100)
#pacf(diffseriesc1, lag.max =100)
model <- arima(series1, order=c(1,1,1))
plot(series1, type="l", xlim=c(0,170))
pred <- predict(model, n.ahead = 18, se.fit=TRUE)
lines(151:168, pred$pred, col="red", type="l")
error <- pred$se
elow <- pred$pred - 1.96*error
ehigh <- pred$pred + 1.96*error
lines(151:168, elow, col="green", type="l")
lines(151:168, ehigh, col="green", type="l")
lines(151:168,scaledpricec1[151:168,1], type="l")
lines(151:168,scaledpricec1[151:168,2], type="l")
lines(151:168,scaledpricec1[151:168,3], type="l")
lines(151:168,scaledpricec1[151:168,4], type="l")
lines(151:168,scaledpricec1[151:168,5], type="l")
lines(151:168,scaledpricec1[151:168,6], type="l")
lines(151:168,scaledpricec2[151:168,1], type="l", col="blue")
lines(151:168,scaledpricec2[151:168,2], type="l", col="blue")
lines(151:168,scaledpricec2[151:168,3], type="l", col="blue")
lines(151:168,scaledpricec4[151:168,1], type="l", col="orange")
lines(151:168,scaledpricec4[151:168,2], type="l", col="orange")
lines(151:168,scaledpricec4[151:168,3], type="l", col="orange")

#time series fitting for cluster 2
seriesc2 <- rowMeans(scaledpricec2)
series2 <- seriesc2[1:150]
#pacf(seriesc1)
#acf(seriesc1, lag.max=100)
#diffseriesc1 <- diff(seriesc1)
#acf(diffseriesc1, lag.max=100)
#pacf(diffseriesc1, lag.max =100)
model2 <- arima(series2, order=c(1,1,1))
plot(series2, type="l", xlim=c(0,170))
pred <- predict(model2, n.ahead = 18, se.fit=TRUE)
lines(151:168, pred$pred, col="red", type="l")
error <- pred$se
elow <- pred$pred - 1.96*error
ehigh <- pred$pred + 1.96*error
lines(151:168, elow, col="green", type="l")
lines(151:168, ehigh, col="green", type="l")
lines(151:168,scaledpricec2[151:168,1], type="l")
lines(151:168,scaledpricec2[151:168,2], type="l")
lines(151:168,scaledpricec2[151:168,3], type="l")
lines(151:168,scaledpricec2[151:168,4], type="l")
lines(151:168,scaledpricec1[151:168,1], type="l", col="blue")
lines(151:168,scaledpricec1[151:168,2], type="l", col="blue")
lines(151:168,scaledpricec1[151:168,3], type="l", col="blue")
lines(151:168,scaledpricec5[151:168,1], type="l", col="orange")
lines(151:168,scaledpricec5[151:168,2], type="l", col="orange")
lines(151:168,scaledpricec5[151:168,3], type="l", col="orange")

#different weigths
allfactorsscaledweighted <- matrix(0, nrow=25, ncol=671)
for (i in 1:25){
  allfactorsscaledweighted[i,] = c(5*scaledprice[,i], scaledpercent[,i], scaledvolume[,i], scaledinterestUS[,i])
}

allfactorsscaledweighted2 <- allfactorsscaledweighted[c(-10,-18),] #removing IOTA and tether

matplot(t(allfactorsscaledweighted2), type="l")

