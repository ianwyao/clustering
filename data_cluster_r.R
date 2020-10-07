library(dplyr)
df <- read.csv("track_2019.csv")
mydata<- df[0:9,c('popularity','tempo','danceability')]

# K-Means Cluster Analysis
fit <- kmeans(mydata, 3) # 3 cluster solution
# get cluster means
aggregate(mydata,by=list(fit$cluster),FUN=mean)
mydata_fit <- data.frame(mydata, fit$cluster)


# Ward Hierarchical Clustering
d <- dist(mydata, method = "euclidean") # distance matrix
fit <- hclust(d, method="ward")
plot(fit) # display dendogram
groups <- cutree(fit, k=3) # cut tree into 3 clusters
# draw dendogram with red borders around the 3 clusters
rect.hclust(fit, k=3, border="red")


# Ward Hierarchical Clustering with Bootstrapped p values
library(pvclust)
fit <- pvclust(mydata, method.hclust="ward",
              method.dist="euclidean")
plot(fit) # dendogram with p values
# add rectangles around groups highly supported by the data
pvrect(fit, alpha=.95)

# Model Based Clustering
library(mclust)
fit <- Mclust(mydata)
plot(fit) # plot results
summary(fit) # display the best model
