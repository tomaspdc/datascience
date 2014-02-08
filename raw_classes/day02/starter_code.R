library(class)
library(ggplot2)

data<-iris            # we are going to work with the iris dataset
labels <-data$Species # we want to store the labels independently
data$Species <- NULL  # and we want to remove the labels from the dataset

set.seed(10)    # random seed (we will have the same random results every time)
train.pct<-0.7  # going to take 70% as training set
N<-nrow(data)   # N is the number of rows in the data

train.index <- sample(1:N,train.pct*N) 
# sample(a,b): takes a sample of size b from a
# be careful, returns indexes, not the reduced dataset

train.data<-data[train.index,] # this gets the indexes from data
test.data<-data[-train.index,] # use the negative to get the set difference (inverse indexes)

train.labels <- as.factor(as.matrix(labels)[train.index,])
test.labels <- as.factor(as.matrix(labels)[-train.index,])
# as.factor: encodes a vector as a factor
# as.matrix: encodes a set values as a matrix

##build model and apply
knn.fit <- knn(train = train.data,         # training set
               test = test.data,           # test set
               cl = train.labels,          # true labels
               k = 3)

table(test.labels,knn.fit)

################ Assignment. Perform cross validation
###########Extension. Perform training/test in an R function.
##Plot error as k increases at given number of folds.

data<-iris
labels <-data$Species
data$Species <- NULL

set.seed(10)
train.pct<-0.7
N<-nrow(data)

do_knn_fit<-function(folds, fold_index){
  testing_indexes<-c(1:N)[folds==fold_index]
  # remember this way of accessing datasets, folds has the same length as data
  # but is just a vector populated with 1:n_folds equally
  testing.data<-data[testing_indexes,]
  # it means that now testing.data is just 1/n_folds size, only the data that belonged
  training.data<-data[-testing_indexes,]
  training.labels<-as.factor(as.matrix(labels)[-testing_indexes,])
  fit<-knn(train = training.data,test = testing.data,cl = training.labels,k = 3)
  return(table(test.labels,knn.fit))
}

n_folds<-5
folds<-sample(1:n_folds,N,replace=T) 
# `folds` is just a 150-long vector populated with 1,...,n_folds equally at random
for(fold_index in c(1:n_folds)){ 
  fit<-do_knn_fit(folds,fold_index)
}

