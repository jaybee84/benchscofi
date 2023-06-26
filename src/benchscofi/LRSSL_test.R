#!/bin/R

source("LRSSL_func.R")

## TRAINING

Xs <- as.matrix(read.table("../../Xs_df.csv", sep = " ", header = F))
Xp <- as.matrix(read.table("../../Xp_df.csv", sep = " ", header = F))
Y <- as.matrix(read.table("../../Y_df.csv", sep = " ", header = F))

X_lst <- list(Xs)

n <- nrow(Y)
c <- ncol(Y)

k <- 10
S_lst <- lapply(X_lst, function(X) t(X)%*%X/sqrt(colSums(X)%*%t(colSums(X))))
S_lst <- lapply(S_lst, function(S) S - diag(diag(S)))

ST <- matrix(0, nrow = n, ncol = n)
for(i in 1:(n-1)){
  for(j in (i+1):n){
    s <- Xp[Y[i,]==1,Y[j,]==1]
    ST[i,j] <- max(s)
  }
}
ST <- ST + t(ST)
S_lst[[length(S_lst)+1]] <- ST

S_knn_lst <- lapply(S_lst, function(S) get.knn.graph(S, k))
L_lst <- lapply(S_knn_lst, function(Sknn) diag(colSums(Sknn))-Sknn)

mu <- 0.01
lam <- 0.01
gam <- 2
train.res <- lrssl(X_lst, L_lst, Y, length(X_lst), length(L_lst), mu, lam, gam, 8000, 1e-6)

## PREDICTING

Y.pred <- lapply(1:mx, function(i) X_lst[[i]]%*%train.res$Gs[[i]])
Y.pred <- lapply(Y.pred, function(Y) Y/rowSums(Y))
Y.pred <- Y.pred/rowSums(Y.pred)
alpha <- train.res$alpha
Y.pred <- lapply(1:mx, function(i) train.res$alpha[i]*Y.pred[[i]])
apply(Y.pred,1,sum,na.rm = TRUE)
