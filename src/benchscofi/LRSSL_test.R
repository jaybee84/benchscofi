#!/bin/R

source("LRSSL_func.R")

k <- 10
mu <- 0.01
lam <- 0.01
gam <- 2
tol <- 1e-2
maxiter <- 1000

## TRAINING

Xs <- as.matrix(read.table("../../Xs_df.csv", sep = " ", header = F))
Xp <- as.matrix(read.table("../../Xp_df.csv", sep = " ", header = F))
Y <- as.matrix(read.table("../../Y_df.csv", sep = " ", header = F))

X_lst <- list(Xs)

S_lst <- lapply(X_lst, function(X) t(X)%*%X/sqrt(colSums(X)%*%t(colSums(X))))
S_lst <- lapply(S_lst, function(S) S - diag(diag(S)))

ST <- matrix(0, nrow = nrow(Y), ncol = nrow(Y))
for(i in 1:(nrow(Y)-1)){
  for(j in (i+1):nrow(Y)){
    s <- Xp[Y[i,]==1,Y[j,]==1]
    ST[i,j] <- max(s)
  }
}
ST <- ST + t(ST)
S_lst[[length(S_lst)+1]] <- ST

S_knn_lst <- lapply(S_lst, function(S) get.knn.graph(S, k))
L_lst <- lapply(S_knn_lst, function(Sknn) diag(colSums(Sknn))-Sknn)

train.res <- lrssl(X_lst, L_lst, Y, length(X_lst), length(L_lst), mu, lam, gam, maxiter, tol)

## PREDICTING

Xs <- as.matrix(read.table("../../Xs_df.csv", sep = " ", header = F))
Xp <- as.matrix(read.table("../../Xp_df.csv", sep = " ", header = F))
Y <- as.matrix(read.table("../../Y_df.csv", sep = " ", header = F))

X_lst <- list(Xs)

Y.pred_lst <- lapply(1:length(X_lst), function(i) X_lst[[i]]%*%train.res$Gs[[i]])
Y.pred_lst <- lapply(Y.pred_lst, function(Y) Y/rowSums(Y))
Y.pred_lst <- lapply(1:length(X_lst), function(i) train.res$alpha[i]*Y.pred_lst[[i]])
Y.pred <- Reduce("+", Y.pred_lst)
