#!/bin/R

## https://raw.githubusercontent.com/LiangXujun/LRSSL/a16a75c028393e7256e3630bc8b7900026061f99/LRSSL.R
get.knn <- function(v, k){
  ind <- order(v, decreasing = T)
  return(ind[1:k])
}

get.knn.graph <- function(S, k){
  n <- nrow(S)
  S.knn <- matrix(0, nrow = n, ncol = n)
  for(i in 1:n){
    ind <- get.knn(S[i,], k)
    S.knn[i,ind] <- 1
    S.knn[ind,i] <- 1
    #S.kkn[i,ind] <- S[i,ind]
    #S.knn[ind,i] <- S[i,ind]
  }
  return(S.knn)
}

lrssl <- function(Xs, Ls, Y, mx, ml, mu, lam, gam, max.iter, eps){
  n <- nrow(Y)
  c <- ncol(Y)
  Gs <- vector("list", mx)
  e1ds <- vector("list", mx)
  ds <- rep(0, mx)
  for(i in 1:mx){
    ds[i] <- nrow(Xs[[i]])
    Gs[[i]] <- matrix(runif(ds[i]*c), nrow = ds[i], ncol = c)
    e1ds[[i]] <- rep(1, ds[i])
  }
  alpha <- rep(1, ml)/ml
  
  check.step <- 20;
  Gs.old <- vector("list", mx)
  As <- vector("list", mx)
  As.pos <- vector("list", mx)
  As.neg <- vector("list", mx)
  Bs <- vector("list", mx)
  Bs.pos <- vector("list", mx)
  Bs.neg <- vector("list", mx)
  L <- matrix(0, nrow = n, ncol = n)
  for(i in 1:ml){
    L <- L + alpha[i]^gam*Ls[[i]]
  }
  
  t <- 0
  while(t < max.iter){
    t <- t + 1
    Q <- Y
    for(i in 1:mx){
      Gs.old[[i]] <- Gs[[i]]
      Q <- Q + mu*t(Xs[[i]])%*%Gs[[i]]
    }
    P <- solve(L + (1 + mx*mu)*diag(1, n, n))
    F.mat <- P%*%Q
    
    for(i in 1:mx){
      As[[i]] <- Xs[[i]]%*%(mu*diag(1,n,n)-mu^2*t(P))%*%t(Xs[[i]]) + lam*(e1ds[[i]]%*%t(e1ds[[i]]))
      As.pos[[i]] <- (As[[i]] + abs(As[[i]]))/2
      As.neg[[i]] <- (abs(As[[i]]) - As[[i]])/2
      Bs[[i]] <- mu*Xs[[i]]%*%P%*%Y
      for(j in 1:mx){
        if(i == j){
          next
        }else{
          Bs[[i]] <- Bs[[i]] + mu^2*Xs[[i]]%*%t(P)%*%t(Xs[[j]])%*%Gs[[j]]
        }
      }
      Bs.pos[[i]] <- (Bs[[i]] + abs(Bs[[i]]))/2
      Bs.neg[[i]] <- (abs(Bs[[i]]) - Bs[[i]])/2
    }
    for(i in 1:mx){
      Gs[[i]] <- Gs[[i]]*sqrt((Bs.pos[[i]] + As.neg[[i]]%*%Gs[[i]])/(Bs.neg[[i]] + As.pos[[i]]%*%Gs[[i]]))
    }
    
    for(i in 1:ml){
      alpha[i] <- (1/sum(diag(t(F.mat)%*%Ls[[i]]%*%F.mat)))^(1/(gam - 1))
    }
    alpha <- alpha/sum(alpha)
    
    L <- matrix(0, nrow = n, ncol = n)
    for(i in 1:ml){
      L <- L + alpha[i]^gam*Ls[[i]]
    }
    
    diff.G <- rep(0, mx)
    for(i in 1:mx){
      diff.G[i] <- norm(Gs[[i]] - Gs.old[[i]], "f")/norm(Gs.old[[i]], "f")
    }
    
    if(t%%check.step == 0){
      mesg <- sprintf("t = %i, diffG mean = %.4e", t, mean(diff.G))
      print(mesg)
    }
    if(mean(diff.G) < eps)
      break
  }
  return(list(Gs = Gs, F.mat = F.mat, alpha = alpha, diff.G = diff.G, t = t))
}

## Test

k <- 10
mu <- 0.01
lam <- 0.01
gam <- 2
tol <- 1e-2
maxiter <- 1000

## TRAIN

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

## PREDICT

Xs <- as.matrix(read.table("../../Xs_df.csv", sep = " ", header = F))
Xp <- as.matrix(read.table("../../Xp_df.csv", sep = " ", header = F))
Y <- as.matrix(read.table("../../Y_df.csv", sep = " ", header = F))

X_lst <- list(Xs)

Y.pred_lst <- lapply(1:length(X_lst), function(i) X_lst[[i]]%*%train.res$Gs[[i]])
Y.pred_lst <- lapply(Y.pred_lst, function(Y) Y/rowSums(Y))
Y.pred_lst <- lapply(1:length(X_lst), function(i) train.res$alpha[i]*Y.pred_lst[[i]])
Y.pred <- Reduce("+", Y.pred_lst)