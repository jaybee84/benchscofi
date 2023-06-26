#!/bin/R

## https://github.com/LiangXujun/LRSSL/LRSSL.R
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