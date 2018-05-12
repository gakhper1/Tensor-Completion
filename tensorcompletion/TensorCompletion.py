import math as mt
import cv2
import tensorly as tl
import random as rn
import numpy as np
import matplotlib.pyplot as plt
LA = np.linalg

def shrinkage_op(X,tau):
    u , sig, v = LA.svd(X, full_matrices=False)
    sig_diag = sig - tau
    np.maximum(sig_diag,np.zeros(len(sig)),sig_diag)
    shrink = np.dot(u[:,:], np.dot(np.diag(sig_diag[:]), v[:,:]))
    return shrink

def truncate_op(X,tau):
    u , sig, v = LA.svd(X, full_matrices=False)
    sig_o = sig
    t = np.zeros(len(sig))+tau
    np.minimum(sig,t,sig)
    trunk = np.dot(u[:,:], np.dot(np.diag(sig[:]), v[:,:]))
    return trunk, sig_o

def add_noise(im,prcnt):
    shpz = (int(im.shape[0]), int(mt.ceil((prcnt/100) * im.shape[1])), int(im.shape[2]))
    shpo = (int(im.shape[0]), int(mt.floor((1-prcnt/100) * im.shape[1])), int(im.shape[2]))
    zz = np.zeros(shpz)
    oo = 0xFF * np.ones(shpo)
    cct = np.concatenate((zz, oo), 1)
    np.random.shuffle(cct.reshape(-1, 3))
    cct = cct.astype(np.uint8)
    imbn = np.bitwise_and(im, cct)
    return imbn, cct

def eq_svtn(T, b, M, Omega):
    X = 0
    for i in range(3):
        X += b[i]*tl.fold(M[i],i,T.shape)
    X = X/np.sum(b)
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            if all(Omega[i, j, :]!= 0):
                X[i, j, :] = T[i, j, :]
    return X

def update(un,T,Omega):
    a = abs(np.random.standard_normal(3))
    a /= sum(a)
    X = 0
    for i in range(3):
        X += a[i]*tl.fold(un[i],i,T.shape)
    for j in range(T.shape[0]):
        for k in range(T.shape[1]):
            if all(Omega[j,k,:] != 0):
                X[j,k,:] = T[j,k,:]
    return X

def silrtc(X,K,Omega):
    b = [rn.uniform(0.00009, 0.00021), rn.uniform(0.00009, 0.00011), rn.uniform(0.00009, 0.00021)]
    a = np.random.dirichlet(np.ones(3),size=1)
    a = a[0,:]
    M = [0, 0, 0]
    for i in range(K):
        for j in range (3):
            M[j] = shrinkage_op(tl.unfold(X,j),a[j]/b[j])
        X = eq_svtn(X,b,M,Omega)
    return X

def falrtc(c,T,K,Omega):
    X = T
    a = abs(np.random.standard_normal(3))
    a /= sum(a)
    m = a/100000
    L = np.sum(m)
    Z = T; W = T; B = 0
    for k in range(K):
        while True:
            theta = (1+mt.sqrt(1+4*L*B))/(2*L)
            W = (theta/L)/(B+theta/L)*Z + B/(B+theta/L)*X

            # compute f_mu(X), f_mu(W), and gradient of f_mu(W)
            fx = 0; fw = 0; fxp = 0; gw = np.zeros(X.shape)
            for i in range(3):
                [trunkX, sigX] = truncate_op(tl.unfold(X,i),m[i]/a[i])
                [trunkW, sigW] = truncate_op(tl.unfold(W,i),m[i]/a[i])
                fx += np.sum(sigX)
                fw += np.sum(sigW)
                gw += tl.fold((a[i]*a[i]/m[i])*trunkW,i,W.shape)

            # replace the known pixels with zeros in gw
            for nr in range(X.shape[0]):
                for nc in range(X.shape[1]):
                    if all(Omega[nr, nc,:] != 0):
                        gw[nr, nc, :] = 0

            if fx <= fw - (LA.norm(gw)**2)/(2*L):
                break
            Xp = W - gw/L
            for r in range(3):
                [_, sig_fxp] = truncate_op(tl.unfold(Xp,r),m[r]/a[r])
                fxp += np.sum(sig_fxp)
            if fxp <= fw - (LA.norm(gw)**2)/(2/L):
                X = Xp
                break
            L = L/c
        Z = Z - theta*gw
        B = B + theta
    return X

def halrtc(T,rho,K, Omega):
    X = np.zeros(T.shape).astype(np.float64)
    a = abs(np.random.standard_normal(3))
    a = a/sum(a)
    Mi = [np.zeros(T.shape),np.zeros(T.shape),np.zeros(T.shape)]
    Yi = [np.zeros(T.shape),np.zeros(T.shape),np.zeros(T.shape)]
    for k in range(K):
        for i in range(3):
            Mi[i] = tl.fold(shrinkage_op(tl.unfold(T,i) +
                                         tl.unfold(Yi[i],i)/rho,a[i]/rho),i,T.shape)
            X += (Mi[i] - Yi[i]/rho)
        X /= 3
        for j in range(X.shape[0]):
            for k in range(X.shape[1]):
                if all(Omega[j, k,:] != 0):
                    X[j,k,:] = T[j,k,:]
        for n in range(3):
            Yi[n] = Yi[n] - rho*(Mi[n]-X)
        rho *= 1.1
    return X

def tmacTT(T,K,Omega):
    X = T
    U = []
    V = []
    un = []
    for n in range(3):
        un.append(tl.unfold(X,n))
        rank = LA.matrix_rank(un[n])
        U.append(np.ones([un[n].shape[0],rank]))
        V.append(np.ones([rank,un[n].shape[1]]))
    for k in range(K):
        for i in range(3):
            un[i] = tl.unfold(X,i)
            U[i] = np.matmul(un[i],np.matrix.transpose(V[i]))
            U_t = np.matrix.transpose(U[i])
            V[i] = np.matmul(np.matmul(LA.pinv(np.matmul(U_t,U[i])),U_t),un[i])
            un[i] = np.matmul(U[i],V[i])
        X = update(un,T,Omega)
    return X

def get_Rse(original, estimate):
    error = LA.norm(original - estimate) / LA.norm(original)
    return error

lena = cv2.imread('C:\\Users\\george\\Desktop\\lena.png')
brx2 = cv2.imread('C:\\Users\\george\\Desktop\\brx2.jpg')
tree = cv2.imread('C:\\Users\\george\\Desktop\\tree.jpg')
gold = cv2.imread('C:\\Users\\george\\Desktop\\gold.jpg')


percent = 85

lena_noise, Omega_lena = add_noise(lena,percent)
brx2_noise, Omega_brx2 = add_noise(lena,percent)
tree_noise, Omega_tree = add_noise(lena,percent)
gold_noise, Omega_gold = add_noise(lena,percent)

lena_silrtc = silrtc(lena_noise,32,Omega_lena)
brx2_falrtc = falrtc(0.5,brx2_noise,300,Omega_brx2)
tree_halrtc = halrtc(tree_noise,.000004,64,Omega_tree)
gold_halrtc = tmacTT(gold_noise,64,Omega_gold)

RSE0 = get_Rse(lena, lena_noise)
RSE1 = get_Rse(brx2, brx2_noise)
RSE2 = get_Rse(tree, tree_noise)
RSE3 = get_Rse(gold, gold_noise)


cv2.imshow('xx1', lena_silrtc.astype(np.uint8))
cv2.imshow('xx2', brx2_falrtc.astype(np.uint8))
cv2.imshow('xx3', tree_halrtc.astype(np.uint8))
cv2.imshow('xx4', gold_halrtc.astype(np.uint8))



