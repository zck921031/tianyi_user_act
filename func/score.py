# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 22:21:37 2015

@author: zck
"""

import sys
sys.path.append('../func')

from load_data import load_data
import numpy as np
import theano.tensor as T


def cos_sim(a,b):
    s = np.sum(np.multiply(a,b),axis=1)
    t = np.multiply(np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1) )
    return s/np.maximum(t, np.finfo(t.dtype).tiny)

def score(a,b):
    sim = cos_sim(a,b)
    precision = sim.mean()
    recall = 1
    f1  = 2.0*precision*recall/(precision+recall)
    return (precision, recall, f1)

def _squared_magnitude(x):
    return T.sqr(x).sum(axis=1)
def _magnitude(x):
    return T.sqrt(T.maximum(_squared_magnitude(x), np.finfo(x.dtype).tiny))
    #return T.sqrt(T.maximum(_squared_magnitude(x), 0.1))
def cosine(x, y):
    return ( (x * y).sum(axis=1) / (_magnitude(x) * _magnitude(y)) )


if __name__ == '__main__':
    (x, name2id, names) = load_data('../dataset/part1-0.txt', 49, 10)
    N = x.shape[0]
    p = x.reshape(N,7,7,10)[:,:6,:,:].mean(axis=1).reshape(N,70)
    #sim = cos_sim(p, x.reshape(N,7,7,10)[:,6,:,:].reshape(N,70) )
    sco = score(p, x.reshape(N,7,7,10)[:,6,:,:].reshape(N,70) )
    print( 'score = ' + str(sco) )
    
    