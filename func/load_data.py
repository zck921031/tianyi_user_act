# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 22:21:37 2015

@author: zck

"""

import numpy as np

"""
return:
    numpy矩阵: (用户总数*天数*网站数)
    name2id {}: 用户名map_to编号(0开始)
    names []: 用户名的list

note:
    第一赛季，数据切换之前(299320*49*10)
    
"""

def load_data(filename, total_day, total_site):
    N = 0
    name2id = {}
    names = []
    kvs = []
    for line in open(filename, 'r'):
        strs = line.split('\t')
        if ( len(strs)!=4 ):
            continue
        name = strs[0]
        if name not in name2id:
            name2id[name] = N
            names.append(name)
            N = N+1
        id = name2id[name]
        day = (int(strs[1][1])-1)*7 + int(strs[1][3])-1
        site = int(strs[2][1:])-1
        assert(day<49 and day>=0)
        assert(site<10 and site>=0)
        assert(id<N and id>=0)
        count = int(strs[3])
        kvs.append( ((id,day,site), count) )
    
    x = np.linspace(0,0, N*total_day*total_site)
    x = x.reshape(N, total_day, total_site)
    for (key,value) in kvs:
        x[key] = value
    return (x, name2id, names)

if __name__ == '__main__':
    (x, name2id, names) = load_data('../dataset/part1-0.txt', 49, 10)
    
