#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/25 18:16
# @Author  : Mang
# @Site    : 
# @File    : operation.py
# @Software: PyCharm


from public_fun import read_data, compute, bagging, stacking,boosting,dnn
import time
import os
os.chdir("D:/python/snpG/cow")



params = {"svr":           # pine
              dict(C=[4000], gamma=[0.01]),
          "krr":
              dict(kernel=["cosine"], alpha=[0.01]),
          "enet":dict(alpha=[0.0001],
                      l1_ratio=[0.3 ],max_iter=[20000])}

X, y = read_data("qtlmas", "G", 0)
# X, y = read_data("cattle", "snp", 0)
# X, y = read_data("pine", "G", 2)
k_cv = 20

# compute(X,y,"svr","G", k_cv,params)
compute(X, y, "enet", "G", k_cv, params)
# compute(X, y, "enet", "G", k_cv, params)
# stacking(X,y,k_cv)
# dnn(X,y,"G",k_cv)


