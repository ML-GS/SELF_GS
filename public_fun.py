#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/25 15:40
# @Author  : Mang
# @Site    : 
# @File    : public_fun.py
# @Software: PyCharm
# import mglearn
import time
import keras
from keras import layers
from keras import models
from keras import optimizers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import keras.backend as K
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression, TweedieRegressor, ElasticNet, BayesianRidge,Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor


def r_2(y_true, y_pred):
    r = np.cov(y_pred, y_true)[0, 1] / (np.var(y_pred) * np.var(y_true)) ** 0.5
    return r ** 2


def read_data(specie, type, trait):
    X = pd.read_pickle("data//%s_%s" % (type, specie))
    print("data//phe_%s" % specie)
    if type != "G":
        X = X.astype("int")
    phe = pd.read_pickle("data//phe_%s" % specie)
    X = np.array(X)
    if specie == "cow":
        return X, np.array(phe)[:, trait]
    else:
        pos = phe.iloc[:, trait].notnull()
        return X[pos, :], np.array(phe)[pos, trait]


def read_data_bayes(specie, type, trait):
    X = pd.read_pickle("bayes//%s_%s" % (type, specie))
    print("bayes//phe_%s" % specie)
    if type != "G":
        X = X.astype("int")
    phe = pd.read_pickle("bayes//phe_%s" % specie)
    X = np.array(X)
    if specie == "cow":
        return X, np.array(phe)[:, trait]
    else:
        pos = phe.iloc[:, trait].notnull()
        return X[pos, :], np.array(phe)[pos, trait]


def data_split(X, y, indexs, type):
    y_train = y[indexs[0]]
    y_test = y[indexs[1]]
    if type == "G":
        X_train = X[indexs[0], :]
        X_test = X[indexs[1], :]
        # X_train = X[indexs[0], :]
        # X_test = X[indexs[1], :]
    else:
        X_train = X[indexs[0], :]
        X_test = X[indexs[1], :]
    return X_train, X_test, y_train, y_test


def data_split_gblup(X, y, indexs):
    y_train = y[indexs[0]]
    y_test = y[indexs[1]]
    X_train = X[indexs[0], :]
    X_test = X[indexs[1], :]
    return X_train, X_test, y_train, y_test


# def heat(results, params,method):
#     results = pd.DataFrame(results)
#     param = params[method]
#     ps = [i for i in param.keys()]
#     p_1 = param[ps[0]]
#     p_2 = param[ps[1]]
#     m,n = len(p_1),len(p_2)
#     scores = np.array(results.mean_test_score).reshape(m,n)
#     mglearn.tools.heatmap(scores, xlabel=ps[0],
#                             xticklabels=p_1,
#                             ylabel=ps[1],
#                             yticklabels=p_2,
#                             cmap="viridis")
#     plt.show()

def heat_map(results, params, method):
    results = pd.DataFrame(results)
    param = params[method]
    ps = [i for i in param.keys()]
    p_1 = param[ps[0]]
    p_2 = param[ps[1]]
    m, n = len(p_1), len(p_2)
    scores = np.array(results.mean_test_score).reshape(m, n)
    fig = plt.figure()
    # 定义画布为1*1个划分，并在第1个位置上进行作图
    ax = fig.add_subplot(111)
    # 定义横纵坐标的刻度
    ax.set_yticks(range(len(p_1)))
    ax.set_yticklabels(p_1)
    ax.set_ylabel(ps[0])
    ax.set_xticks(range(len(p_2)))
    ax.set_xticklabels(p_2)
    ax.set_xlabel(ps[1])
    # 作图并选择热图的颜色填充风格，这里选择hot
    im = ax.imshow(scores, cmap=plt.cm.hot_r)
    # 增加右侧的颜色刻度条
    plt.colorbar(im)
    # 增加标题
    plt.title(method)
    plt.ion()
    plt.pause(15)
    plt.close()


def compute(X, y, method, type, k_cv, params):
    kfold = KFold(n_splits=k_cv, shuffle=True, random_state=0)
    kfold_split = kfold.split(y)
    res = []
    methods = {"svr": SVR(),
               "krr": KernelRidge(),
               "enet": ElasticNet()}
    time_use = []
    for i in range(k_cv):
        print(i)
        split_index = kfold_split.__next__()
        indexs = split_index[0], split_index[1]
        X_train, X_test, y_train, y_test = data_split(X, y, indexs, type)
        grid_search = GridSearchCV(methods[method], param_grid=params[method], cv=5, n_jobs=15)
        grid_search.fit(X_train, y_train)
        test_pre = grid_search.predict(X_test)
        # print("test score: ",grid_search.score(X_test, y_test))
        # print("train score: ",grid_search.score(X_train, y_train))
        print(grid_search.best_params_)
        print(method, " accuracy: ", r_2(y_test, test_pre) ** 0.5)
        res.append(r_2(y_test, test_pre) ** 0.5)
        results = pd.DataFrame(grid_search.cv_results_)
        # heat_map(results, params, method)
        time_use.append(results["mean_fit_time"].mean())
        print(" the mean time of each combination", results["mean_fit_time"].mean())
        print( " mean accuracy: ", np.array(res).mean())

    print(method + "-" + type + " mean accuracy: ", np.array(res).mean())
    print(method + "-" + type + " mean std: ", np.array(res).std())
    print(pd.DataFrame(res))
    # print(method + "-" + type + " mean time used: ", np.array(time_use).mean())
    res.append(np.array(res).mean())
    res.append(np.array(res).std())
    res.append(np.array(time_use).mean())
    res = np.array(res)
    np.savetxt("results//%s_%s.txt" % (method, type), res)



def stacking(X, y, k_cv):
    res = []
    estimators = [
        ('krr', KernelRidge(kernel="cosine", alpha=0.001)),
        ('svr', SVR(C=2000, gamma=0.001)),
        ("enet", ElasticNet(alpha=0.00001, l1_ratio=0.0005, max_iter=10000))
    ]
    reg = StackingRegressor(
        estimators=estimators,
        n_jobs=15,
        final_estimator=LinearRegression()
    )
    kfold = KFold(n_splits=k_cv, shuffle=True, random_state=0)
    vaild_split = kfold.split(y)
    for i in range(k_cv):
        split_index = vaild_split.__next__()
        test_index = split_index[1]
        y_test = y[test_index]
        trainval_index = split_index[0]
        X_trainval = X[trainval_index, :]
        X_test = X[test_index, :]
        y_trainval = y[trainval_index]
        reg.fit(X_trainval, y_trainval)
        print((reg.score(X_trainval, y_trainval)) ** 0.5)
        test_pre = reg.predict(X_test)
        print("accuracy: ", (r_2(y_test, test_pre)) ** 0.5)
        res.append(r_2(y_test, test_pre) ** 0.5)
        print("mean acacuracy: ", np.array(res).mean())
    print("mean acacuracy: ", np.array(res).mean())


def bagging(X, y, k_cv):
    kfold = KFold(n_splits=k_cv, shuffle=True, random_state=0)
    regr = BaggingRegressor(base_estimator=BayesianRidge(n_iter=1000),
                            n_estimators=20, random_state=0,
                            max_samples=1.0, max_features=0.7, n_jobs=15)
    # regr = BaggingRegressor(base_estimator=SVR(C=40,gamma=0.01),
    #                         n_estimators=100, random_state=0,
    #                         max_samples=0.8,max_features=0.8,n_jobs=15)
    vaild_split = kfold.split(y)
    for i in range(k_cv):
        split_index = vaild_split.__next__()
        test_index = split_index[1]
        y_test = y[test_index]
        trainval_index = split_index[0]
        X_trainval = X[trainval_index, :]
        X_test = X[test_index, :]
        y_trainval = y[trainval_index]
        regr.fit(X_trainval, y_trainval)
        print((regr.score(X_trainval, y_trainval)) ** 0.5)
        test_pre = regr.predict(X_test)
        print("accuracy: ", (r_2(y_test, test_pre)) ** 0.5)


def boosting(X, y, k_cv):
    kfold = KFold(n_splits=k_cv, shuffle=True, random_state=0)
    regr = AdaBoostRegressor(base_estimator=SVR(C=40, gamma=0.01),
                             random_state=319, n_estimators=40,
                             learning_rate=0.01, loss="square")
    vaild_split = kfold.split(y)
    for i in range(k_cv):
        split_index = vaild_split.__next__()
        test_index = split_index[1]
        y_test = y[test_index]
        trainval_index = split_index[0]
        X_trainval = X[trainval_index, :]
        X_test = X[test_index, :]
        y_trainval = y[trainval_index]
        regr.fit(X_trainval, y_trainval)
        print((regr.score(X_trainval, y_trainval)) ** 0.5)
        test_pre = regr.predict(X_test)
        print("accuracy: ", (r_2(y_test, test_pre)) ** 0.5)



def generate_bayesfile(specie, type, trait, k_cv):
    X, y = read_data_bayes(specie, type, trait)
    kfold = KFold(n_splits=k_cv, shuffle=True, random_state=0)
    kfold_split = kfold.split(y)
    for i in range(k_cv):
        split_index = kfold_split.__next__()
        indexs = split_index[0], split_index[1]
        X_train, X_test, y_train, y_test = data_split(X, y, indexs, type)
        pd.DataFrame(X_train).to_csv("bayes/X_train%s_%dtime%d.txt" % (specie, trait, i), sep=" ", header=None,
                                     index=None, )
        pd.DataFrame(X_test).to_csv("bayes/X_test%s_%dtime%d.txt" % (specie, trait, i), sep=" ", header=None,
                                    index=None, )
        pd.DataFrame(y_train).to_csv("bayes/y_train%s_%dtime%d.txt" % (specie, trait, i), sep=" ", header=None,
                                     index=None)
        pd.DataFrame(y_test).to_csv("bayes/y_test_%s_%dtime%d.txt" % (specie, trait, i), sep=" ", header=None,
                                    index=None)




def dnn(X, y, type, k_cv):
    kfold = KFold(n_splits=k_cv, shuffle=True, random_state=0)
    kfold_split = kfold.split(y)
    res = []
    for i in range(k_cv):
        print(i)
        split_index = kfold_split.__next__()
        indexs = split_index[0], split_index[1]
        X_train, X_test, y_train, y_test = data_split(X, y, indexs, type)
        model = models.Sequential()
        model.add(layers.Dense(32, activation="relu", input_shape=(X.shape[1],)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(32, activation="relu"))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(16, activation="relu"))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(16, activation="relu"))
        # model.add(layers.Dropout(0.05))
        model.add(layers.Dense(1))
        model.compile(loss='mse',
                      optimizer=optimizers.RMSprop(lr=0.0001))
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                       patience=20, verbose=1, mode='auto',
                                                       baseline=None, restore_best_weights=True)
        history = model.fit(X_train, y_train, validation_split=0.2,
                            batch_size=16, epochs=1000, verbose=0, callbacks=[early_stopping])
        pred1 = model.predict(X_test)
        pred1 = np.array([i[0] for i in pred1])
        pred2 = model.predict(X_train)
        pred2 = np.array([i[0] for i in pred2])
        print("accuracy#########################:::::", (r_2(y_test, pred1)) ** 0.5)
        print("accuracy#########################:::::", (r_2(y_train, pred2)) ** 0.5)
        res.append(r_2(y_test, pred1) ** 0.5)
        print(np.array(res).mean())
    print(res)
    print(np.array(res).mean())


def compute_qtlmas(X_train,y_train,X_test,y_test, method,  params):
    methods = {"svr": SVR(),
               "krr": KernelRidge(),
               "enet": ElasticNet()}
    grid_search = GridSearchCV(methods[method], param_grid=params[method], cv=5, n_jobs=15)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    test_pre = grid_search.predict(X_test)
    return test_pre

def stacking_qtlmas(X_trainval, y_trainval,X_test,y_test):
    res = []
    estimators = [
        ('krr', KernelRidge(kernel="cosine", alpha=0.005)),
        ('svr', SVR(C=2500, gamma=0.001)),
        ("enet", ElasticNet(alpha=0.00001, l1_ratio=0.0005, max_iter=10000))
    ]
    reg = StackingRegressor(
        estimators=estimators,
        n_jobs=15,
        final_estimator=LinearRegression()
    )

    reg.fit(X_trainval, y_trainval)
    print((reg.score(X_trainval, y_trainval)) ** 0.5)
    test_pre = reg.predict(X_test)
    return test_pre
