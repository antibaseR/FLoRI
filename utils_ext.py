#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 11:36:53 2022

@author: Xinlei Chen
"""


import os
import random
import pandas as pd
import numpy as np

## https://developer.apple.com/metal/tensorflow-plugin/
## /Users/xtan/miniforge3/bin/python3
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.tree import export_text

from sklearn.model_selection import GridSearchCV

import collections

# pd.options.mode.chained_assignment = None  # default='warn'


def set_random(seed_value):
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    return


def normalize(df_tr, df_te, names_to_norm):
    df = df_tr.copy()
    df_test = df_te.copy()

    # normalize X & S
    scaler = StandardScaler()
    df[names_to_norm] = scaler.fit_transform(df[names_to_norm])
    df_test[names_to_norm] = scaler.transform(df_test[names_to_norm])

    return(df.copy(), df_test.copy())


def get_ps_fit(df_train, use_covars): #, mod, is_tune
    df_tr = df_train.copy()

    #TODO: tune PS fit, right now only rf with default params
    # params = {'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 100}
    # regr = RandomForestClassifier(random_state=1234, 
    #                                 n_estimators=params["n_estimators"], 
    #                                 min_samples_split=params["min_samples_split"], 
    #                                 max_features=params["max_features"],
    #                                 ).fit(df_tr[use_covars], df_tr["A"])

    regr = RandomForestClassifier(random_state=1234).fit(df_tr[use_covars], df_tr["A"])
    
    return(regr)


def keras_wrap(x_train, train_labels, train_wts, x_test, loss_fn, act_out, 
               layer=2, node=1024, dropout=0.2, n_epoch=100, bsize=64, act="relu", 
               opt="Adam", val_split=0.2, is_early_stop=True, verb=0):

    if is_early_stop:
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
        callback = [early_stop]
    else:
        callback = None

    # set input_dim for the number of features
    if len(x_train.shape) == 1:
        input_dim = 1
    else:
        input_dim = x_train.shape[1]
    
    # model
    model = Sequential()
    for i in range(layer):
        if i==0:
            model.add(Dense(node, input_dim=input_dim, activation=act)) # Hidden 1
            model.add(Dropout(dropout))
        else:
           model.add(Dense(node, activation=act)) # Hidden 2 
           model.add(Dropout(dropout))
           
    model.add(Dense(1, activation=act_out)) # Output
    
    model.compile(loss=loss_fn, optimizer=opt)
    model.fit(x_train, train_labels, 
              sample_weight=train_wts,
              epochs=n_epoch, batch_size=bsize,
              validation_split=val_split, callbacks=callback, verbose=verb)
    
    # predict
    pred_test = model.predict(x_test).flatten()
    pred_train = model.predict(x_train).flatten()
    return pred_test, pred_train, model


def hyper_tuning(x_train, train_labels, train_wts, loss_fn, act_out,
                 param_grid, n_cv=5, n_jobs=1):
    """
    layers = [2,3]
    nodes=[100,300,512]
    dropout=[0.2] #,0.4
    activation = ["sigmoid","relu"]
    optimizer = ["adam"] #,"nadam"
    bsize = [32,64] #,128
    n_epochs = [50,100] #,200

    bst_params = hyper_tuning(x_train, train_labels, train_wts, loss_fn, act_out,
                              param_grid, n_cv=5, n_jobs=1)
    """
    # set input_dim for the number of features
    if len(x_train.shape) == 1:
        input_dim = 1
    else:
        input_dim = x_train.shape[1]
    
    def create_model(layers,nodes,activation,optimizer,dropout):
        model = Sequential()
        for i in range(layers):
            if i==0:
                model.add(Dense(nodes, input_dim=input_dim))
                model.add(Activation(activation))
                model.add(Dropout(dropout))
            else:
                model.add(Dense(nodes, activation=activation)) 
                model.add(Activation(activation))
                model.add(Dropout(dropout))
    
        model.add(Dense(units=1, activation=act_out))
        model.compile(optimizer=optimizer, loss=loss_fn)
        return model

    if act_out == "sigmoid": #for classification
        model = KerasClassifier(build_fn=create_model, verbose=0)
    else: #None #for regression including quantile
        model = KerasRegressor(build_fn=create_model, verbose=0)
    
    assert param_grid is not None
    layers = param_grid['layers']
    nodes = param_grid['nodes']
    dropout = param_grid['dropouts']
    activation = param_grid['acts']
    optimizer = param_grid['opts']
    bsizes = param_grid['bsizes']
    n_epochs = param_grid['n_epochs']

    param_grid = dict(layers=layers, nodes=nodes, activation=activation, optimizer=optimizer, 
                      dropout=dropout, batch_size=bsizes, epochs=n_epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=n_cv, n_jobs=n_jobs)
    
    grid_result = grid.fit(x_train, train_labels, sample_weight=train_wts)
    
    print("hyperparams tuning:", grid_result.best_score_,grid_result.best_params_)

    bst_params = grid_result.best_params_
    layer = bst_params['layers']
    node = bst_params['nodes']
    dropout = bst_params['dropout']
    n_epoch = bst_params['epochs']
    bsize = bst_params['batch_size']
    act = bst_params['activation']
    opt = bst_params['optimizer']

    return layer, node, dropout, n_epoch, bsize, act, opt


def fit_expectation(obj_name, df, use_covars, df_pred, is_class, is_tune, param_grid, df_val=None):
    """E(Y|X=x,A=a) or E(Y|X=x,S=s,A=a)
    (model separated by A)
    """
    df_tr = df.copy()
    df_te = df_pred.copy()
    if df_val is not None:
        df_va = df_val.copy()

    # fit on df0/df1
    if is_class:
        loss_fn = 'binary_crossentropy'
        act_out = 'sigmoid'
    else:
        loss_fn = "mean_squared_error"
        act_out = None
    
    if is_tune:
        layer, node, dropout, n_epoch, bsize, act, opt = \
                hyper_tuning(df_tr[use_covars], df_tr["Y"], None, loss_fn, act_out,
                    param_grid, n_cv=5, n_jobs=1)
        Yhat, _, regr = keras_wrap(df_tr[use_covars], df_tr["Y"], None, 
                            df_te[use_covars], loss_fn, act_out, 
                            layer, node, dropout, n_epoch, bsize, act, opt, 
                            val_split=None, is_early_stop=False, verb=0)
    else:
        if is_class:
            Yhat, _, regr = keras_wrap(df_tr[use_covars], df_tr["Y"], None, 
                                df_te[use_covars], loss_fn, act_out, 
                                layer=1, node=1024, dropout=0.2, n_epoch=10, bsize=8, act="relu", opt="Adam", 
                                val_split=0.2, is_early_stop=True, verb=0)
            
        else:
            Yhat, _, regr = keras_wrap(df_tr[use_covars], df_tr["Y"], None, 
                                df_te[use_covars], loss_fn, act_out, 
                                layer=2, node=1024, dropout=0.2, n_epoch=100, bsize=64, act="relu", opt="Adam", 
                                val_split=0.2, is_early_stop=True, verb=0)
            # Yhat, _, regr = keras_expectation(df_tr[use_covars], df_tr["Y"], df_te[use_covars], loss)

    if df_val is not None:
        if not is_class: #NRMSE
            y_tr = regr.predict(df_tr[use_covars])
            rms = mean_squared_error(df_tr['Y'], y_tr, squared=False)
            #met = rms / (np.max(df_tr['Y']) - np.min(df_tr['Y']))
            #print(obj_name, "evaluate on train: NRMSE", met)

            y_va = regr.predict(df_va[use_covars])
            rms = mean_squared_error(df_va['Y'], y_va, squared=False)
            #met = rms / (np.max(df_va['Y']) - np.min(df_va['Y']))
            #print(obj_name, "evaluate on test : NRMSE", met)
        elif is_class: #AUC
            y_tr = regr.predict(df_tr[use_covars])
            #met = roc_auc_score(df_tr['Y'], y_tr)
            #print(obj_name, "evaluate on train: AUC", met)

            y_va = regr.predict(df_va[use_covars])
            #met = roc_auc_score(df_va['Y'], y_va)
            #print(obj_name, "evaluate on test : AUC", met)

    return(Yhat, regr)


def extractDigits(lst):
    return list(map(lambda el:[el], lst))

def df_to_dic(data, id_name, x_name, y_name):
    
    data_dic = {'users': [], 'user_data':{}, 'num_samples':[]}

    for i in data[id_name].unique().tolist():
        uname = 'f_{0:05d}'.format(i)  
        subdata = data[data[id_name]==i].reset_index(drop=True)

        data_dic['users'].append(uname)
        data_dic['user_data'][uname] = {'x': subdata[x_name].reset_index(drop=True).values.tolist(), 
                                              'y': extractDigits(subdata[y_name].tolist())}
        data_dic['num_samples'].append(subdata.shape[0])
    return data_dic

        
def save_dic(data, path):
    import json
    with open(path,'w') as outfile:
        json.dump(data, outfile)