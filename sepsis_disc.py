import os
os.chdir('/Users/chenxinlei/RFL-IDR')
import utils_ext as utils
from main import FedAvg
import pandas as pd
import numpy as np
import random
import json
import pickle
import scipy
from numpy.random import seed
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn import linear_model



os.chdir('/Users/chenxinlei/RFL-IDR/ditto')

train_path = "./data/flori/data/train/mytrain.json"
test_path = "./data/flori/data/test/mytest.json"

value_all_list_flori = []
value_vul_list_flori = []
value_all_list_exp = []
value_vul_list_exp = []
value_all_list_naive = []
value_vul_list_naive = []

obj_all_list_flori = []
obj_all_list_exp = []
obj_all_list_naive = []
obj_vul_list_flori = []
obj_vul_list_exp = []
obj_vul_list_naive = []

obj_all_subject_flori = pd.DataFrame()
obj_all_subject_exp = pd.DataFrame()
obj_all_subject_naive = pd.DataFrame()
obj_vul_subject_flori = pd.DataFrame()
obj_vul_subject_exp = pd.DataFrame()
obj_vul_subject_naive = pd.DataFrame()

value_all_hosp_flori = pd.DataFrame()
value_all_hosp_exp = pd.DataFrame()
value_all_hosp_naive = pd.DataFrame()
value_vul_hosp_flori = pd.DataFrame()
value_vul_hosp_exp = pd.DataFrame()
value_vul_hosp_naive = pd.DataFrame()

value_all_subject_flori = pd.DataFrame()
value_all_subject_exp = pd.DataFrame()
value_all_subject_naive = pd.DataFrame()
value_vul_subject_flori = pd.DataFrame()
value_vul_subject_exp = pd.DataFrame()
value_vul_subject_naive = pd.DataFrame()


# set seed
#np.random.seed(12345)
np.random.seed(725)
n_sim = 1



# load and pre-process the data
data = pd.read_csv("./rdata/sepsis_with_outcome_test.csv")
data = data.sort_values("S").reset_index(drop=True)
data['id']=range(data.shape[0])

data.drop(columns=data.columns[0], axis=1, inplace=True)
column_names = list(data.columns.values)

dummies = pd.get_dummies(data['S'], dtype=int).rename(columns=lambda x: 'S'+str(x))
data = pd.concat([data, dummies], axis=1)
dummies_names = dummies.columns.values.tolist()

Y_name = "Y"
A_name = "A"
X_names = column_names[1:16]
X_names_cont = ["age", "weight", "rr", "temp", "gcs", "plt", "bun", "wbc", "gluc", "creat", "lactate", "sofa_total"]
X_names_exp = X_names + dummies_names
S_names = None
XSA_names = X_names_exp + A_name.split(sep=None, maxsplit=-1)
is_rct = False
is_class = True
s_type = "disc"
qua_use = 0.25
hosp = data['S'].unique().tolist()
nhosp = len(hosp)
i_sim = 0



# loop
for i_sim in range(n_sim):
    
    np.random.seed((12345+i_sim))
    
    # split the data into training and testing set
    train_data = data.groupby('S').apply(lambda x: x.sample(frac=0.7))
    test_data = data.loc[list(set(data.index) - set(train_data.index.get_level_values(1)))]

    train_data = train_data.sort_index().reset_index(drop=True)
    test_data = test_data.sort_index().reset_index(drop=True)
    train_data['data'] = 'train'
    test_data['data'] ='test'

    # normalize the whole dataset, the training set, and the testing set
    train_data_norm, test_data_norm = utils.normalize(train_data, test_data, X_names_cont)
    data_norm = pd.concat([train_data_norm, test_data_norm]).sort_values('id').reset_index(drop=True)
    
    Y0mat = data_norm.loc[:, ['train_PROB_A0_S' + str(h) for h in hosp]].to_numpy()
    Y1mat = data_norm.loc[:, ['train_PROB_A1_S' + str(h) for h in hosp]].to_numpy()


    # estimate is_vul
    if s_type == 'disc':
        minMatrix = np.minimum(Y0mat, Y1mat)
        minValue = pd.DataFrame(minMatrix).min(axis=1)

        is_vul = [999]*data_norm.shape[0]
        for i in range(data_norm.shape[0]):
            vul_S = np.where(minMatrix[i,]<=minValue[i])
            vul_hosp = [hosp[j] for j in vul_S[0]]
            if data_norm['S'][i] in vul_hosp:
                is_vul[i] = 1
            else: 
                is_vul[i] = 0

        data_norm["is_vul"] = is_vul

    print(roc_auc_score(data_norm[Y_name], data_norm['Yhat'])) 
    
     
    # get the Y0mat and Y1mat for the testing data -- used in the evaluation of the objective
    Y0mat_test = pd.DataFrame(Y0mat)
    Y0mat_test['data'] = data_norm['data']
    Y0mat_test = Y0mat_test[Y0mat_test['data']=='test']
    Y0mat_test = Y0mat_test.drop(columns = 'data').reset_index(drop=True)
    Y0mat_min = Y0mat_test.min(axis=1)
    
    Y1mat_test = pd.DataFrame(Y1mat)
    Y1mat_test['data'] = data_norm['data']
    Y1mat_test = Y1mat_test[Y1mat_test['data']=='test']
    Y1mat_test = Y1mat_test.drop(columns = 'data').reset_index(drop=True)
    Y1mat_min = Y1mat_test.min(axis=1)


    # get the training and testing set with the new variables added
    train_data_norm = data_norm[data_norm['data']=='train'].reset_index(drop=True).copy()
    test_data_norm = data_norm[data_norm['data']=='test'].reset_index(drop=True).copy()


    test_df = test_data_norm.copy()


    for method in ['flori', 'exp', 'naive']:
        
        
        if method == 'flori':
            
            print('############################### flori ###############################')
            
            g_qua0 = Y0mat.min(axis = 1) # change this to quantile
            g_qua1 = Y1mat.min(axis = 1)
            y_qua = g_qua1-g_qua0

            data_norm["label_flori"] = y_qua

            train_data_norm = data_norm[data_norm['data']=='train'].reset_index(drop=True).copy()
            test_data_norm = data_norm[data_norm['data']=='test'].reset_index(drop=True).copy()

            train_data_dic = utils.df_to_dic(train_data_norm, 'S', X_names, 'label_flori')
            utils.save_dic(train_data_dic, train_path)
            test_data_dic = utils.df_to_dic(test_data_norm, 'S', X_names, 'label_flori')
            utils.save_dic(test_data_dic, test_path)
            
            y_pred = FedAvg('flori', 'reg', num_rounds=500, num_f=len(X_names), 
                            layers=4, units=1024, drop_rate=0.2, 
                            learning_rate=0.5, num_epochs=1, 
                            batch_size=64, class_weight=1)   
            
            test_df['d_pred_flori'] = np.where((y_pred>=0), 1, 0)
            test_df['d_pred_true_flori'] = np.where((test_data_norm['label_flori']>=0), 1, 0)
        


        if method == 'exp':
            
            print('############################### Exp ###############################')
            
            test_data_dic = utils.df_to_dic(data_norm, 'S', X_names, 'Yhat')
            utils.save_dic(test_data_dic, test_path)

            for i in range(len(data_norm[A_name].unique())):
                train = train_data_norm[train_data_norm[A_name]==i].reset_index(drop=True).copy()
                train_data_dic = utils.df_to_dic(train, 'S', X_names, 'Yhat')
                utils.save_dic(train_data_dic, train_path)
                
                data_norm['Y'+str(i)+'_tild'] = FedAvg('flori', 'reg', num_rounds=2000, num_f=len(X_names), 
                                                       layers=1, units=1024, drop_rate=0.2, 
                                                       learning_rate=0.0005, num_epochs=10, 
                                                       batch_size=32, class_weight=1)
                
                
            data_norm['Yhat_tild_exp'] = np.where((data_norm[A_name]==1), data_norm['Y1_tild'], data_norm['Y0_tild'])
            print(mean_squared_error(data_norm['Y0_hat'], data_norm['Y0_tild']))
            print(mean_squared_error(data_norm['Y1_hat'], data_norm['Y1_tild']))
          
            data_norm["label_exp"] = data_norm['Y1_tild']-data_norm['Y0_tild']

            train_data_norm = data_norm[data_norm['data']=='train'].reset_index(drop=True).copy()
            test_data_norm = data_norm[data_norm['data']=='test'].reset_index(drop=True).copy()

            train_data_dic = utils.df_to_dic(train_data_norm, 'S', X_names, 'label_exp')
            utils.save_dic(train_data_dic, train_path)
            test_data_dic = utils.df_to_dic(test_data_norm, 'S', X_names, 'label_exp')
            utils.save_dic(test_data_dic, test_path)
            
            y_pred = FedAvg('flori', 'reg', num_rounds=400, num_f=len(X_names), 
                            layers=3, units=1024, drop_rate=0.2, 
                            learning_rate=0.2, num_epochs=10, 
                            batch_size=64, class_weight=1)

            test_df['d_pred_exp'] = np.where((y_pred>=0), 1, 0)
            test_df['d_pred_true_exp'] = np.where((test_data_norm['label_exp']>=0), 1, 0)


        
        if method == 'naive':
            
            print('############################### Naive ###############################')
            
            test_data_dic = utils.df_to_dic(data_norm, 'S', X_names, Y_name)
            utils.save_dic(test_data_dic, test_path)
            
            for i in range(len(data_norm[A_name].unique())):
                train = train_data_norm[train_data_norm[A_name]==i].reset_index(drop=True).copy()
                train_data_dic = utils.df_to_dic(train, 'S', X_names, Y_name)
                utils.save_dic(train_data_dic, train_path)
                
                data_norm['Y'+str(i)+'_tild'] = FedAvg('flori', 'log', num_rounds=1, num_f=len(X_names), 
                                                       layers=1, units=8, drop_rate=0, 
                                                       learning_rate=0.0001, num_epochs=10, 
                                                       batch_size=32, class_weight=1)
                
            
            data_norm['Yhat_tild_naive'] = np.where((data_norm[A_name]==1), data_norm['Y1_tild'], data_norm['Y0_tild'])
            print(roc_auc_score(data_norm[Y_name], data_norm['Yhat'])) 
                            
            data_norm["label_naive"] = data_norm['Y1_tild']-data_norm['Y0_tild']

            train_data_norm = data_norm[data_norm['data']=='train'].reset_index(drop=True).copy()
            test_data_norm = data_norm[data_norm['data']=='test'].reset_index(drop=True).copy()
            
            train_data_dic = utils.df_to_dic(train_data_norm, 'S', X_names, 'label_naive')
            utils.save_dic(train_data_dic, train_path)
            test_data_dic = utils.df_to_dic(test_data_norm, 'S', X_names, 'label_naive')
            utils.save_dic(test_data_dic, test_path)
            
            y_pred = FedAvg('flori', 'reg', num_rounds=700, num_f=len(X_names), 
                            layers=3, units=1024, drop_rate=0.2, 
                            learning_rate=0.01, num_epochs=10, 
                            batch_size=64, class_weight=1)
            
            test_df['d_pred_naive'] = np.where((y_pred>=0), 1, 0)
            test_df['d_pred_true_naive'] = np.where((test_data_norm['label_naive']>=0), 1, 0)



        # evaluation
        A_pred = np.where((y_pred>=0), 1, 0)
        test_df['A_pred'] = A_pred.astype(int)

        test_df['Yhat_pred'] = np.where((test_df['A_pred']==1), test_df['Y1_hat'], test_df['Y0_hat'])
        value_all = np.mean(test_df['Yhat_pred'])
        test_df_vul = test_df[test_df['is_vul']==1]
        value_vul = np.mean(test_df_vul['Yhat_pred'])
        
        value_all_by_hosp = test_df.groupby('S').apply(lambda x: np.mean(x['Yhat_pred']))
        value_all_by_subject = test_df['Yhat_pred']
        value_vul_by_hosp = test_df_vul.groupby('S').apply(lambda x: np.mean(x['Yhat_pred']))
        value_vul_by_subject = test_df_vul['Yhat_pred']
            
        
        obj_est = [999]*(test_df.shape[0])
        for i in range(test_df.shape[0]):
            if A_pred[i] == 0:
                obj_est[i] = Y0mat_min[i]
            else:
                obj_est[i] = Y1mat_min[i]
                
        obj_vul_est = pd.DataFrame()
        obj_vul_est['obj'] = obj_est
        obj_vul_est['is_vul'] = np.array(test_df['is_vul'])
        obj_vul_est = obj_vul_est[obj_vul_est['is_vul']==1]
        obj_vul_est = obj_vul_est['obj']

        if method == 'flori':
            
            value_all_list_flori.append(value_all)
            value_vul_list_flori.append(value_vul)
            obj_all_list_flori.append(np.mean(obj_est))
            obj_vul_list_flori.append(np.mean(obj_vul_est))
            
            value_all_hosp_flori = pd.concat([value_all_hosp_flori, value_all_by_hosp], ignore_index=True)
            value_all_subject_flori = pd.concat([value_all_subject_flori, value_all_by_subject], ignore_index=True)
            obj_all_subject_flori = pd.concat([obj_all_subject_flori, pd.Series(obj_est)], ignore_index=True)
            
            value_vul_hosp_flori = pd.concat([value_vul_hosp_flori, value_vul_by_hosp], ignore_index=True)
            value_vul_subject_flori = pd.concat([value_vul_subject_flori, value_vul_by_subject], ignore_index=True)
            obj_vul_subject_flori = pd.concat([obj_vul_subject_flori, obj_vul_est], ignore_index=True)
            
            
        elif method == 'exp':
            
            value_all_list_exp.append(value_all)
            value_vul_list_exp.append(value_vul)
            obj_all_list_exp.append(np.mean(obj_est))
            obj_vul_list_exp.append(np.mean(obj_vul_est))
            
            value_all_hosp_exp = pd.concat([value_all_hosp_exp, value_all_by_hosp], ignore_index=True)
            value_all_subject_exp = pd.concat([value_all_subject_exp, value_all_by_subject], ignore_index=True)
            obj_all_subject_exp = pd.concat([obj_all_subject_exp, pd.Series(obj_est)], ignore_index=True)
            
            value_vul_hosp_exp = pd.concat([value_vul_hosp_exp, value_vul_by_hosp], ignore_index=True)
            value_vul_subject_exp = pd.concat([value_vul_subject_exp, value_vul_by_subject], ignore_index=True)
            obj_vul_subject_exp = pd.concat([obj_vul_subject_exp, obj_vul_est], ignore_index=True)
            
        else:
            
            value_all_list_naive.append(value_all)
            value_vul_list_naive.append(value_vul)
            obj_all_list_naive.append(np.mean(obj_est))
            obj_vul_list_naive.append(np.mean(obj_vul_est))
            
            value_all_hosp_naive = pd.concat([value_all_hosp_naive, value_all_by_hosp], ignore_index=True)
            value_all_subject_naive = pd.concat([value_all_subject_naive, value_all_by_subject], ignore_index=True)
            obj_all_subject_naive = pd.concat([obj_all_subject_naive, pd.Series(obj_est)], ignore_index=True)
            
            value_vul_hosp_naive = pd.concat([value_vul_hosp_naive, value_vul_by_hosp], ignore_index=True)
            value_vul_subject_naive = pd.concat([value_vul_subject_naive, value_vul_by_subject], ignore_index=True)
            obj_vul_subject_naive = pd.concat([obj_vul_subject_naive, obj_vul_est], ignore_index=True)

        print('Value (all) from ' + str(method) + ': ' + str(value_all))
        print('Value (vulnerable) from ' + str(method) + ': ' + str(value_vul))

"""

value_df = pd.DataFrame()
value_df['method'] = ['flori', 'exp', 'naive']
value_df['objective_all'] = [np.mean(np.array(obj_all_list_flori)), np.mean(np.array(obj_all_list_exp)), np.mean(np.array(obj_all_list_naive))]
value_df['objective_all_std'] = [np.std(np.array(obj_all_list_flori)), np.std(np.array(obj_all_list_exp)), np.std(np.array(obj_all_list_naive))]
value_df['objective_vul'] = [np.mean(np.array(obj_vul_list_flori)), np.mean(np.array(obj_vul_list_exp)), np.mean(np.array(obj_vul_list_naive))]
value_df['objective_vul_std'] = [np.std(np.array(obj_vul_list_flori)), np.std(np.array(obj_vul_list_exp)), np.std(np.array(obj_vul_list_naive))]
value_df['value_all'] = [np.mean(np.array(value_all_list_flori)), np.mean(np.array(value_all_list_exp)), np.mean(np.array(value_all_list_naive))]
value_df['value_all_std'] = [np.std(np.array(value_all_list_flori)), np.std(np.array(value_all_list_exp)), np.std(np.array(value_all_list_naive))]
value_df['value_vul'] = [np.mean(np.array(value_vul_list_flori)), np.mean(np.array(value_vul_list_exp)), np.mean(np.array(value_vul_list_naive))]
value_df['value_vul_std'] = [np.std(np.array(value_vul_list_flori)), np.std(np.array(value_vul_list_exp)), np.std(np.array(value_vul_list_naive))]

value_df['Objective (all)'] = round(value_df['objective_all'],2).astype(str) + ' (' + round(value_df['objective_all_std'],2).astype(str) + ')'
value_df['Objective (vulnerable)'] = round(value_df['objective_vul'],2).astype(str) + ' (' + round(value_df['objective_vul_std'],2).astype(str) + ')'
value_df['Value (all)'] = round(value_df['value_all'],2).astype(str) + ' (' + round(value_df['value_all_std'],2).astype(str) + ')'
value_df['Value (vulnerable)'] = round(value_df['value_vul'],2).astype(str) + ' (' + round(value_df['value_vul_std'],2).astype(str) + ')'
    
value_df.to_csv('/Users/chenxinlei/RFL-IDR/result/Sepsis results/value_df_sepsis.csv', index=False)  
data_norm.to_csv('/Users/chenxinlei/RFL-IDR/result/Sepsis results/data_norm.csv', index=False)  


value_all_hosp_flori.to_csv('/Users/chenxinlei/RFL-IDR/result/Sepsis results/value_df_sepsis_flori_hosp.csv', index=False)
value_all_hosp_exp.to_csv('/Users/chenxinlei/RFL-IDR/result/Sepsis results/value_df_sepsis_exp_hosp.csv', index=False)
value_all_hosp_naive.to_csv('/Users/chenxinlei/RFL-IDR/result/Sepsis results/value_df_sepsis_naive_hosp.csv', index=False)

value_all_subject_flori.to_csv('/Users/chenxinlei/RFL-IDR/result/Sepsis results/value_df_sepsis_flori_sub.csv', index=False)
value_all_subject_exp.to_csv('/Users/chenxinlei/RFL-IDR/result/Sepsis results/value_df_sepsis_exp_sub.csv', index=False)
value_all_subject_naive.to_csv('/Users/chenxinlei/RFL-IDR/result/Sepsis results/value_df_sepsis_naive_sub.csv', index=False)

obj_all_subject_flori.to_csv('/Users/chenxinlei/RFL-IDR/result/Sepsis results/obj_df_sepsis_flori_sub.csv', index=False)
obj_all_subject_exp.to_csv('/Users/chenxinlei/RFL-IDR/result/Sepsis results/obj_df_sepsis_exp_sub.csv', index=False)
obj_all_subject_naive.to_csv('/Users/chenxinlei/RFL-IDR/result/Sepsis results/obj_df_sepsis_naive_sub.csv', index=False)

value_vul_hosp_flori.to_csv('/Users/chenxinlei/RFL-IDR/result/Sepsis results/value_df_sepsis_flori_hosp_vul.csv', index=False)
value_vul_hosp_exp.to_csv('/Users/chenxinlei/RFL-IDR/result/Sepsis results/value_df_sepsis_exp_hosp_vul.csv', index=False)
value_vul_hosp_naive.to_csv('/Users/chenxinlei/RFL-IDR/result/Sepsis results/value_df_sepsis_naive_hosp_vul.csv', index=False)

value_vul_subject_flori.to_csv('/Users/chenxinlei/RFL-IDR/result/Sepsis results/value_df_sepsis_flori_sub_vul.csv', index=False)
value_vul_subject_exp.to_csv('/Users/chenxinlei/RFL-IDR/result/Sepsis results/value_df_sepsis_exp_sub_vul.csv', index=False)
value_vul_subject_naive.to_csv('/Users/chenxinlei/RFL-IDR/result/Sepsis results/value_df_sepsis_naive_sub_vul.csv', index=False)

obj_vul_subject_flori.to_csv('/Users/chenxinlei/RFL-IDR/result/Sepsis results/obj_df_sepsis_flori_sub_vul.csv', index=False)
obj_vul_subject_exp.to_csv('/Users/chenxinlei/RFL-IDR/result/Sepsis results/obj_df_sepsis_exp_sub_vul.csv', index=False)
obj_vul_subject_naive.to_csv('/Users/chenxinlei/RFL-IDR/result/Sepsis results/obj_df_sepsis_naive_sub_vul.csv', index=False)

"""
