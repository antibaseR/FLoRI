import os
os.chdir('/Users/chenxinlei/RFL-IDR')
from genData import genData
import utils_ext as utils
from main import FedAvg
import pandas as pd
import numpy as np
import random
import json
import pickle
from numpy.random import seed
import gc

os.chdir('/Users/chenxinlei/RFL-IDR/ditto')

train_path = "./data/flori/data/train/mytrain.json"
test_path = "./data/flori/data/test/mytest.json"

num_rounds = 50

# settings
#np.random.seed(12345)
nhosp = 10
hosp = range(nhosp)
hosp_size = 2000
n_sim = 100


gc.collect()
gc.collect()
gc.collect()


for sim_setting in [7]:
    
    gc.collect()
    gc.collect()
    gc.collect()
    
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
    
    class_opt_all_list_flori = []
    class_opt_vul_list_flori = []
    class_opt_all_list_exp = []
    class_opt_vul_list_exp = []
    class_opt_all_list_naive = []
    class_opt_vul_list_naive = []
    
    class_flori_all_list_flori = []
    class_flori_vul_list_flori = []
    class_flori_all_list_exp = []
    class_flori_vul_list_exp = []
    class_flori_all_list_naive = []
    class_flori_vul_list_naive = []
    
    obj_all_hosp_flori = pd.DataFrame()
    obj_all_hosp_exp = pd.DataFrame()
    obj_all_hosp_naive = pd.DataFrame()
    obj_vul_hosp_flori = pd.DataFrame()
    obj_vul_hosp_exp = pd.DataFrame()
    obj_vul_hosp_naive = pd.DataFrame()

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
    
    class_opt_all_hosp_flori = pd.DataFrame()
    class_opt_all_hosp_exp = pd.DataFrame()
    class_opt_all_hosp_naive = pd.DataFrame()
    class_opt_vul_hosp_flori = pd.DataFrame()
    class_opt_vul_hosp_exp = pd.DataFrame()
    class_opt_vul_hosp_naive = pd.DataFrame()
    
    class_flori_all_hosp_flori = pd.DataFrame()
    class_flori_all_hosp_exp = pd.DataFrame()
    class_flori_all_hosp_naive = pd.DataFrame()
    class_flori_vul_hosp_flori = pd.DataFrame()
    class_flori_vul_hosp_exp = pd.DataFrame()
    class_flori_vul_hosp_naive = pd.DataFrame()

    class_opt_all_subject_flori = pd.DataFrame()
    class_opt_all_subject_exp = pd.DataFrame()
    class_opt_all_subject_naive = pd.DataFrame()
    class_opt_vul_subject_flori = pd.DataFrame()
    class_opt_vul_subject_exp = pd.DataFrame()
    class_opt_vul_subject_naive = pd.DataFrame()

    class_flori_all_subject_flori = pd.DataFrame()
    class_flori_all_subject_exp = pd.DataFrame()
    class_flori_all_subject_naive = pd.DataFrame()
    class_flori_vul_subject_flori = pd.DataFrame()
    class_flori_vul_subject_exp = pd.DataFrame()
    class_flori_vul_subject_naive = pd.DataFrame()
    
    
    # loop
    for i_sim in range(n_sim):
        
        gc.collect()
        gc.collect()
        gc.collect()
        
        np.random.seed((12345+i_sim))
        
        # generate data, and split into train and test set
        Y_name, A_name, X_names, S_names, X_names_exp, is_rct, is_class, s_type, qua_use, data = genData(which_sim = sim_setting, nhosp=nhosp, hosp_size=hosp_size)
        XSA_names = X_names_exp + A_name.split(sep=None, maxsplit=-1)
        XA_names = X_names + A_name.split(sep=None, maxsplit=-1)
        
        train_data = data.groupby('S').apply(lambda x: x.sample(frac=0.8))
        test_data = data.loc[list(set(data.index) - set(train_data.index.get_level_values(1)))]

        train_data = train_data.sort_index().reset_index(drop=True)
        test_data = test_data.sort_index().reset_index(drop=True)

        train_data['data'] = 'train'
        test_data['data'] ='test'



        # normalize the whole dataset, the training set, and the testing set
        train_data_norm, test_data_norm = utils.normalize(train_data, test_data, X_names)
        #train_data_norm, data_norm = utils.normalize(train_data, data, X_names)
        data_norm = pd.concat([train_data_norm, test_data_norm]).sort_values('id').reset_index(drop=True)



        # Estimate the potential outcome
        Ymat=pd.DataFrame()
        
        train = train_data_norm.copy()
        train_data_dic = utils.df_to_dic(train, 'S', XSA_names, Y_name)
        utils.save_dic(train_data_dic, train_path)

        test = pd.DataFrame()
        test_new = data_norm.copy()
        test_new = test_new.drop(test_new.columns[test_new.columns.get_loc('S0'):(nhosp+test_new.columns.get_loc('S0'))], axis=1)
        for j in hosp:
            test_new['S']=j
            for i in range(len(data_norm[A_name].unique())):
                test_new[A_name]=i
                test = pd.concat([test, test_new])

        dummies_test = pd.get_dummies(test['S'], dtype=int).rename(columns=lambda x: 'S'+str(x))
        test = pd.concat([test, dummies_test], axis=1)
        test['id_temp'] = range(test.shape[0])
        test = test.reset_index(drop=True)
        test_data_dic = utils.df_to_dic(test, 'S', XSA_names, Y_name)
        utils.save_dic(test_data_dic, test_path)

        Ymat = FedAvg(data='flori', model='reg', num_rounds=500, num_f=len(XSA_names), 
                      layers=2, units=1024, drop_rate=0.2, 
                      learning_rate=0.001, num_epochs=1, 
                      batch_size=32, class_weight=1)
                
        ranges = []
        for start in range(0, len(Ymat), nhosp*hosp_size*2):
            end = start + (nhosp*hosp_size-1)
            ranges.append(list(range(start, end + 1)))
        # Flatten the list if you want a single list of all values
        y0_sequence = [item for sublist in ranges for item in sublist]
        
        ranges = []
        for start in range((nhosp*hosp_size), len(Ymat), nhosp*hosp_size*2):
            end = start + (nhosp*hosp_size-1)
            ranges.append(list(range(start, end + 1)))
        # Flatten the list if you want a single list of all values
        y1_sequence = [item for sublist in ranges for item in sublist]


        
        #index_half = int(test.shape[0]/2)
        Y0mat = np.array(Ymat[y0_sequence]).reshape(nhosp, data_norm.shape[0]).T
        Y1mat = np.array(Ymat[y1_sequence]).reshape(nhosp, data_norm.shape[0]).T
        
        
        """
        # estimate is_vul
        minMatrix = np.minimum(Y0mat, Y1mat)
        minValue = pd.DataFrame(minMatrix).quantile(q=0.5, axis=1)
        #minValue = 0.5

        is_vul = [999]*data_norm.shape[0]
        for i in range(data_norm.shape[0]):
            vul_S = np.where(minMatrix[i,]<=minValue[i])
            vul_hosp = [hosp[j] for j in vul_S[0]]
            if data_norm['S'][i] in vul_hosp:
                is_vul[i] = 1
            else: 
                is_vul[i] = 0

        data_norm["is_vul"] = is_vul
        #see = data_norm[data_norm['is_vul']==1]
        """



        # estimated E(Y|X,S,A) under the actual hospital
        Y0_hat = []
        Y1_hat = []

        for i in range(data_norm.shape[0]):
            true_hosp = data_norm['S'][i]
            Y0_hat_new = Y0mat[i, true_hosp]
            Y1_hat_new = Y1mat[i, true_hosp]
            Y0_hat.append(Y0_hat_new)
            Y1_hat.append(Y1_hat_new)
                
        data_norm['Y0_hat'] = Y0_hat
        data_norm['Y1_hat'] = Y1_hat
        data_norm['Yhat'] = np.where((data_norm[A_name]==1), data_norm['Y1_hat'], data_norm['Y0_hat'])
        print('MSE is '+str(np.mean((data_norm['Yhat']-data_norm['Y'])**2)))
        
        
        
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



        # Split the dataset again
        train_data_norm = data_norm[data_norm['data']=='train'].copy()
        test_data_norm = data_norm[data_norm['data']=='test'].copy()



        # Save the dataset
        data_norm.to_json('./data/flori/data/df.json')
        train_data_norm.to_json('./data/flori/data/traindf.json')
        test_data_norm.to_json('./data/flori/data/testdf.json')



        # Read the dataset
        data_norm = pd.read_json('./data/flori/data/df.json')
        train_data_norm = pd.read_json('./data/flori/data/traindf.json')
        test_data_norm = pd.read_json('./data/flori/data/testdf.json')

        data_norm.to_csv('/Users/chenxinlei/RFL-IDR/data_for_r/data_norm.csv', index=False)  



        for method in ['flori', 'exp']:
            
            gc.collect()
            gc.collect()
            gc.collect()
            
            
            if method == 'flori':
                
                g_qua0 = Y0mat.min(axis = 1)
                g_qua1 = Y1mat.min(axis = 1)
                y_qua = g_qua1-g_qua0

                data_norm["label_flori"] = y_qua

                train_data_norm = data_norm[data_norm['data']=='train'].reset_index(drop=True).copy()
                test_data_norm = data_norm[data_norm['data']=='test'].reset_index(drop=True).copy()

                train_data_dic = utils.df_to_dic(train_data_norm, 'S', X_names, 'label_flori')
                utils.save_dic(train_data_dic, train_path)
                test_data_dic = utils.df_to_dic(test_data_norm, 'S', X_names, 'label_flori')
                utils.save_dic(test_data_dic, test_path)
            
                y_pred = FedAvg('flori', 'reg', num_rounds, num_f=len(X_names),
                                layers=2, units=1024, drop_rate=0.2, 
                                learning_rate=0.01, num_epochs=1, batch_size=32, class_weight=1)
                
                #y_pred = test_data_norm['label_flori']

            
            
            
            if method == 'exp':
                
                """
                train = train_data_norm.copy()
                train_data_dic = utils.df_to_dic(train, 'S', XA_names, 'Yhat')
                utils.save_dic(train_data_dic, train_path)
                
                test = pd.DataFrame()
                test_new = data_norm.copy()
                for i in range(len(data_norm[A_name].unique())):
                    test_new[A_name]=i
                    test = pd.concat([test, test_new])
                test_data_dic = utils.df_to_dic(test, 'S', XA_names, 'Yhat')
                utils.save_dic(test_data_dic, test_path)
                
                Ytild = FedAvg('flori', 'reg', num_rounds=500, num_f=len(XA_names), 
                               layers=1, units=1024, drop_rate=0.2, 
                               learning_rate=0.001, num_epochs=1, 
                               batch_size=32, class_weight=1)
                
                index_half = int(test.shape[0]/2)
                data_norm['Y0_tild'] = Ytild[0:index_half]
                data_norm['Y1_tild'] = Ytild[index_half:]
                """
                
                test_data_dic = utils.df_to_dic(data_norm, 'S', X_names, 'Yhat')
                utils.save_dic(test_data_dic, test_path)

                for i in range(len(data_norm[A_name].unique())):
                    train = train_data_norm[train_data_norm[A_name]==i].reset_index(drop=True).copy()
                    train_data_dic = utils.df_to_dic(train, 'S', X_names, 'Yhat')
                    utils.save_dic(train_data_dic, train_path)
                    data_norm['Y'+str(i)+'_tild'] =  FedAvg('flori', 'reg', num_rounds=500, num_f=len(X_names), 
                                                            layers=1, units=1024, drop_rate=0.2, 
                                                            learning_rate=0.001, num_epochs=1, 
                                                            batch_size=32, class_weight=1)
                
                data_norm["label_exp"] = data_norm['Y1_tild']-data_norm['Y0_tild']

                train_data_norm = data_norm[data_norm['data']=='train'].reset_index(drop=True).copy()
                test_data_norm = data_norm[data_norm['data']=='test'].reset_index(drop=True).copy()

                train_data_dic = utils.df_to_dic(train_data_norm, 'S', X_names, 'label_exp')
                utils.save_dic(train_data_dic, train_path)
                test_data_dic = utils.df_to_dic(test_data_norm, 'S', X_names, 'label_exp')
                utils.save_dic(test_data_dic, test_path)
                
                y_pred = FedAvg('flori', 'reg', num_rounds, num_f=len(X_names), 
                                layers=2, units=1024, drop_rate=0.2, 
                                learning_rate=0.01, num_epochs=1, batch_size=32, class_weight=1)
                
                #y_pred = test_data_norm['label_exp']
                
            
            """
            if method == 'naive':
                
                train = train_data_norm.copy()
                train_data_dic = utils.df_to_dic(train, 'S', XA_names, Y_name)
                utils.save_dic(train_data_dic, train_path)
                
                test = pd.DataFrame()
                test_new = data_norm.copy()
                for i in range(len(data_norm[A_name].unique())):
                    test_new[A_name]=i
                    test = pd.concat([test, test_new])
                test_data_dic = utils.df_to_dic(test, 'S', XA_names, Y_name)
                utils.save_dic(test_data_dic, test_path)
                
                Ytild = FedAvg('flori', 'reg', num_rounds=500, num_f=len(XA_names), 
                               layers=1, units=1024, drop_rate=0.2, 
                               learning_rate=0.001, num_epochs=1, 
                               batch_size=32, class_weight=1)
                
                index_half = int(test.shape[0]/2)
                data_norm['Y0_tild'] = Ytild[0:index_half]
                data_norm['Y1_tild'] = Ytild[index_half:]

                data_norm["label_naive"] = data_norm['Y1_tild']-data_norm['Y0_tild']

                train_data_norm = data_norm[data_norm['data']=='train'].reset_index(drop=True).copy()
                test_data_norm = data_norm[data_norm['data']=='test'].reset_index(drop=True).copy()

                train_data_dic = utils.df_to_dic(train_data_norm, 'S', X_names, 'label_naive')
                utils.save_dic(train_data_dic, train_path)
                test_data_dic = utils.df_to_dic(test_data_norm, 'S', X_names, 'label_naive')
                utils.save_dic(test_data_dic, test_path)
                
                #y_pred = FedAvg('flori', 'reg', num_rounds, num_f=len(X_names), 
                #                layers=2, units=1024, drop_rate=0.2, 
                #                learning_rate=0.01, num_epochs=1, batch_size=32, class_weight=1)
                
                y_pred = test_data_norm['label_naive']
            """


                
            # evaluation
            test_df = pd.read_json('./data/flori/data/testdf.json')
            A_pred = np.where((y_pred>=0), 1, 0)
            test_df['A_pred'] = A_pred.astype(int)

            ## reward
            test_df['Yhat_true'] = np.where((test_df['A_pred']==1), test_df['Y1_true'], test_df['Y0_true'])
            value_all = np.mean(test_df['Yhat_true'])
            test_df_vul = test_df[test_df['is_vul']==1]
            value_vul = np.mean(test_df_vul['Yhat_true'])
            
            value_all_by_hosp = test_df.groupby('S').apply(lambda x: np.mean(x['Yhat_true']))
            value_all_by_subject = test_df['Yhat_true']
            value_vul_by_hosp = test_df_vul.groupby('S').apply(lambda x: np.mean(x['Yhat_true']))
            value_vul_by_subject = test_df_vul['Yhat_true']
                
            ## objective
            obj_est = [999]*(test_df.shape[0])
            for i in range(test_df.shape[0]):
                if A_pred[i] == 0:
                    obj_est[i] = Y0mat_min[i]
                else:
                    obj_est[i] = Y1mat_min[i]
            test_df['obj_est'] = obj_est
            obj_all = np.mean(obj_est)
            
            test_df_vul = test_df[test_df['is_vul']==1]
            obj_vul = np.mean(test_df_vul['obj_est'])
                       
            obj_all_by_hosp = test_df.groupby('S').apply(lambda x: np.mean(x['obj_est']))
            obj_all_by_subject = test_df['obj_est']
            obj_vul_by_hosp = test_df_vul.groupby('S').apply(lambda x: np.mean(x['obj_est']))
            obj_vul_by_subject = test_df_vul['obj_est']
            
            ## classification rate
            ### Opt
            test_df['class_opt'] = np.where((test_df['A_pred'] == test_df['d_opt']), 1, 0)
            class_opt_all = np.mean(test_df['class_opt'])
            test_df_vul = test_df[test_df['is_vul']==1]
            class_opt_vul = np.mean(test_df_vul['class_opt'])

            class_opt_all_by_hosp = test_df.groupby('S').apply(lambda x: np.mean(x['class_opt']))
            class_opt_all_by_subject = test_df['class_opt']
            class_opt_vul_by_hosp = test_df_vul.groupby('S').apply(lambda x: np.mean(x['class_opt']))
            class_opt_vul_by_subject = test_df_vul['class_opt']
            
            ### Flori
            test_df['class_flori'] = np.where((test_df['A_pred'] == test_df['d']), 1, 0)
            class_flori_all = np.mean(test_df['class_flori'])
            test_df_vul = test_df[test_df['is_vul']==1]
            class_flori_vul = np.mean(test_df_vul['class_flori'])

            class_flori_all_by_hosp = test_df.groupby('S').apply(lambda x: np.mean(x['class_flori']))
            class_flori_all_by_subject = test_df['class_flori']
            class_flori_vul_by_hosp = test_df_vul.groupby('S').apply(lambda x: np.mean(x['class_flori']))
            class_flori_vul_by_subject = test_df_vul['class_flori']
            

            if method == 'flori':
                
                value_all_list_flori.append(value_all)
                value_vul_list_flori.append(value_vul)
                obj_all_list_flori.append(obj_all)
                obj_vul_list_flori.append(obj_vul)
                class_opt_all_list_flori.append(class_opt_all)
                class_opt_vul_list_flori.append(class_opt_vul)
                class_flori_all_list_flori.append(class_flori_all)
                class_flori_vul_list_flori.append(class_flori_vul)
                
                value_all_hosp_flori = pd.concat([value_all_hosp_flori, value_all_by_hosp], ignore_index=True)
                value_all_subject_flori = pd.concat([value_all_subject_flori, value_all_by_subject], ignore_index=True)
                class_opt_all_hosp_flori = pd.concat([class_opt_all_hosp_flori, class_opt_all_by_hosp], ignore_index=True)
                class_opt_all_subject_flori = pd.concat([class_opt_all_subject_flori, class_opt_all_by_subject], ignore_index=True)
                class_flori_all_hosp_flori = pd.concat([class_flori_all_hosp_flori, class_flori_all_by_hosp], ignore_index=True)
                class_flori_all_subject_flori = pd.concat([class_flori_all_subject_flori, class_flori_all_by_subject], ignore_index=True)
                obj_all_hosp_flori = pd.concat([obj_all_hosp_flori, obj_all_by_hosp], ignore_index=True)
                obj_all_subject_flori = pd.concat([obj_all_subject_flori, obj_all_by_subject], ignore_index=True)
                
                value_vul_hosp_flori = pd.concat([value_vul_hosp_flori, value_vul_by_hosp], ignore_index=True)
                value_vul_subject_flori = pd.concat([value_vul_subject_flori, value_vul_by_subject], ignore_index=True)
                class_opt_vul_hosp_flori = pd.concat([class_opt_vul_hosp_flori, class_opt_vul_by_hosp], ignore_index=True)
                class_opt_vul_subject_flori = pd.concat([class_opt_vul_subject_flori, class_opt_vul_by_subject], ignore_index=True)
                class_flori_vul_hosp_flori = pd.concat([class_flori_vul_hosp_flori, class_flori_vul_by_hosp], ignore_index=True)
                class_flori_vul_subject_flori = pd.concat([class_flori_vul_subject_flori, class_flori_vul_by_subject], ignore_index=True)
                obj_vul_hosp_flori = pd.concat([obj_vul_hosp_flori, obj_vul_by_hosp], ignore_index=True)
                obj_vul_subject_flori = pd.concat([obj_vul_subject_flori, obj_vul_by_subject], ignore_index=True)
                
                
            elif method == 'exp':
                
                value_all_list_exp.append(value_all)
                value_vul_list_exp.append(value_vul)
                obj_all_list_exp.append(obj_all)
                obj_vul_list_exp.append(obj_vul)
                class_opt_all_list_exp.append(class_opt_all)
                class_opt_vul_list_exp.append(class_opt_vul)
                class_flori_all_list_exp.append(class_flori_all)
                class_flori_vul_list_exp.append(class_flori_vul)

                
                value_all_hosp_exp = pd.concat([value_all_hosp_exp, value_all_by_hosp], ignore_index=True)
                value_all_subject_exp = pd.concat([value_all_subject_exp, value_all_by_subject], ignore_index=True)
                class_opt_all_hosp_exp = pd.concat([class_opt_all_hosp_exp, class_opt_all_by_hosp], ignore_index=True)
                class_opt_all_subject_exp = pd.concat([class_opt_all_subject_exp, class_opt_all_by_subject], ignore_index=True)
                class_flori_all_hosp_exp = pd.concat([class_flori_all_hosp_exp, class_flori_all_by_hosp], ignore_index=True)
                class_flori_all_subject_exp = pd.concat([class_flori_all_subject_exp, class_flori_all_by_subject], ignore_index=True)
                obj_all_hosp_exp = pd.concat([obj_all_hosp_exp, obj_all_by_hosp], ignore_index=True)
                obj_all_subject_exp = pd.concat([obj_all_subject_exp, obj_all_by_subject], ignore_index=True)
                
                value_vul_hosp_exp = pd.concat([value_vul_hosp_exp, value_vul_by_hosp], ignore_index=True)
                value_vul_subject_exp = pd.concat([value_vul_subject_exp, value_vul_by_subject], ignore_index=True)
                class_opt_vul_hosp_exp = pd.concat([class_opt_vul_hosp_exp, class_opt_vul_by_hosp], ignore_index=True)
                class_opt_vul_subject_exp = pd.concat([class_opt_vul_subject_exp, class_opt_vul_by_subject], ignore_index=True)
                class_flori_vul_hosp_exp = pd.concat([class_flori_vul_hosp_exp, class_flori_vul_by_hosp], ignore_index=True)
                class_flori_vul_subject_exp = pd.concat([class_flori_vul_subject_exp, class_flori_vul_by_subject], ignore_index=True)
                obj_vul_hosp_exp = pd.concat([obj_vul_hosp_exp, obj_vul_by_hosp], ignore_index=True)
                obj_vul_subject_exp = pd.concat([obj_vul_subject_exp, obj_vul_by_subject], ignore_index=True)
                
            #else:
                
                value_all_list_naive.append(value_all)
                value_vul_list_naive.append(value_vul)
                obj_all_list_naive.append(obj_all)
                obj_vul_list_naive.append(obj_vul)
                class_opt_all_list_naive.append(class_opt_all)
                class_opt_vul_list_naive.append(class_opt_vul)
                class_flori_all_list_naive.append(class_flori_all)
                class_flori_vul_list_naive.append(class_flori_vul)
                
                value_all_hosp_naive = pd.concat([value_all_hosp_naive, value_all_by_hosp], ignore_index=True)
                value_all_subject_naive = pd.concat([value_all_subject_naive, value_all_by_subject], ignore_index=True)
                class_opt_all_hosp_naive = pd.concat([class_opt_all_hosp_naive, class_opt_all_by_hosp], ignore_index=True)
                class_opt_all_subject_naive = pd.concat([class_opt_all_subject_naive, class_opt_all_by_subject], ignore_index=True)
                class_flori_all_hosp_naive = pd.concat([class_flori_all_hosp_naive, class_flori_all_by_hosp], ignore_index=True)
                class_flori_all_subject_naive = pd.concat([class_flori_all_subject_naive, class_flori_all_by_subject], ignore_index=True)
                obj_all_hosp_naive = pd.concat([obj_all_hosp_naive, obj_all_by_hosp], ignore_index=True)
                obj_all_subject_naive = pd.concat([obj_all_subject_naive, obj_all_by_subject], ignore_index=True)
                
                value_vul_hosp_naive = pd.concat([value_vul_hosp_naive, value_vul_by_hosp], ignore_index=True)
                value_vul_subject_naive = pd.concat([value_vul_subject_naive, value_vul_by_subject], ignore_index=True)
                class_opt_vul_hosp_naive = pd.concat([class_opt_vul_hosp_naive, class_opt_vul_by_hosp], ignore_index=True)
                class_opt_vul_subject_naive = pd.concat([class_opt_vul_subject_naive, class_opt_vul_by_subject], ignore_index=True)
                class_flori_vul_hosp_naive = pd.concat([class_flori_vul_hosp_naive, class_flori_vul_by_hosp], ignore_index=True)
                class_flori_vul_subject_naive = pd.concat([class_flori_vul_subject_naive, class_flori_vul_by_subject], ignore_index=True)
                obj_vul_hosp_naive = pd.concat([obj_vul_hosp_naive, obj_vul_by_hosp], ignore_index=True)
                obj_vul_subject_naive = pd.concat([obj_vul_subject_naive, obj_vul_by_subject], ignore_index=True)

            print('Value (all) from ' + str(method) + ': ' + str(value_all))
            print('Value (vulnerable) from ' + str(method) + ': ' + str(value_vul))


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
    
    value_df['class_opt_all'] = [np.mean(np.array(class_opt_all_list_flori)), np.mean(np.array(class_opt_all_list_exp)), np.mean(np.array(class_opt_all_list_naive))]
    value_df['class_opt_all_std'] = [np.std(np.array(class_opt_all_list_flori)), np.std(np.array(class_opt_all_list_exp)), np.std(np.array(class_opt_all_list_naive))]
    value_df['class_opt_vul'] = [np.mean(np.array(class_opt_vul_list_flori)), np.mean(np.array(class_opt_vul_list_exp)), np.mean(np.array(class_opt_vul_list_naive))]
    value_df['class_opt_vul_std'] = [np.std(np.array(class_opt_vul_list_flori)), np.std(np.array(class_opt_vul_list_exp)), np.std(np.array(class_opt_vul_list_naive))]
    
    value_df['class_flori_all'] = [np.mean(np.array(class_flori_all_list_flori)), np.mean(np.array(class_flori_all_list_exp)), np.mean(np.array(class_flori_all_list_naive))]
    value_df['class_flori_all_std'] = [np.std(np.array(class_flori_all_list_flori)), np.std(np.array(class_flori_all_list_exp)), np.std(np.array(class_flori_all_list_naive))]
    value_df['class_flori_vul'] = [np.mean(np.array(class_flori_vul_list_flori)), np.mean(np.array(class_flori_vul_list_exp)), np.mean(np.array(class_flori_vul_list_naive))]
    value_df['class_flori_vul_std'] = [np.std(np.array(class_flori_vul_list_flori)), np.std(np.array(class_flori_vul_list_exp)), np.std(np.array(class_flori_vul_list_naive))]
    
    
    value_df['Objective (all)'] = round(value_df['objective_all'],2).astype(str) + ' (' + round(value_df['objective_all_std'],2).astype(str) + ')'
    value_df['Objective (vulnerable)'] = round(value_df['objective_vul'],2).astype(str) + ' (' + round(value_df['objective_vul_std'],2).astype(str) + ')'
    
    value_df['Value (all)'] = round(value_df['value_all'],2).astype(str) + ' (' + round(value_df['value_all_std'],2).astype(str) + ')'
    value_df['Value (vulnerable)'] = round(value_df['value_vul'],2).astype(str) + ' (' + round(value_df['value_vul_std'],2).astype(str) + ')'
    
    value_df['Class. Opt. (all)'] = round(value_df['class_opt_all'],2).astype(str) + ' (' + round(value_df['class_opt_all_std'],2).astype(str) + ')'
    value_df['Class. Opt. (vulnerable)'] = round(value_df['class_opt_vul'],2).astype(str) + ' (' + round(value_df['class_opt_vul_std'],2).astype(str) + ')'
    
    value_df['Class. Flori (all)'] = round(value_df['class_flori_all'],2).astype(str) + ' (' + round(value_df['class_flori_all_std'],2).astype(str) + ')'
    value_df['Class. Flori (vulnerable)'] = round(value_df['class_flori_vul'],2).astype(str) + ' (' + round(value_df['class_flori_vul_std'],2).astype(str) + ')'
    
   
    value_df.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/value_df'+str(sim_setting)+'.csv', index=False)  
    data_norm.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/data_norm.csv', index=False)  


    value_all_hosp_flori.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/value_df_sim_flori_hosp'+ str(sim_setting) +'.csv', index=False)
    value_all_hosp_exp.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/value_df_sim_exp_hosp'+ str(sim_setting) +'.csv', index=False)
    value_all_hosp_naive.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/value_df_sim_naive_hosp'+ str(sim_setting) +'.csv', index=False)

    value_all_subject_flori.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/value_df_sim_flori_sub'+ str(sim_setting) +'.csv', index=False)
    value_all_subject_exp.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/value_df_sim_exp_sub'+ str(sim_setting) +'.csv', index=False)
    value_all_subject_naive.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/value_df_sim_naive_sub'+ str(sim_setting) +'.csv', index=False)
    
    class_opt_all_hosp_flori.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/class_opt_df_sim_flori_hosp'+ str(sim_setting) +'.csv', index=False)
    class_opt_all_hosp_exp.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/class_opt_df_sim_exp_hosp'+ str(sim_setting) +'.csv', index=False)
    class_opt_all_hosp_naive.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/class_opt_df_sim_naive_hosp'+ str(sim_setting) +'.csv', index=False)

    class_opt_all_subject_flori.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/class_opt_df_sim_flori_sub'+ str(sim_setting) +'.csv', index=False)
    class_opt_all_subject_exp.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/class_opt_df_sim_exp_sub'+ str(sim_setting) +'.csv', index=False)
    class_opt_all_subject_naive.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/class_opt_df_sim_naive_sub'+ str(sim_setting) +'.csv', index=False)

    class_flori_all_hosp_flori.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/class_flori_df_sim_flori_hosp'+ str(sim_setting) +'.csv', index=False)
    class_flori_all_hosp_exp.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/class_flori_df_sim_exp_hosp'+ str(sim_setting) +'.csv', index=False)
    class_flori_all_hosp_naive.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/class_flori_df_sim_naive_hosp'+ str(sim_setting) +'.csv', index=False)

    class_flori_all_subject_flori.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/class_flori_df_sim_flori_sub'+ str(sim_setting) +'.csv', index=False)
    class_flori_all_subject_exp.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/class_flori_df_sim_exp_sub'+ str(sim_setting) +'.csv', index=False)
    class_flori_all_subject_naive.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/class_flori_df_sim_naive_sub'+ str(sim_setting) +'.csv', index=False)

    obj_all_hosp_flori.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/obj_df_sim_flori_hosp'+ str(sim_setting) +'.csv', index=False)
    obj_all_hosp_exp.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/obj_df_sim_exp_hosp'+ str(sim_setting) +'.csv', index=False)
    obj_all_hosp_naive.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/obj_df_sim_naive_hosp'+ str(sim_setting) +'.csv', index=False)

    obj_all_subject_flori.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/obj_df_sim_flori_sub'+ str(sim_setting) +'.csv', index=False)
    obj_all_subject_exp.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/obj_df_sim_exp_sub'+ str(sim_setting) +'.csv', index=False)
    obj_all_subject_naive.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/obj_df_sim_naive_sub'+ str(sim_setting) +'.csv', index=False)



    value_vul_hosp_flori.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/value_df_sim_flori_hosp_vul'+ str(sim_setting) +'.csv', index=False)
    value_vul_hosp_exp.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/value_df_sim_exp_hosp_vul'+ str(sim_setting) +'.csv', index=False)
    value_vul_hosp_naive.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/value_df_sim_naive_hosp_vul'+ str(sim_setting) +'.csv', index=False)

    value_vul_subject_flori.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/value_df_sim_flori_sub_vul'+ str(sim_setting) +'.csv', index=False)
    value_vul_subject_exp.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/value_df_sim_exp_sub_vul'+ str(sim_setting) +'.csv', index=False)
    value_vul_subject_naive.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/value_df_sim_naive_sub_vul'+ str(sim_setting) +'.csv', index=False)

    obj_vul_hosp_flori.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/obj_df_sim_flori_hosp_vul'+ str(sim_setting) +'.csv', index=False)
    obj_vul_hosp_exp.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/obj_df_sim_exp_hosp_vul'+ str(sim_setting) +'.csv', index=False)
    obj_vul_hosp_naive.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/obj_df_sim_naive_hosp_vul'+ str(sim_setting) +'.csv', index=False)

    obj_vul_subject_flori.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/obj_df_sim_flori_sub_vul'+ str(sim_setting) +'.csv', index=False)
    obj_vul_subject_exp.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/obj_df_sim_exp_sub_vul'+ str(sim_setting) +'.csv', index=False)
    obj_vul_subject_naive.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/obj_df_sim_naive_sub_vul'+ str(sim_setting) +'.csv', index=False)
    
    class_opt_vul_hosp_flori.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/class_opt_df_sim_flori_hosp_vul'+ str(sim_setting) +'.csv', index=False)
    class_opt_vul_hosp_exp.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/class_opt_df_sim_exp_hosp_vul'+ str(sim_setting) +'.csv', index=False)
    class_opt_vul_hosp_naive.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/class_opt_df_sim_naive_hosp_vul'+ str(sim_setting) +'.csv', index=False)

    class_opt_vul_subject_flori.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/class_opt_df_sim_flori_sub_vul'+ str(sim_setting) +'.csv', index=False)
    class_opt_vul_subject_exp.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/class_opt_df_sim_exp_sub_vul'+ str(sim_setting) +'.csv', index=False)
    class_opt_vul_subject_naive.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/class_opt_df_sim_naive_sub_vul'+ str(sim_setting) +'.csv', index=False)

    class_flori_vul_hosp_flori.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/class_flori_df_sim_flori_hosp_vul'+ str(sim_setting) +'.csv', index=False)
    class_flori_vul_hosp_exp.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/class_flori_df_sim_exp_hosp_vul'+ str(sim_setting) +'.csv', index=False)
    class_flori_vul_hosp_naive.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/class_flori_df_sim_naive_hosp_vul'+ str(sim_setting) +'.csv', index=False)

    class_flori_vul_subject_flori.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/class_flori_df_sim_flori_sub_vul'+ str(sim_setting) +'.csv', index=False)
    class_flori_vul_subject_exp.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/class_flori_df_sim_exp_sub_vul'+ str(sim_setting) +'.csv', index=False)
    class_flori_vul_subject_naive.to_csv('/Users/chenxinlei/RFL-IDR/result/Simulation results/class_flori_df_sim_naive_sub_vul'+ str(sim_setting) +'.csv', index=False)
    
    test_df.to_csv("/Users/chenxinlei/RFL-IDR/result/Simulation results/df2.csv")
