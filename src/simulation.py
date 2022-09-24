"""
Created on August 2022
@author: Leo Guelman
"""

#import os
#os.chdir('/Users/lguelman/Library/Mobile Documents/com~apple~CloudDocs/LG_Files/Presentations/ECML_22/source/experiments')

import numpy as np
import pandas as pd
import sys
sys.path.append('code')
import subset_search
from sklearn import linear_model
import pickle
import plotly.express as px
from plotly.offline import plot

B = 30 # Number of repetitions

def sim(B=30, n_examples_task = 300, n_tasks = 10, n_examples_test_task=10000, n_test_tasks = 1, 
        use_hsic = True,return_mse = False, delta = 0.05):
    
    mse_pool_res = []
    mse_subset_res = []
    
    for i in range(B):
        print("Iteration:", i)        
        np.random.seed(i+10)
        x_p = 3 # number of features
        n_predictors = x_p * 2 + 1 # number of features + T + interactions
        n_ex = []
        alpha = np.random.uniform(-1, 2.5, 5)
        sigma = 1.5
        sx1 = 1
        sx2 = 0.1
        sx3 = 1
        
        train_x = np.zeros((1, n_predictors))
        train_y = np.zeros(1)
        
        for task in range(n_tasks):
            gamma_task = np.random.uniform(0, 1)
            x1 = np.random.normal(0, sx1,(n_examples_task, 1))
            x3 = np.random.normal(0, sx3, (n_examples_task,1))
            T = np.expand_dims(np.random.binomial(1, 0.5, n_examples_task), 1)
            N_y = np.random.normal(0, sigma, (n_examples_task, 1))
            y = alpha[0] * x1 + alpha[1] * x3 + alpha[2] * T + \
                alpha[3] * x1 * T + alpha[4] * x3 * T + N_y
            y1 = alpha[0] * x1 + alpha[1] * x3 + alpha[2] * 1 + \
                 alpha[3] * x1 * 1 + alpha[4] * x3 * 1 + N_y
            y0 = alpha[0] * x1 + alpha[1] * x3 + alpha[2] * 0 + \
                alpha[3] * x1 * 0 + alpha[4] * x3 * 0 + N_y
            x2 = gamma_task*y + np.random.normal(0, sx2, (n_examples_task, 1))
            x_task = np.concatenate([x1, x2, x3, T, x1*T, x2 * T, x3 * T],axis = 1)
            train_x = np.append(train_x, x_task, axis = 0)
            train_y = np.append(train_y, y)
            n_ex.append(n_examples_task)
        
        n_ex = np.array(n_ex)
        train_x =  train_x[1:, :]
        train_y = train_y[1:, np.newaxis]
        
        test_x_T1 = np.zeros((1, n_predictors))
        test_x_T0 = np.zeros((1, n_predictors))
        test_y_T1 = np.zeros(1)
        test_y_T0 = np.zeros(1)
        
        
        for task in range(n_test_tasks):
        
            gamma_task = np.random.uniform(-1, 0)
            x1 = np.random.normal(0, sx1,(n_examples_test_task, 1))
            x3 = np.random.normal(0, sx3, (n_examples_test_task,1))
            T = np.expand_dims(np.random.binomial(1, 0.5, n_examples_test_task), 1)
            N_y = np.random.normal(0, sigma, (n_examples_test_task, 1))
            y = alpha[0] * x1 + alpha[1] * x3 + alpha[2] * T + \
                alpha[3] * x1 * T + alpha[4] * x3 * T + N_y
            y1 = alpha[0] * x1 + alpha[1] * x3 + alpha[2] * 1 + \
                 alpha[3] * x1 * 1 + alpha[4] * x3 * 1 + N_y
            y0 = alpha[0] * x1 + alpha[1] * x3 + alpha[2] * 0 + \
                alpha[3] * x1 * 0 + alpha[4] * x3 * 0 + N_y
            x2 = gamma_task*y + np.random.normal(0, sx2, (n_examples_test_task, 1))
            x_task_T1 = np.concatenate([x1, x2, x3, np.expand_dims(np.repeat(1, n_examples_test_task),1),
                                        x1*1, x2 * 1, x3 * 1],axis = 1)
            x_task_T0 = np.concatenate([x1, x2, x3, np.expand_dims(np.repeat(0, n_examples_test_task),1),
                                        x1*0, x2 * 0, x3 * 0],axis = 1)
            test_x_T1 = np.append(test_x_T1, x_task_T1, axis = 0)
            test_x_T0 = np.append(test_x_T0, x_task_T0, axis = 0)
            #test_x = np.append(test_x, x_task, axis = 0)
            test_y_T1 = np.append(test_y_T1, y1)
            test_y_T0 = np.append(test_y_T0, y0)
        
        test_x_T1 = test_x_T1[1:,:]
        test_x_T0 = test_x_T0[1:,:]
        test_y_T1 = test_y_T1[1:,np.newaxis]
        test_y_T0 = test_y_T0[1:,np.newaxis]
        
        
        s_hat = subset_search.subset(train_x, train_y, n_ex, valid_split=0.5, 
                                     delta=delta, use_hsic=use_hsic)
        
        regr_pool = linear_model.LinearRegression()
        regr_pool.fit(train_x, train_y)
        pred_pool = regr_pool.predict(test_x_T1) - regr_pool.predict(test_x_T0)
        actual = test_y_T1 - test_y_T0
        mse_pool = np.mean((actual - pred_pool) ** 2)
        #print("MSE in test for pooled: %f" % mse_pool)
        
        mse_pool_res.append(mse_pool)
        
        #if s_hat.size >0:
        regr_subset = linear_model.LinearRegression()
        regr_subset.fit(train_x[:,s_hat], train_y)
        pred_subset = regr_subset.predict(test_x_T1[:, s_hat]) - regr_subset.predict(test_x_T0[:, s_hat])
        #else:
        #    pred_subset = np.mean(train_y)
        
        mse_subset = np.mean((actual - pred_subset) ** 2)
        mse_subset_res.append(mse_subset)
        
    return mse_pool_res, mse_subset_res
        

n_examples_task_vals = [200, 400, 800, 1200]
n_tasks_vals = [3, 6, 10, 15, 20]

res = []

for i, e1 in enumerate(n_examples_task_vals):
    for j, e2 in enumerate(n_tasks_vals):
        print(e1)
        print(e2)
        sim1, sim2 = sim(B=B, n_examples_task = n_examples_task_vals[i], n_tasks = n_tasks_vals[j], n_examples_test_task=10000, n_test_tasks = 1, 
        use_hsic = True, return_mse = False, delta = 0.05)
        case = {'n_samples' : e1, 
                'n_environments':e2,
                'pool_err':np.array(sim1), 
                'proposed_err':np.array(sim2),
                }
        res.append(case)


file_name = "res.pkl"

open_file = open(file_name, "wb")
pickle.dump(res, open_file)
open_file.close()

#open_file = open(file_name, "rb")
#res = pickle.load(open_file)
#open_file.close()

res_df = pd.DataFrame.from_dict(res)
res_df1 = res_df[['n_samples', 'n_environments']]

res_df2 = res_df[['pool_err']]
res_df2 = res_df2.to_numpy().flatten()

res_df3 = res_df[['proposed_err']]
res_df3 = res_df3.to_numpy().flatten()

df_ls = []
for i in range(res_df1.shape[0]):
    df = pd.DataFrame.from_dict({'id': range(B),
                                 'n_samples' : res_df1['n_samples'][i], 
                                 'n_environments':res_df1['n_environments'][i],
                                 'pool_err': res_df2[i], 
                                 'proposed_err': res_df3[i]
                                })
    
    df_ls.append(df)
    

df_final = pd.concat(df_ls, axis=0)

df_final = pd.melt(df_final,id_vars=['n_samples', 'n_environments'], value_vars=['pool_err', "proposed_err"], 
           var_name= 'method', value_name ='cate_mse')

df_final['n_samples']= df_final['n_samples'].map(str)

result = df_final.groupby(['n_samples', 'n_environments', 'method'], as_index=False).agg(
                      {'cate_mse':['mean','std']})


result
