import pandas as pd
import numpy as np
import scipy.io as sio
import os
import time
import random
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.feature_selection import SelectPercentile, f_regression, r_regression
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
import argparse
import mat73
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from neuroCombat import neuroCombat, neuroCombatFromTraining
from sklearn.linear_model import LinearRegression


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, help="random seed", default=0)
parser.add_argument("--pheno", type=str, help="which phenotype to predict",
                    choices=['cbcl_scr_syn_anxdep_r', 'cbcl_scr_syn_internal_r', 'cbcl_scr_syn_external_r',
                    'cbcl_scr_syn_aggressive_r', 'age', 'mr', 'attn'],
                    default='age')
args = parser.parse_args()
seed = args.seed
pheno = args.pheno



def zscore(a):
    '''
    Function to return a z-scored version of input a
    '''    
    a_mean = a.mean()
    a_sd = a.std()
    a_z = (a-a_mean) / a_sd
    
    return a_z, a_mean, a_sd

def r_to_p(r, n):
    '''
    Function to convert r values to p values
    '''
    t = r / np.sqrt((1-r**2)/ (n-2) )
    p = 2*stats.t.sf(abs(t), df=n-2)
    return p

# custom scorer (Pearson's r) for grid search
def my_custom_loss_func(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]
score = make_scorer(my_custom_loss_func, greater_is_better=True)


dname = 'abcd_twins'  # dataset name
k = 20  # folds of cross-validation
k_inner = 5  # nested folds for grid search
per_feat = 0.05

# # loop over phenotypes   
results_pheno = []
results_model_type = []
results_leak_twins = []
results_seed = []
results_r = []
for seed in range(seed, seed+1):
        
    # loop over all models (ridge regression, connectome-based predictive modeling, support vector regression, random forest regression)
    for model_type in ['ridge', 'cpm', 'svr', 'rf']: 
        
        # loop over case of leakage and no leakage
        for leak_twins in [True, False]:

            # Load all possible datasets for this specific phenotype
            dat = sio.loadmat( os.path.join('/home/mjr239/project/repro_data/test_abcd', dname+'_feat.mat') )

            # read in phenotype and covariates
            df_tmp = pd.read_csv( os.path.join('/home/mjr239/project/repro_data/test_abcd', dname+'_python_table.csv') )
            all_possible_covars = ['age', 'sex', 'motion_vals', 'site', 'family_id']
            df_include_vars = [c for c in all_possible_covars if ((c in df_tmp.keys()) & (pheno!=c))]
            df_include_vars.append(pheno)
            df_tmp = df_tmp[df_include_vars]
            df_tmp['sex'] = df_tmp.sex.replace('F', 0).replace('M', 1)  # replace sex variables

            # delete entries missing any data
            good_idx = np.where(df_tmp.isna().sum(axis=1)==0)[0]  # remove rows with missing data
            df_tmp = df_tmp.iloc[good_idx, :].reset_index(drop=True)
            X_tmp = np.copy(dat['X'][:, good_idx])

            # now delete entries which are left with only one twin
            df_tmp['family_freq'] = df_tmp['family_id'].map(df_tmp['family_id'].value_counts())
            good_idx = np.where(df_tmp.family_freq==2)[0]  # remove rows with only one twin
            df_tmp2 = df_tmp.iloc[good_idx, :].reset_index(drop=True)
            X_tmp2 = np.copy(X_tmp[:, good_idx])
            
            # remove if too few participants per site
            siteval, sitecount = np.unique(df_tmp2.site, return_counts=True)

            if np.min(sitecount)>=5:
                df_covar = df_tmp.iloc[good_idx, :].reset_index(drop=True)
                X = np.copy(X_tmp[:, good_idx]).T
                
            else:  
                df_tmp2['site_freq'] = df_tmp2['site'].map(df_tmp2['site'].value_counts())
                good_idx = np.where( df_tmp2.site_freq>=6 )[0]  # remove if too few per site
                df_covar = df_tmp2.iloc[good_idx, :].reset_index(drop=True)
                X = np.copy(X_tmp2[:, good_idx]).T

            del df_tmp, df_tmp2, X_tmp, X_tmp2

            # get behavior variable
            y = np.array(df_covar[pheno]).ravel()
            n = len(y)

            # get covariates
            all_keys = df_covar.keys()
            covar_keys = [k for k in all_keys if ((k!=pheno) and (k!='site') and (k!='family_id') 
                                                  and (k!='family_freq') and (k!='site_freq'))]
            C_all = np.array(df_covar[covar_keys])
            if 'site' in all_keys:
                site = np.array(df_covar['site'])


            # kfold CV
            if leak_twins: 
                n_split = n
            else:
                family_ids = np.array(df_covar['family_id'])
                unique_family_ids = np.unique(family_ids)
                n_split = len(unique_family_ids)

            kf = KFold(n_splits=k, shuffle=True, random_state=seed)
            yp = np.zeros((n,)) 
            fold_assignment = np.zeros((n,))

            for fold_idx, (train_idx, test_idx) in tqdm(enumerate(kf.split(np.arange(n_split)))):
                # modify train/test selection for family structure
                if leak_twins:
                    pass
                else:  # sample by family (twins)
                    family_train_ids = unique_family_ids[train_idx]
                    family_test_ids = unique_family_ids[test_idx]
                    train_idx = np.where([sub_id in family_train_ids for sub_id in family_ids])[0]
                    test_idx = np.where([sub_id in family_test_ids for sub_id in family_ids])[0]                    

                # get train/test splits
                X_train = X[train_idx, :]
                y_train = y[train_idx]
                X_test = X[test_idx, :]
                y_test = y[test_idx]
                fold_assignment[test_idx] = fold_idx



                # correct for covars
                ntrain, _ = X_train.shape
                C_train = C_all[train_idx, :]
                Beta = np.matmul( np.matmul( np.linalg.inv( np.matmul(C_train.T, C_train)), C_train.T), X_train)
                X_train = X_train - np.matmul(C_train, Beta)

                C_test = C_all[test_idx, :]
                X_test = X_test - np.matmul(C_test, Beta)     


                # Site correction
                covars = {'batch':site[train_idx]}
                covars = pd.DataFrame(covars) 
                data_combat = neuroCombat(dat=X_train.T,
                                          covars=covars,
                                          batch_col='batch')
                X_train = data_combat['data'].T
                # apply to test data
                test_combat_results = neuroCombatFromTraining(dat=X_test.T,
                                                  batch=site[test_idx],
                                                  estimates=data_combat['estimates'])
                X_test = test_combat_results['data'].T

                # feature selection 
                r = r_regression(X_train, y_train)
                p = r_to_p(r, len(train_idx)) 
                pthresh = np.percentile(p, 100*per_feat)
                sig_feat_loc = np.where(p<pthresh)[0]

                # fit model                           
                inner_cv = KFold(n_splits=k_inner, shuffle=True, random_state=seed) 

                if model_type=='svr':
                    regr = GridSearchCV(estimator=SVR(kernel='rbf'), param_grid={'C':np.logspace(-3, 3, 7)},
                                   cv=inner_cv, scoring=score)
                    regr.fit(X_train[:, sig_feat_loc], y_train)
                    # now test
                    yp[test_idx] = regr.predict(X_test[:, sig_feat_loc]) 
                elif model_type=='cpm':
                    feat_sign = np.sign(r[sig_feat_loc])
                    pos_mask = np.where(feat_sign>0)[0]
                    neg_mask = np.where(feat_sign<0)[0]

                    X_train_summary = X_train[:, sig_feat_loc[pos_mask]].sum(axis=1) - X_train[:, sig_feat_loc[neg_mask]].sum(axis=1)
                    X_test_summary = X_test[:, sig_feat_loc[pos_mask]].sum(axis=1) - X_test[:, sig_feat_loc[neg_mask]].sum(axis=1)

                    regr = LinearRegression()
                    regr.fit(X_train_summary.reshape(-1, 1), y_train)
                    # now test
                    yp[test_idx] = regr.predict(X_test_summary.reshape(-1, 1))
                elif model_type=='rf':  # random forest
                    regr = GridSearchCV(estimator=RandomForestRegressor(n_estimators=10),
                                        param_grid={'max_depth':[3, 5, 7, 9]},
                                        cv=inner_cv, scoring=score)
                    regr.fit(X_train[:, sig_feat_loc], y_train)
                    # now test
                    yp[test_idx] = regr.predict(X_test[:, sig_feat_loc]) 
                elif model_type=='ridge':
                    regr = GridSearchCV(estimator=Ridge(), param_grid={'alpha':np.logspace(-3, 3, 7)},
                                   cv=inner_cv, scoring=score)
                    regr.fit(X_train[:, sig_feat_loc], y_train)
                    # now test
                    yp[test_idx] = regr.predict(X_test[:, sig_feat_loc]) 

            results_seed.append(seed)
            results_pheno.append(pheno)
            results_model_type.append(model_type)
            results_r.append( np.corrcoef(y, yp)[0, 1] )
            results_leak_twins.append(leak_twins)

# make dataframe of results
df_results = pd.DataFrame({'pheno':results_pheno, 'model_type':results_model_type,
                          'leak_twins':results_leak_twins, 'seed':results_seed,
                          'r':results_r})
                          
# save results
df_results.to_csv('/home/mjr239/project/leakage/family_analysis/results/results_seed_' + str(seed) + '_pheno_' + pheno + '.csv', index=False)


