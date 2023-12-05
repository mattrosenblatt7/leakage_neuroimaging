import pandas as pd
import numpy as np
import scipy.io as sio
import os
import multiprocessing
import time
import random
from tqdm import tqdm
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_regression, r_regression
import argparse
from scipy import stats
from neuroCombat import neuroCombat, neuroCombatFromTraining
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

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

# Set up parser and arguments
parser = argparse.ArgumentParser()
parser.add_argument("--leakage_type", type=str, help="which leakage to perform",
                    choices=['gold', 'gold_zscore', 'leak_zscore',
                            'leak_feature', 'leak_site', 'leak_covars',
                            'leak_family', 'leak_subj_5', 
                             'leak_subj_10', 'leak_subj_20',
                             'gold_minus_site', 'gold_minus_covars',
                            'gold_minus_site_covars'],
                    default='gold')
parser.add_argument("--start_seed", type=int, help="start seed", default=0)
parser.add_argument("--end_seed", type=int, help="end seed", default=99)
parser.add_argument("--k", type=int, help="k-folds", default=5)
parser.add_argument("--per_feat", type=float, help="percentage of features", default=0.05)
parser.add_argument("--dti", type=int, help="running dti supplemental analysis", default=0)
parser.add_argument("--model_type", type=str, help="which model to use",
                    choices=['ridge', 'svr', 'cpm'],
                    default='ridge')

# Parse arguments
args = parser.parse_args()
leakage_type = args.leakage_type
start_seed = args.start_seed
end_seed = args.end_seed
k = args.k
per_feat = args.per_feat
dti = args.dti
model_type = args.model_type

# custom scorer (Pearson's r) for grid search
def my_custom_loss_func(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]
score = make_scorer(my_custom_loss_func, greater_is_better=True)

# Load in dataset
if 'site' in leakage_type:
    dataset_names = ['hbn', 'hcpd', 'abcd']  # no site info for PNC
elif leakage_type=='leak_family':
    dataset_names = ['hcpd', 'abcd']  # no family info for HBN or PNC
else:
    dataset_names = ['hbn', 'hcpd', 'pnc', 'abcd']

# supplemental analyses (dti)
if dti==1:
    dataset_names = ['hcpddti']
    save_path = '/home/mjr239/project/leakage/leakage_results/dti_results'
    print(save_path)
elif model_type=='svr':
    save_path = '/home/mjr239/project/leakage/leakage_results/svr_results'
    print(save_path)
elif model_type=='cpm':
    save_path = '/home/mjr239/project/leakage/leakage_results/cpm_results'
    print(save_path)
else:
    save_path = '/home/mjr239/project/leakage/leakage_results/'

# three phenotypes in this study
pheno_all = ['mr', 'attn', 'age']  

# loop over phenotypes   
for pheno in pheno_all:   
    
    # Load all possible datasets for this specific phenotype
    for train_dataset in dataset_names:
        datasets = dict()
        for dname in dataset_names:
            dat = sio.loadmat( os.path.join('/home/mjr239/project/repro_data/', dname+'_feat.mat') )
            
            # read in phenotype and covariates
            df_tmp = pd.read_csv( os.path.join('/home/mjr239/project/repro_data/', dname+'_python_table.csv') )
            all_possible_covars = ['age', 'sex', 'motion_vals', 'site', 'family_id']
            df_include_vars = [c for c in all_possible_covars if ((c in df_tmp.keys()) & (pheno!=c))]
            df_include_vars.append(pheno)
            df_tmp = df_tmp[df_include_vars]
            good_idx = np.where(df_tmp.isna().sum(axis=1)==0)[0]  # remove rows with missing data
            df_tmp = df_tmp.iloc[good_idx, :]
            df_tmp['sex'] = df_tmp.sex.replace('F', 0).replace('M', 1)  # replace sex variables

            # add this to dictionary of datasets
            datasets[dname] = dict()
            datasets[dname]['X'] = dat['X'][:, good_idx]   
            datasets[dname]['behav'] = df_tmp         

    # loop over seeds
    for seed in range(start_seed, end_seed+1):

        # loop over training sets
        for train_dataset in dataset_names:
            
            # set save name
            save_name = os.path.join(save_path,
                         leakage_type,
                         pheno + '_' + train_dataset + '_k' + str(k) + '_perfeat' + str(per_feat) + \
                                     '_seed_' + str(seed) +  '.npz'
                        )
            skip if file exists already
            if os.path.isfile(save_name):  # skip if file was already saved
                print('exists')
                continue
       
            # set x and y data
            X = datasets[train_dataset]['X'].T
            y = np.array(datasets[train_dataset]['behav'][pheno])
            n = len(y)

            # covariates
            all_keys = datasets[train_dataset]['behav'].keys()
            covar_keys = [k for k in all_keys if ((k!=pheno) and (k!='site') and (k!='family_id'))]
            C_all = np.array(datasets[train_dataset]['behav'][covar_keys])
            if 'site' in all_keys:
                site = np.array(datasets[train_dataset]['behav']['site'])

            # subject leakage - add additional subjects to dataset
            if 'leak_subj' in leakage_type:
                random.seed(seed)
                perc_leakage = int(leakage_type.split('_')[-1])  # get percentage of subject leakage (5, 10, 20% in this study)
                print('******** Leakage percentage ' + str(perc_leakage) + '*********')
                rs_idx = np.array(random.sample(range(n), round(perc_leakage/100*n)))  # random sample idx
                
                # repeat data points for brain data (X), behavioral data (y), site, and covariates
                X = np.vstack((X, X[rs_idx, :]))
                y = np.hstack((y, y[rs_idx]))
                if 'site' in all_keys:
                    site = np.hstack((site, site[rs_idx]))
                C_all = np.vstack((C_all, C_all[rs_idx, :]))
                n = len(y)

            # feature selection leakage
            if leakage_type=='leak_feature':
                r = r_regression(X, y)
                p = r_to_p(r, n) 
                pthresh = np.percentile(p, 100*per_feat)
                leaky_feat_loc = np.where(p<pthresh)[0]
            else:
                pass

            # z-score behavioral variable leakage
            if leakage_type=='leak_zscore':
                y, _, _ = zscore(y)

            # z-scoring without leakage
            if leakage_type=='gold_zscore':  # initialize this so we can save
                y_z = np.zeros((n,))

            # leakage for covariate regression in whole dataset
            if leakage_type=='leak_covars':
                ntrain, _ = X.shape
                Beta = np.matmul( np.matmul( np.linalg.inv( np.matmul(C_all.T, C_all)), C_all.T), X)
                X = X - np.matmul(C_all, Beta)    

            # site correction (i.e., ComBat) leakage
            if leakage_type=='leak_site':
                covars = {'batch':site} 
                covars = pd.DataFrame(covars)  
                data_combat = neuroCombat(dat=X.T,
                    covars=covars,
                    batch_col='batch')
                X = data_combat['data'].T

            # kfold CV
            if (leakage_type=='leak_family') or ('family_id' not in datasets[train_dataset]['behav'].keys()) \
  or ('subj' in leakage_type): 
                n_split = n  # if no family information, split on the subject (not family) level
            else:  # if there is family information, split on the family (not subject) level
                family_ids = np.array(datasets[train_dataset]['behav']['family_id'])
                unique_family_ids = np.unique(family_ids)
                n_split = len(unique_family_ids)
            
            # k-fold splits
            kf = KFold(n_splits=k, shuffle=True, random_state=seed)
            yp = np.zeros((n,)) # initialize predicted y
            fold_assignment = np.zeros((n,))  # initialize which fold each point is in 
            coef_all = np.zeros((X.shape[1], k))  # initialize coefficients
            
            # loop over folds
            for fold_idx, (train_idx, test_idx) in tqdm(enumerate(kf.split(np.arange(n_split)))):
                # modify train/test selection for family structure
                if (leakage_type=='leak_family') or ('family_id' not in datasets[train_dataset]['behav'].keys()) \
  or ('subj' in leakage_type):
                    pass
                else:  # get train/test indices based on family structure
                    family_train_ids = unique_family_ids[train_idx]
                    family_test_ids = unique_family_ids[test_idx]
                    train_idx = np.where([id in family_train_ids for id in family_ids])[0]
                    test_idx = np.where([id in family_test_ids for id in family_ids])[0]                    

                # get train/test splits
                X_train = X[train_idx, :]
                y_train = y[train_idx]
                X_test = X[test_idx, :]
                y_test = y[test_idx]
                fold_assignment[test_idx] = fold_idx

                # proper within-fold z-scoring
                if leakage_type=='gold_zscore':
                    y_train, y_train_mean, y_train_sd = zscore(y_train)
                    y_test = (y_test - y_train_mean) / y_train_sd
                    y_z[test_idx] = y_test  # save z-scored for comparison

                # correct for covars (unless doing covariate leakage)
                if (leakage_type!='leak_covars') and (leakage_type!='gold_minus_covars') and (leakage_type!='gold_minus_site_covars'):
                    # fit in training data
                    ntrain, _ = X_train.shape
                    C_train = C_all[train_idx, :]
                    Beta = np.matmul( np.matmul( np.linalg.inv( np.matmul(C_train.T, C_train)), C_train.T), X_train)
                    X_train = X_train - np.matmul(C_train, Beta)

                    # apply covariate regression to test data
                    C_test = C_all[test_idx, :]
                    X_test = X_test - np.matmul(C_test, Beta)          

                # correct for site (only if site is a variable and we are not looking at site leakage)
                if ('site' in all_keys) and (leakage_type!='leak_site') \
                  and (leakage_type!='gold_minus_site') and (leakage_type!='gold_minus_site_covars'):
                    # first performn it in the training data
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
                if leakage_type=='leak_feature':
                    sig_feat_loc = np.copy(leaky_feat_loc)  # feature leakage, use features from whole dataset
                else:  # otherwise get features in training data only
                    r = r_regression(X_train, y_train)
                    p = r_to_p(r, len(train_idx)) 
                    pthresh = np.percentile(p, 100*per_feat)
                    sig_feat_loc = np.where(p<pthresh)[0]

                # fit model                           
                if train_dataset=='abcd':  # split-half if ABCD and SVR
                    inner_cv = KFold(n_splits=2, shuffle=True, random_state=seed)
                else:
                    inner_cv = KFold(n_splits=k, shuffle=True, random_state=seed) 
                
                if model_type=='svr':  # support vector regression
                    regr = GridSearchCV(estimator=SVR(kernel='rbf'), param_grid={'C':np.logspace(-3, 3, 7)},
                                   cv=inner_cv, scoring=score)
                    regr.fit(X_train[:, sig_feat_loc], y_train)
                    
                    # now test
                    yp[test_idx] = regr.predict(X_test[:, sig_feat_loc]) 
                elif model_type=='cpm':
                    # get cpm masks
                    feat_sign = np.sign(r[sig_feat_loc])
                    pos_mask = np.where(feat_sign>0)[0]
                    neg_mask = np.where(feat_sign<0)[0]
                    
                    # get cpm feature summary
                    X_train_summary = X_train[:, sig_feat_loc[pos_mask]].sum(axis=1) - X_train[:, sig_feat_loc[neg_mask]].sum(axis=1)
                    X_test_summary = X_test[:, sig_feat_loc[pos_mask]].sum(axis=1) - X_test[:, sig_feat_loc[neg_mask]].sum(axis=1)

                    # fit linear regression
                    regr = LinearRegression()
                    regr.fit(X_train_summary.reshape(-1, 1), y_train)
                    
                    # now test
                    yp[test_idx] = regr.predict(X_test_summary.reshape(-1, 1))
                elif model_type=='ridge':  # ridge regression
                    regr = GridSearchCV(estimator=Ridge(), param_grid={'alpha':np.logspace(-3, 3, 7)},
                                   cv=inner_cv, scoring=score)
                    regr.fit(X_train[:, sig_feat_loc], y_train)
                    
                    # now test
                    yp[test_idx] = regr.predict(X_test[:, sig_feat_loc]) 

                # store coefficients (for ridge only)
                if model_type=='ridge':
                    coef_all[sig_feat_loc, fold_idx] = regr.best_estimator_.coef_
                      
            # save results
            if model_type=='ridge':  # save coef for ridge only
                if leakage_type=='gold_zscore':  # if z-scoring within folds, save the new "y" data
                    np.savez(save_name, yp=yp, y_true=y_z, coef=np.mean(coef_all, axis=1),
                            fold_assignment=fold_assignment)
                else:
                    np.savez(save_name, yp=yp, y_true=y, coef=np.mean(coef_all, axis=1),
                            fold_assignment=fold_assignment)
            else:
                if leakage_type=='gold_zscore':  # if z-scoring within folds, save the new "y" data
                    np.savez(save_name, yp=yp, y_true=y_z, coef=np.nan,
                            fold_assignment=fold_assignment)
                else:
                    np.savez(save_name, yp=yp, y_true=y, coef=np.nan,
                            fold_assignment=fold_assignment)
