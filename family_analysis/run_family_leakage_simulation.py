import pandas as pd
import numpy as np
import scipy.io as sio
import mat73
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
    
    a_mean = a.mean()
    a_sd = a.std()
    a_z = (a-a_mean) / a_sd
    
    return a_z, a_mean, a_sd

def r_to_p(r, n):
    t = r / np.sqrt((1-r**2)/ (n-2) )
    p = 2*stats.t.sf(abs(t), df=n-2)
    return p

# custom scorer (Pearson's r) for grid search
def my_custom_loss_func(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]
    

score = make_scorer(my_custom_loss_func, greater_is_better=True)

# get arguments
parser = argparse.ArgumentParser()
parser.add_argument("--pheno", type=str, help="which phenotype to predict",
                    choices=['mr', 'age', 'attn'],
                    default='mr')
parser.add_argument("--seed", type=int, help="random seed", default=0)

args = parser.parse_args()
pheno = args.pheno
seed = args.seed

# for this manuscript, train in only ABCD
train_dataset = 'abcd'
save_path = '/home/mjr239/project/leakage/family_analysis/percentage_simulation/results/'
k = 5
per_feat = 0.05

# Read in .mat file data
datasets = dict()
dat = mat73.loadmat( os.path.join('/home/mjr239/project/repro_data/', train_dataset+'_feat.mat') )

# read in phenotype and covariates
df_tmp = pd.read_csv( os.path.join('/home/mjr239/project/repro_data/', train_dataset+'_python_table.csv') )
all_possible_covars = ['age', 'sex', 'motion_vals', 'site', 'family_id']
df_include_vars = [c for c in all_possible_covars if ((c in df_tmp.keys()) & (pheno!=c))]
df_include_vars.append(pheno)
df_tmp = df_tmp[df_include_vars]
good_idx = np.where(df_tmp.isna().sum(axis=1)==0)[0]  # remove rows with missing data
df_tmp = df_tmp.iloc[good_idx, :]
df_tmp['sex'] = df_tmp.sex.replace('F', 0).replace('M', 1)  # replace sex variables

# add this to dictionary of datasets
datasets[train_dataset] = dict()
datasets[train_dataset]['X'] = dat['X'][:, good_idx]   
datasets[train_dataset]['behav'] = df_tmp    

# add in number of participants for each family
datasets[train_dataset]['behav']['family_freq'] = datasets[train_dataset]['behav']['family_id'].map(datasets[train_dataset]['behav']['family_id'].value_counts())

# get covariate keys
all_keys = datasets[train_dataset]['behav'].keys()
covar_keys = [k for k in all_keys if ((k!=pheno) and (k!='site') and (k!='family_id') 
                                     and (k!='family_freq') and (k!='site'))]


# discard sites from multi-family data with too few participants
multi_family_idx = np.where(np.array(datasets[train_dataset]['behav']['family_freq'])>1)[0]  # indices of participants coming from families with multiple members
n_multi = len(multi_family_idx)
# extract data of participants coming from families with multiple members
X_multi = datasets[train_dataset]['X'][:, multi_family_idx].T
y_multi = np.array(datasets[train_dataset]['behav'][pheno])[multi_family_idx]
site_multi = np.array(datasets[train_dataset]['behav']['site'])[multi_family_idx]
C_multi = np.array(datasets[train_dataset]['behav'][covar_keys])[multi_family_idx, :]
fam_id_multi = np.array(datasets[train_dataset]['behav'].family_id)[multi_family_idx]

# single family data
single_family_idx = np.where(np.array(datasets[train_dataset]['behav']['family_freq'])==1)[0]  # indices of participants coming from single-member family
n_single = len(single_family_idx)
# extract data from participants coming from single-member family
X_single = datasets[train_dataset]['X'][:, single_family_idx].T
y_single = np.array(datasets[train_dataset]['behav'][pheno])[single_family_idx]
site_single = np.array(datasets[train_dataset]['behav']['site'])[single_family_idx]
C_single = np.array(datasets[train_dataset]['behav'][covar_keys])[single_family_idx, :]
fam_id_single = np.array(datasets[train_dataset]['behav'].family_id)[single_family_idx]        

print('Single family size: {:d}, multi family size: {:d}'.format(n_single, n_multi))

del datasets  # delete to save memory

perc_min = n_multi / (n_single + n_multi)  # minimum possible percentage for subsampling
for perc_family in [s for s in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] if s>perc_min]:

    # randomly sample multi family data
    n_single_sample = int(np.round(n_multi*(1/perc_family-1)))
    np.random.seed(seed)
    shuffle_idx = np.random.permutation(n_single_sample)

    # include single family data
    X_tmp = np.vstack(( X_multi, X_single[ shuffle_idx[:n_single_sample], :] ))
    y_tmp = np.hstack(( y_multi, y_single[shuffle_idx[:n_single_sample]] ))
    site_tmp = np.hstack((site_multi, site_single[ shuffle_idx[:n_single_sample] ]))
    C_all_tmp = np.vstack((C_multi, C_single[ shuffle_idx[:n_single_sample], :]))
    family_ids_tmp = np.hstack(( fam_id_multi, fam_id_single[shuffle_idx[:n_single_sample]] ))
    
    # modify so only sites with enough participants are included
    # remove if too few participants per site
    siteval, sitecount = np.unique(site_tmp, return_counts=True)  # unique sites and their counts
    sites_to_keep = siteval[np.where(sitecount>=10)[0]]  # only include sites with at least 10 participants
    idx_sites_to_keep = np.where( np.isin(site_tmp, sites_to_keep) )[0]  # find data matching sites that we should keep

    # update with removed sites
    X = X_tmp[idx_sites_to_keep, :]
    y = y_tmp[idx_sites_to_keep]
    site = site_tmp[idx_sites_to_keep]
    C_all = C_all_tmp[idx_sites_to_keep, :]
    family_ids = family_ids_tmp[idx_sites_to_keep]
    del X_tmp, y_tmp, site_tmp, C_all_tmp, family_ids_tmp

    for leak_family in [True, False]:

        print('Pheno: {:s}, percentage family: {:.2f}, leakage: {:s}'.format( pheno, perc_family, str(leak_family) ))
        
        # set save name and check if it exists
        save_name = os.path.join(save_path,
                                 'family_percentage_' + str(perc_family) + '_leak_' + str(leak_family) + \
                                 '_pheno_' +pheno + '_' + train_dataset + '_k' + str(k) + '_perfeat' + str(per_feat) + \
                                 '_seed_' + str(seed) +  '.npz'
                                )
        # skip if file exists already
        if os.path.isfile(save_name):
            print('exists')
            continue


        # kfold splits will be determined by whether or not there is family leakage
        if leak_family: 
            n_split = len(y)
        else:
            unique_family_ids = np.unique(family_ids)
            n_split = len(unique_family_ids)

        # kfold cross-validation
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        yp = np.zeros((len(y),)) 
        fold_assignment = np.zeros((len(y),))
        coef_all = np.zeros((X.shape[1], k))
        for fold_idx, (train_idx, test_idx) in tqdm(enumerate(kf.split(np.arange(n_split)))):
            # modify train/test selection for family structure
            if leak_family:
                pass
            else:
                family_train_ids = unique_family_ids[train_idx]
                family_test_ids = unique_family_ids[test_idx]
                train_idx = np.where([id in family_train_ids for id in family_ids])[0]
                test_idx = np.where([id in family_test_ids for id in family_ids])[0]                    

            # get train/test splits
            X_train = X[train_idx, :].copy()
            y_train = y[train_idx].copy()
            X_test = X[test_idx, :].copy()
            y_test = y[test_idx].copy()
            fold_assignment[test_idx] = fold_idx

            # correct for covars
            ntrain, _ = X_train.shape
            C_train = C_all[train_idx, :]
            Beta = np.matmul( np.matmul( np.linalg.inv( np.matmul(C_train.T, C_train)), C_train.T), X_train)
            X_train = X_train - np.matmul(C_train, Beta)
            C_test = C_all[test_idx, :]
            X_test = X_test - np.matmul(C_test, Beta)  
            
            # correct for site 
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

            # fit model: inner cv with only 2 folds
            inner_cv = KFold(n_splits=2, shuffle=True, random_state=seed)

            # Ridge regression
            regr = GridSearchCV(estimator=Ridge(), param_grid={'alpha':np.logspace(-3, 3, 7)},
                           cv=inner_cv, scoring=score)
            regr.fit(X_train[:, sig_feat_loc], y_train)
            # now test
            yp[test_idx] = regr.predict(X_test[:, sig_feat_loc]) 
            

            # save results
            np.savez(save_name, yp=yp, y_true=y, fold_assignment=fold_assignment)
