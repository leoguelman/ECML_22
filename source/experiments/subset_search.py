import numpy as np
import scipy as sp
from sklearn import linear_model
from sklearn import svm
from sklearn import metrics
import itertools
import utils
import sys

# Source Rojas-Carulla
def full_search(train_x, train_y, valid_x, valid_y, n_ex, n_ex_valid, 
                use_hsic, alpha, return_n_best=None):
    """
    Perform search over all possible subsets of features. 

    Args:
        dataset: internal dataset object.
        use_hsic: whether to use HSIC. If not, Levene test is used.
        alpha: level for the statistical test of equality of distributions 
          (HSIC or Levene).
        return_n_best: return top n subsets (in terms of test statistic). 
          Default returns only the best subset. 

    """
    num_tasks = len(n_ex)
    n_ex_cum = np.cumsum(n_ex)

    index_task = 0
    best_subset = []
    accepted_sets = []
    accepted_mse = []
    all_sets = []
    all_pvals = []

    num_s = np.sum(n_ex)
    num_s_valid = np.sum(n_ex_valid)
    best_mse = 1e10

    rang = np.arange(train_x.shape[1])
    maxevT = -10
    maxpval = 0
    num_accepted = 0
    current_inter = np.arange(train_x.shape[1])

    #Get numbers for the mean
    pred_valid = np.mean(train_y)
    residual = valid_y - pred_valid

    if use_hsic:
        valid_dom = utils.mat_hsic(valid_y, n_ex_valid)
        ls = utils.np_getDistances(residual, residual)
        sx = 0.5 * np.median(ls.flatten())
        stat, a, b = utils.numpy_HsicGammaTest(residual, valid_dom,
                                               sx, 1, 
                                               DomKer = valid_dom)
        pvals = 1. - sp.stats.gamma.cdf(stat, a, scale=b)
    else:
        residTup = utils.levene_pval(residual, n_ex, num_tasks)
        pvals = sp.stats.levene(*residTup)[1]

    if (pvals > alpha):
        mse_current  = np.mean((valid_y - pred_valid) ** 2)
        if mse_current < best_mse:
            best_mse = mse_current
            best_subset = []
            accepted_sets.append([])
            accepted_mse.append(mse_current)
    
    all_sets.append([])
    all_pvals.append(pvals)

    for i in range(1, rang.size + 1):
        for s in itertools.combinations(rang, i):
            currentIndex = rang[np.array(s)]
            regr = linear_model.LassoCV()
            
            #Train regression with given subset on training data
            regr.fit(train_x[:, currentIndex], 
                     train_y.flatten())

            #Compute mse for the validation set
            pred = regr.predict(
              valid_x[:, currentIndex])[:,np.newaxis]

            #Compute residual
            residual = valid_y - pred

            if use_hsic:
                valid_dom = utils.mat_hsic(valid_y, n_ex_valid)
                ls = utils.np_getDistances(residual, residual)
                sx= 0.5 * np.median(ls.flatten())
                stat, a, b = utils.numpy_HsicGammaTest(
                    residual, valid_dom, sx, 1, DomKer = valid_dom)
                pvals = 1.- sp.stats.gamma.cdf(stat, a, scale=b)
            else:
                residTup = utils.levene_pval(residual, n_ex_valid, num_tasks)
                pvals = sp.stats.levene(*residTup)[1]
            
            all_sets.append(s)
            all_pvals.append(pvals)
                                                                            
            if (pvals > alpha):
                mse_current = np.mean((pred - valid_y) ** 2)
                if mse_current < best_mse: 
                    best_mse = mse_current
                    best_subset = s
                    current_inter = np.intersect1d(current_inter, s)
                    accepted_sets.append(s)
                    accepted_mse.append(mse_current)


    if len(accepted_sets) == 0:
        all_pvals = np.array(all_pvals).flatten()
        sort_pvals = np.argsort(all_pvals)
        idx_max = sort_pvals[-1]
        best_subset = all_sets[idx_max]
        accepted_sets.append(best_subset)

    if return_n_best:
        return [np.array(s) for s in accepted_sets[-return_n_best:]]
    else:
        return np.array(best_subset)



def subset(x, y, n_ex, delta, valid_split, use_hsic = False, 
           return_n_best = None):

    """
    Run Algorithm 1 for full subset search. 

    Args:
        x: train features. Shape [n_examples, n_features].
        y: train labels. Shape [n_examples, 1].
        n_ex: list with number of examples per task (should be ordered in 
          train_x and train_y). Shape: [n_tasks]
        delta: Significance level of statistical test.
        use_hsic: use HSIC? If False, Levene is used. 
        return_n_best: number of subsets to return. 
    """

    train_x, train_y, valid_x, valid_y, n_ex_train, n_ex_valid = \
      utils.split_train_valid(x, y, n_ex, valid_split)
    
    subset = full_search(train_x, train_y, valid_x, valid_y,
                         n_ex_train, n_ex_valid, use_hsic, 
                         delta, return_n_best = return_n_best)

    return subset


