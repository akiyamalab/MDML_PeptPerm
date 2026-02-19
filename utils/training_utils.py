import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from scipy.stats import pearsonr


from sklearn.linear_model import Lasso, Ridge
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor


import socket
host = socket.gethostname()
if host != 'MacBook-Pro.local':
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor



def delete_similarity(X, y, threshold=0.95):
    """
    For descriptor pairs whose correlation > th,
    eliminate the one with the lower correlation with the objective variable y.
    """
    features_delete = set()

    for i in range(X.shape[1]):
        if X.columns[i] in features_delete:
            continue
        a = X.iloc[:, i].values
        for j in range(i):
            if X.columns[j] in features_delete:
                continue
            b = X.iloc[:, j].values
            R = abs(pearsonr(a, b)[0])
            if R > threshold:
                cor_a = abs(pearsonr(a, y)[0])
                cor_b = abs(pearsonr(b, y)[0])
                if cor_a <= cor_b:
                    features_delete.add(X.columns[i])
                else:
                    features_delete.add(X.columns[j])

    return list(features_delete)



def create_model(name, is_search=True, best_params=None):
    """
    Create a model for grid search / with the best parameters.
    """
    if not is_search:
        if name == 'Lasso':
            model = Lasso(alpha=best_params['alpha'], max_iter=10000)
        elif name == 'Ridge':
            model = Ridge(alpha=best_params['alpha'])
        elif name == 'SVM':
            model = svm.SVR(C=best_params['C'], epsilon=best_params['epsilon'], gamma=best_params['gamma'])
        elif name == 'RF':
            if best_params['max_depth'] not in [2, 5, 10, 15, 20]:
                max_depth = None
            else:
                max_depth = int(best_params['max_depth'])
            if type(best_params['max_features']) != str:
                max_features = None
            else:
                max_features = best_params['max_features']
            model = RandomForestRegressor(n_estimators=best_params['n_estimators'], max_depth=max_depth, \
                                          min_samples_split=best_params['min_samples_split'], max_features=max_features, random_state=2024)
        elif name == 'XGBoost':
            model = XGBRegressor(random_state=2024, device='cpu', n_estimators=int(best_params['n_estimators']), learning_rate=best_params['learning_rate'], \
                                max_depth=int(best_params['max_depth']), min_child_weight=int(best_params['min_child_weight']), gamma=best_params['gamma'], \
                                subsample=best_params['subsample'], reg_alpha=best_params['reg_alpha'], reg_lambda=best_params['reg_lambda'],n_jobs=1)
        elif name == 'LightGBM':
            model = LGBMRegressor(random_state=2024, verbosity=-1,
                                  n_estimators=int(best_params['n_estimators']), learning_rate=best_params['learning_rate'], max_depth=int(best_params['max_depth']), \
                                  min_data_in_leaf=int(best_params['min_data_in_leaf']), num_leaves=int(best_params['num_leaves']), \
                                  #   min_child_weight=best_params['min_child_weight'], subsample=best_params['subsample'],
                                  reg_alpha=best_params['reg_alpha'], reg_lambda=best_params['reg_lambda'])
    else:
        if name == 'Lasso':
            model = Lasso(max_iter=10000)
        elif name == 'Ridge':
            model = Ridge()
        elif name == 'SVM':
            model = svm.SVR()
        elif name == 'RF':
            model = RandomForestRegressor(random_state=2024)
        elif name == 'XGBoost':
            model = XGBRegressor(random_state=2024, device='cpu',n_jobs=1,verbosity=2)
        elif name == 'LightGBM':
            model = LGBMRegressor(random_state=2024, verbosity=-1)

    return model



def evaluate_model(X, Y, round_num=3, classification=False, preds_probs=None):
    """
    Evaluate the model performance.
    return: acc, precision, recall, f1, auc, mcc for classification,
            MAE, RMSE, R, MSE, R2 for regression.
    """
    if len(X) < 2:
        return 'nan', 'nan'
    else:
        if classification:
            acc = np.round(accuracy_score(X, Y), round_num)
            precision = np.round(precision_score(X, Y), round_num)
            recall = np.round(recall_score(X, Y), round_num)
            f1 = np.round(f1_score(X, Y), round_num)
            auc = np.round(roc_auc_score(X, preds_probs), round_num)
            mcc = np.round(matthews_corrcoef(X, Y), round_num)
            return acc, precision, recall, f1, auc, mcc
        else:
            MAE = np.round(mean_absolute_error(X, Y), round_num)
            RMSE = np.round(np.sqrt(mean_squared_error(X, Y)), round_num)
            R = np.round(pearsonr(X, Y)[0], round_num)
            MSE = np.round(mean_squared_error(X, Y), round_num)
            R2 = np.round(r2_score(X, Y), round_num)
            return MAE, RMSE, R, MSE, R2



def shuffle_array(arr, seed=None):
    """
    Shuffle the array until all elements are different.
    """
    if seed is not None:
        np.random.seed(seed)

    arr_shuffled = arr.copy()
    while True:
        np.random.shuffle(arr_shuffled)
        if not np.any(arr == arr_shuffled):
            break
    return arr_shuffled
