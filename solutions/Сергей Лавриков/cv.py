from __future__ import unicode_literals
import gc

from sklearn.metrics import roc_auc_score
from skopt import BayesSearchCV
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime
from copy import deepcopy
from scipy.stats import rankdata

from adversial_validation import adversial_train_test_split
from data_loading import load_csi_test, load_csi_train, load_features, CACHE_DIR, load_consumption
from data_prepare import features, session_kpi, merge_all, as_category, main_cell_kpi
from transformers.pandas_subset import PandasSubset

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

target = 'CSI'

search_spaces = {'subset__' + f: [True] for f in features}
search_spaces.update({
    'estimator__num_leaves': [17],
    'estimator__max_depth': [-1],
    'estimator__min_child_samples': [12],#[16],#[56],
    'estimator__max_bin': [10],
    'estimator__subsample': [0.516],
    'estimator__subsample_freq': [1],
    'estimator__colsample_bytree': [0.9],
    'estimator__min_child_weight': [82],
    'estimator__subsample_for_bin': [200000],
    'estimator__min_split_gain': [0.9],
    'estimator__reg_alpha': [0.1],
    'estimator__reg_lambda': [0.99],
})


def status_print(optim_result):
    print(f'Best ROC-AUC: {np.round(bayes_cv_tuner.best_score_, 4),}, '
          f'current={np.round(bayes_cv_tuner.cv_results_["mean_test_score"][-1], 4)}, '
          f'std={np.round(bayes_cv_tuner.cv_results_["std_test_score"][-1], 4)}')


if __name__ == '__main__':

    if os.path.isfile(os.path.join(CACHE_DIR, 'train_df.feather')):
        train_df = pd.read_feather(os.path.join(CACHE_DIR, 'train_df.feather'))

        train_df[train_df.select_dtypes(include=[np.float32]).columns] = \
            train_df[train_df.select_dtypes(include=[np.float32]).columns].astype(np.float16)
        gc.collect()
    else:
        train_df = load_csi_train()
        train_feat_df = load_features('train')
        train_consumtion_df = load_consumption('train')
        train_data_avg_df = session_kpi(None, None, None, 'train_data_avg')
        train_data_chnn_df = session_kpi(None, None, None, 'train_data_chnn')
        train_voice_avg_df = session_kpi(None, None, None, 'train_voice_avg')
        train_voice_chnn_df = session_kpi(None, None, None, 'train_voice_chnn')
        gc.collect()

        train_main_cell_avg_df = main_cell_kpi(None, None, None, 'train_main_avg_kpi')
        train_main_cell_chnn_df = main_cell_kpi(None, None, None, 'train_main_chnn_kpi')
        gc.collect()

        train_df = merge_all(train_df,
                             train_feat_df,
                             train_consumtion_df,
                             train_data_avg_df, train_data_chnn_df, train_voice_avg_df, train_voice_chnn_df,
                             train_main_cell_avg_df, train_main_cell_chnn_df)

        train_df[train_df.select_dtypes(include=[np.float16]).columns] = \
            train_df[train_df.select_dtypes(include=[np.float16]).columns].astype(np.float32)
        train_df[train_df.select_dtypes(include=['datetime64']).columns] = \
            train_df[train_df.select_dtypes(include=['datetime64']).columns].astype(str)

        train_df.to_feather(os.path.join(CACHE_DIR, 'train_df.feather'))

        train_df[train_df.select_dtypes(include=[np.float32]).columns] = \
            train_df[train_df.select_dtypes(include=[np.float32]).columns].astype(np.float16)
        gc.collect()

    train_df = as_category(train_df)

    train_y = train_df['CSI']
    train_X = train_df.drop(['CSI', 'CONTACT_DATE', 'SNAP_DATE'], axis=1)
    gc.collect()


    class FeaturePredictor(BaseEstimator):
        def __init__(self, **params):
            self.pipeline = Pipeline([
                ('subset', PandasSubset(**{k: True for k in features})),
                ('estimator', LGBMClassifier(objective='binary',
                                             learning_rate=0.01,
                                             num_leaves=7,
                                             max_depth=-1,
                                             min_child_samples=100,
                                             max_bin=105,
                                             subsample=0.7,
                                             subsample_freq=1,
                                             colsample_bytree=0.8,
                                             min_child_weight=0,
                                             subsample_for_bin=200000,
                                             min_split_gain=0,
                                             reg_alpha=0,
                                             reg_lambda=0,
                                             n_estimators=500,
                                             n_jobs=4,
                                             is_unbalance=True,
                                             # random_state=42,
                                             class_weight='balanced',
                                             silent=True,
                                             verbose=-1,
                                             metric='auc'
                                             )),
            ])
            self.set_params(**params)
            self.models = []

        def fit(self, X, y):
            Xs = self.pipeline.named_steps['subset'].fit_transform(X)

            self.models = []
            for train_ix, val_ix in RepeatedStratifiedKFold(5, n_repeats=10).split(Xs, y):
                model = deepcopy(self.pipeline.named_steps['estimator'].fit(Xs.iloc[train_ix], y.iloc[train_ix],
                                                                            eval_metric="auc",
                                                                            eval_set=(Xs.iloc[val_ix], y.iloc[val_ix]),
                                                                            early_stopping_rounds=100,
                                                                            verbose=False,
                                                                            ))
                self.models.append(model)

            del Xs
            gc.collect()
            return self

        def predict(self, X):
            Xs = self.pipeline.named_steps['subset'].transform(X)
            y_pred = []
            for m in self.models:
                y_pred.append(rankdata(m.predict_proba(Xs)[:, 1]) / Xs.shape[0])
            pred_arr = np.array(y_pred)
            return np.mean(pred_arr, axis=0)

        def predict_proba(self, X):
            pred_arr = self.predict(X).reshape(-1, 1)
            pred_arr = np.hstack([1.0-pred_arr, pred_arr])
            return pred_arr

        def get_params(self, deep=True):
            return self.pipeline.get_params(deep)

        def set_params(self, **params):
            self.pipeline.set_params(**params)
            return self

        def feature_importances(self):
            arr = np.array([m.feature_importances_ for m in self.models])
            arr = MinMaxScaler().fit_transform (arr.T).T

            return arr.mean(axis=0), np.std(arr, axis=0)


    print("Training...")

    bayes_cv_tuner = BayesSearchCV(
        estimator=FeaturePredictor(),
        search_spaces=search_spaces,
        scoring='roc_auc',
        cv=RepeatedStratifiedKFold(5, 4),
        n_jobs=1,
        pre_dispatch=4,
        n_iter=1,
        verbose=0,
        refit=True,
        random_state=42,
    )

    best_estimator = bayes_cv_tuner.estimator
    clf_name = best_estimator.__class__.__name__

    if clf_name != 'FeaturePredictor':
        cols = list(set(train_X.columns).difference(train_X.select_dtypes(include='category').columns))
        train_X.loc[:, cols] = train_X.loc[:, cols].fillna(0).replace([np.inf, -np.inf], 0)
        for c in train_X.select_dtypes(include='category').columns:
            train_X.loc[:, c] = train_X.loc[:, c].cat.codes

    result = bayes_cv_tuner.fit(train_X, train_y, callback=status_print)

    best_estimator = bayes_cv_tuner.best_estimator_

    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)
    all_models.to_csv(f"cv_results/{clf_name}_cv_{datetime.now().strftime('%d_%H_%M')}.csv")

    if clf_name=='FeaturePredictor' or clf_name=='RandomForestClassifier':
        if clf_name == 'FeaturePredictor':
            importances, std = best_estimator.feature_importances()
        if clf_name == 'RandomForestClassifier':
            importances = best_estimator.feature_importances_
            std = np.std([tree.feature_importances_ for tree in best_estimator.estimators_],
                         axis=0)

        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")

        for ix, imp in enumerate(importances):
            if imp > 0 and ix < len(features):
                print(f"'{features[ix]}',")

        for f in range(len(importances)):
            if indices[f] < len(features):
                print(f"{f+1}. feature {features[indices[f]]} ({importances[indices[f]]}, {std[indices[f]]})")


    if os.path.isfile(os.path.join(CACHE_DIR, 'test_df.feather')):
        test_df = pd.read_feather(os.path.join(CACHE_DIR, 'test_df.feather'))

        test_df[test_df.select_dtypes(include=[np.float32]).columns] = \
            test_df[test_df.select_dtypes(include=[np.float32]).columns].astype(np.float16)
        gc.collect()
    else:
        test_df = load_csi_test()
        test_feat_df = load_features('test')
        train_consumtion_df = load_consumption('test')
        test_data_avg_df = session_kpi(None, None, None, 'test_data_avg')
        test_data_chnn_df = session_kpi(None, None, None, 'test_data_chnn')
        test_voice_avg_df = session_kpi(None, None, None, 'test_voice_avg')
        test_voice_chnn_df = session_kpi(None, None, None, 'test_voice_chnn')
        gc.collect()

        test_main_cell_avg_df = main_cell_kpi(None, None, None, 'test_main_avg_kpi')
        test_main_cell_chnn_df = main_cell_kpi(None, None, None, 'test_main_chnn_kpi')
        gc.collect()

        test_df = merge_all(test_df,
                            test_feat_df,
                            train_consumtion_df,
                            test_data_avg_df, test_data_chnn_df, test_voice_avg_df, test_voice_chnn_df,
                            test_main_cell_avg_df, test_main_cell_chnn_df)

        test_df[test_df.select_dtypes(include=[np.float16]).columns] = \
            test_df[test_df.select_dtypes(include=[np.float16]).columns].astype(np.float32)
        test_df[test_df.select_dtypes(include=['datetime64']).columns] = \
            test_df[test_df.select_dtypes(include=['datetime64']).columns].astype(str)

        test_df.to_feather(os.path.join(CACHE_DIR, 'test_df.feather'))

        test_df[test_df.select_dtypes(include=[np.float32]).columns] = \
            test_df[test_df.select_dtypes(include=[np.float32]).columns].astype(np.float16)
        gc.collect()

    test_df = as_category(test_df)

    test_X = test_df.drop(['CONTACT_DATE', 'SNAP_DATE'], axis=1)

    if clf_name != 'FeaturePredictor':
        cols = list(set(test_X.columns).difference(test_X.select_dtypes(include='category').columns))
        test_X.loc[:, cols] = test_X.loc[:, cols].fillna(0).replace([np.inf, -np.inf], 0)
        for c in test_X.select_dtypes(include='category').columns:
            test_X.loc[:, c] = test_X.loc[:, c].cat.codes

    adv_auc = 0
    adv_train_x, adv_train_y, adv_test_x, adv_test_y = adversial_train_test_split(train_X.loc[:, features], train_y,
                                                                                  test_X.loc[:, features],
                                                                                  topK=1000)
    bayes_cv_tuner._fit_best_model(adv_train_x, adv_train_y)
    adv_pred_y = bayes_cv_tuner.predict_proba(adv_test_x)[:, 1]
    adv_auc = roc_auc_score(adv_test_y, adv_pred_y)
    print(f'Adversial AUC = {adv_auc} by {len(adv_test_y)} samples')

    bayes_cv_tuner._fit_best_model(train_X, train_y)
    test_y = bayes_cv_tuner.predict_proba(test_X)
    df = pd.DataFrame(test_y[:, 1])
    df.to_csv(f"submits/"
              f"{best_estimator.__class__.__name__}"
              f"_{datetime.now().strftime('%d_%H_%M')}"
              f"_{bayes_cv_tuner.best_score_:0.4f}"
              f"_{adv_auc:0.4f}.csv",
              header=None,
              index=None)
