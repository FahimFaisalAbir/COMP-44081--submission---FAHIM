# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 19:37:57 2025

@author: Fahim
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
            

## Sales model for Redemption forecast

class EnsembleModelSales:
    def __init__(self):
        self.models = {

            'XGB':XGBRegressor(objective='reg:absoluteerror',booster='gbtree',
                           learning_rate=0.1, max_depth=5, n_estimators=600, n_jobs=-1, 
                           reg_alpha=0.4,reg_lambda=6,subsample=0.8,random_state=42,verbose=-1),

            
            'LGBM':lgb.LGBMRegressor(n_estimators=700,learning_rate=0.05,max_depth=5,
                                  subsample=0.8,min_child_samples=7, num_leaves=31, reg_alpha=0,
                                  n_jobs=-1,random_state=42,verbose=-1),
            
            'DecisionTree': DecisionTreeRegressor(max_depth=6,min_samples_split=7,min_samples_leaf=5,
                                                 random_state=42),
        
            
            'RandomForest': RandomForestRegressor(n_estimators=500,max_depth=6,min_samples_split=2,min_samples_leaf=7,n_jobs=-1)
        }

    def fit(self, X_train, y_train):
        # train multiple models
        for name, model in self.models.items():
            model.fit(X_train, y_train)

    def predict(self, X_test):
        # predict
        if X_test.isnull().any().any():
            X_test = X_test.fillna(0)
            
        preds = pd.DataFrame()
        for name, model in self.models.items():
            preds[name] = model.predict(X_test)
            
        # Fill any NaNs if needed 
        preds_filled = preds.copy()
        preds_filled = preds_filled.apply(lambda row: row.fillna(row.mean()), axis=1)
 
        preds_filled = preds_filled.fillna(0)
        preds_filled['Ensemble'] = preds_filled.mean(axis=1)

        return preds_filled['Ensemble'].values


    def get_feature_importances(self):
        importances = {}
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances[name] = model.feature_importances_
        return importances
