# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 17:50:50 2025

@author: Fahim
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb

# Ensemble model for Redemption forecast

class EnsembleModelRedemption:
    def __init__(self):
        self.models = {

            'XGB':XGBRegressor(learning_rate=0.1, max_depth=6, n_estimators=300, n_jobs=-1, random_state=42,verbose=-1),
            
            'LGBM':lgb.LGBMRegressor(n_estimators=300,learning_rate=0.03,max_depth=5,subsample=0.8,
                          colsample_bytree=0.8,n_jobs=-1,random_state=42,verbose=-1),
            
            'DecisionTree': DecisionTreeRegressor(max_depth=6,min_samples_split=7,min_samples_leaf=5,
                                                 random_state=42),
        
            
            'RandomForest': RandomForestRegressor(n_estimators=300,max_depth=6,min_samples_split=10,min_samples_leaf=5) 
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
    

