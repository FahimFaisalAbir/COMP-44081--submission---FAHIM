from EnsembleModelRedemption import EnsembleModelRedemption

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

#metric
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import mean_absolute_error,mean_squared_error
from itertools import product    

#pre-processing tools
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit,GridSearchCV

# import models
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from pmdarima import auto_arima    
from prophet import Prophet
from prophet.make_holidays import make_holidays_df
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# season converter 
def month_to_season(month):
    if month in [3, 4, 5]:
        return 1
    elif month in [6, 7, 8]:
        return 2
    elif month in [9, 10, 11]:
        return 3
    elif month in [12, 1, 2]:
        return 4
    else:
        return 0
    
    
# Redemption Model class
class RedemptionModel:

    def __init__(self, X, target_col):
        '''
        Args:
        X (pandas.DataFrame): Dataset of predictors, output from load_data()
        target_col (str): column name for target variable
        '''
            
        self._predictions = {}
        self.X = X
        self.target_col = target_col
        self.results = {}
         # dict of dicts with model results

    
    def score(self, truth, preds):
        
        # Score our predictions - modify this method as you like
        mask = truth != 0
        #get rid of null values so filter out zero y_true values
        return MAPE(truth[mask], preds[mask])
        #return mean_absolute_error(truth, preds)

    def average_predictions_by_doy(self, preds: pd.Series) -> pd.Series:
        # Return average prediction per day of yr across years
        # Map predictions to day of yr
        
        result = preds.to_frame('pred')
        result['doy'] = result.index.dayofyear

        # Group by day of yr and take mean for all the same day
        typical_year = result.groupby('doy')['pred'].mean()

        # Expand back to full index 
        full_preds = preds.index.to_series().map(lambda ts: typical_year[ts.dayofyear])
        return pd.Series(full_preds.values, index=preds.index)

    
    def run_models(self, n_splits=3, test_size=365):
        '''Run the models and store results for cross validated splits in
        self.results.
        '''
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        cnt = 0
        # Time series split
#         figs,axss=plt.subplots(n_splits,1,figsize=(15,15),sharex=True)
#         X_train[self.target_col].plot(ax=axss[cnt],label='Train sets', title=f'Data train/test split {cnt}')
#         X_test[self.target_col].plot(ax=axss[cnt],label='Test sets')
#         axss[cnt].axvline(X_test.index.min(), color='black', ls='--')
#         plt.show()

        for train_idx, test_idx in tscv.split(self.X):
            #train
            X_train = self.X.iloc[train_idx]
            X_train=X_train.copy()
            
            #encoding categorical season feature and adding it to train data
            le=LabelEncoder()
            X_train['season']=X_train.index.month.map(month_to_season)
            X_train['season'] = le.fit_transform(X_train['season'])
            
            #test
            X_test = self.X.iloc[test_idx]
            X_test = X_test.copy()
            
            #encoding categorical season feature and adding it to test data
            X_test['season']=X_test.index.month.map(month_to_season)
            X_test['season'] = le.transform(X_test['season'])
            print('val shape', X_train.shape,X_test.shape)
            
            # LSTM Model tried here
            

            # Base model provided 
            preds_base = self._base_model(X_train, X_test)
            # store results 
            self._store_results('Base', cnt, X_test[self.target_col], preds_base)
            # print mean squared error beside MAPE score
            print(mean_squared_error(X_test[self.target_col], preds_base))
            #print mape score(true,preds)
            print('Base',self.score(X_test[self.target_col], preds_base))
            # plot score 
            self.plot(preds_base, X_test[self.target_col], 'Base')
            

            # ARIMA
            #arima to detect trends, seasonality 
            preds_arima,conf_int_arima=self._auto_arima_model(X_train, X_test)
            self._store_results('ARIMA', cnt, X_test[self.target_col], preds_arima)
            print(mean_squared_error(X_test[self.target_col], preds_arima))
            print('ARIMA',self.score(X_test[self.target_col],preds_arima, ))
            self.plot(preds_arima, X_test[self.target_col], 'ARIMA',conf_int_arima)
            
            # ETS
            #Exponential Smoothing for data with trend
            preds_ets=self._ets_model(X_train, X_test)
            self._store_results('ETS', cnt, X_test[self.target_col], preds_ets)
            print(mean_squared_error(X_test[self.target_col], preds_ets))
            print('ETS',self.score(X_test[self.target_col],preds_ets))
            self.plot(preds_ets, X_test[self.target_col], 'ETS')
            
            #seasonal stat model with grid search with week, month,3 month, year seasonality
            # Grid search parameters
            seasonal_options = [7, 30, 120,365]
            window_options = [ 7, 14 , 21]

            preds_seasonal, conf_int_seasonal, best_params, score = self._seasonal_smooth_grid_search(
                X_train, X_test, seasonal_options, window_options
            )
            self._store_results('seasonal smoothing', cnt, X_test[self.target_col], preds_seasonal)
            print(mean_squared_error(X_test[self.target_col], preds_seasonal))
            print('seasonal smoothing',self.score( X_test[self.target_col],preds_seasonal,))
            self.plot(preds_seasonal, X_test[self.target_col], 'Seasonal smoothing', conf_int_seasonal)
            
            
            #Prophet model to include seasonality and holiday effect
            # use multiplicative mode to capture weekly, monthly, yearly seasonality
            preds_prophet, conf_prophet = self._prophet_seasonal_holiday_model(X_train, X_test)
            preds_typical = self.average_predictions_by_doy(preds_prophet)
            self._store_results('Prophet Season Holiday', cnt, X_test[self.target_col], preds_typical)
            print(mean_squared_error(X_test[self.target_col], preds_prophet))
            print('Prophet Season Holiday',self.score( X_test[self.target_col],preds_prophet))
            self.plot(preds_typical, X_test[self.target_col], 'Prophet Season Holiday', conf_prophet)
            
            
            #Random forest - ensemble of trees to come up with generalized prediction   

            preds_rf=self._rf_model(X_train, X_test)
            self._store_results('RF', cnt, X_test[self.target_col],  preds_rf)
            print(mean_squared_error(X_test[self.target_col],  preds_rf))
            print('RF',self.score(X_test[self.target_col],  preds_rf))
            self.plot( preds_rf, X_test[self.target_col], 'RF')
            
            #XGBoost to include tree based boosting models 
            preds_xgb =self._xgboost_model(X_train, X_test)
            self._store_results('XGB', cnt, X_test[self.target_col], preds_xgb)
            print(mean_squared_error(X_test[self.target_col], preds_xgb))
            print('XGB',self.score(X_test[self.target_col],preds_xgb))    
            self.plot(preds_xgb, X_test[self.target_col], 'XGB')
            
            #Light GBM to come up with efficient inference 
            preds_lgbm=self._lightgbm_model(X_train, X_test)
            self._store_results('LGBM', cnt, X_test[self.target_col], preds_lgbm)
            print(mean_squared_error(X_test[self.target_col], preds_lgbm))
            print('LGBM',self.score(X_test[self.target_col], preds_lgbm))
            self.plot(preds_lgbm, X_test[self.target_col], 'LGBM')
            
            
            # Ensemble model of decision tree, random forest , XGb and LGBM
            preds_ens = self._ensemble_model(X_train, X_test)
            self._store_results('Ensemble model', cnt, X_test[self.target_col], preds_ens)
            print(mean_squared_error(X_test[self.target_col], preds_ens))
            print('Ensemble model',self.score(X_test[self.target_col], preds_ens))
            self.plot(preds_ens, X_test[self.target_col], 'Ensemble model')
            
            
            cnt += 1
        
    def _store_results(self, model_name, split_num, truth, preds):
        # Store results
        if model_name not in self.results:
            self.results[model_name] = {}
        self.results[model_name][split_num] = self.score(truth, preds)

    def _base_model(self, train, test):
        res = sm.tsa.seasonal_decompose(train[self.target_col], period=365)
        res_clip = res.seasonal.apply(lambda x: max(0, x))
        res_clip.index = res_clip.index.dayofyear
        res_clip = res_clip.groupby(res_clip.index).mean()
        res_dict = res_clip.to_dict()
        return pd.Series(index=test.index,
                         data=[res_dict[doy] for doy in test.index.dayofyear])
    
    def _auto_arima_model(self, train, test, seasonal_period=365):
        y_train = train[self.target_col]

        # Fit Auto-ARIMA model
#         model = auto_arima(
#             y_train,
#             seasonal=True,
#             m=seasonal_period,           # 365 for annual seasonality
#             stepwise=True,
#             suppress_warnings=True,
#             error_action='ignore',
#             trace=False
#         )
        # use ARIMA to find out optimal p,d,q value 
        model = pm.auto_arima(y_train, stepwise=False,  seasonal=True)

        # Forecast using best ARIMA model
        forecast, conf_int = model.predict(n_periods=len(test), return_conf_int=True)
        forecast_index = test.index
        
        #conf interval to see how it handles uncertainity 
        forecast_series = pd.Series(forecast, index=forecast_index)
        conf_df = pd.DataFrame(
            conf_int, 
            columns=['lower', 'upper'], 
            index=forecast_index
        )

        return forecast_series, conf_df
    
    #Exponential Smoothing for data with trend
    def _ets_model(self, train, test):
        y_train = train[self.target_col]

        # Fit Holt-Winters Exponential Smoothing
        model = ExponentialSmoothing(y_train,
                                      seasonal='add',
                                      seasonal_periods=365,
                                      trend='add',
                                      initialization_method='estimated')
        model_fit = model.fit()

        # Forecast the test period
        forecast = model_fit.forecast(steps=len(test))
        forecast.index = test.index

        # ETS does not give uncertainty 
        return forecast
    
    def _seasonal_smooth_grid_search(self, train, val, seasonal_periods_list, window_sizes):
        best_score = float('inf')
        best_params = None
        best_preds = None
        best_conf = None

        for season_len, win in product(seasonal_periods_list, window_sizes):
            try:
                res = sm.tsa.seasonal_decompose(train[self.target_col], period=season_len)
                seasonal = res.seasonal.groupby(res.seasonal.index.dayofyear).mean()
                doy = val.index.dayofyear
                preds = doy.map(seasonal).values
                #smooth prediction within different window size (week/month)
                preds_smoothed = pd.Series(preds, index=val.index).rolling(win, center=True, min_periods=1).mean()

                # Residuals used with bootstrap 
                residuals = train[self.target_col] - train.index.dayofyear.map(seasonal)
                bootstraps = [preds_smoothed + np.random.choice(residuals, size=len(preds_smoothed), replace=True)
                              for _ in range(100)]
                lower = np.percentile(bootstraps, 2.5, axis=0)
                upper = np.percentile(bootstraps, 97.5, axis=0)
                conf_int = pd.DataFrame({'lower': lower, 'upper': upper}, index=val.index)

                score = self.score(val[self.target_col], preds_smoothed)

                if score < best_score:
                    best_score = score
                    best_params = (season_len, win)
                    best_preds = preds_smoothed
                    best_conf = conf_int
            except Exception as e:
                # Ignore any failure for problematic grid value combination
                continue
        
        return best_preds, best_conf, best_params, best_score


    

    def _prophet_seasonal_holiday_model(self, train, test, country='CA'):
        # prepare dataframe for model
        train_df = train.reset_index().rename(columns={'Timestamp': 'ds', self.target_col: 'y'})

        # consider candian holidays
        holidays_df = make_holidays_df(year_list=train_df['ds'].dt.year.unique(), country=country)

        # Prophet with holiday effects

        #Multiplicative used to capture seasonal effects with trend
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            holidays=holidays_df,
            seasonality_mode='multiplicative'
        )
        # Train/ fit Prophet model
        model.fit(train_df)

        # Forecast test period len using prophet 
        future = model.make_future_dataframe(periods=len(test), freq='D')
        forecast = model.predict(future)

        # get forecast values and reset index
        forecast_test = forecast.set_index('ds').loc[test.index]
        
        # prophet's day pred columns is yhat so fetch yhat as forecast
        preds = forecast_test['yhat']
        conf_int = forecast_test[['yhat_lower', 'yhat_upper']]
        
        # collect confidence interval

        conf_int = conf_int.rename(columns={'yhat_lower': 'lower', 'yhat_upper': 'upper'})

        return preds, conf_int



    def _xgboost_model(self, train, test):
        # prepare dataframe for xgb 
        df = pd.concat([train, test])


        # Add seasonal trend from data
        # feature eng
        seasonal = train.groupby(train.index.dayofyear)[self.target_col].mean().to_dict()
        df['seasonal_base'] = df.index.dayofyear.map(seasonal)

        # Add lags and it should be less than forecasting horizon
        df['lag7'] = df[self.target_col].shift(7)
        df['lag14'] = df[self.target_col].shift(14)
        df['lag28'] = df[self.target_col].shift(28)
        df['lag56'] = df[self.target_col].shift(56)
        df['lag_year'] = df[self.target_col].shift(356)
        df['Sales_lag1'] = df['Sales Count'].shift(1)
 
        # Drop rows with NA as some intial lags might be 0 
        df.dropna(inplace=True)
        #print(df.head())
        

        features = ['Sales_lag1','dayofweek', 'dayofyear', 'monthly', 'season','weekofyear', 
                    'seasonal_base', 'lag7', 'lag14', 'lag28','lag56','lag_year']
        
        # new train  test set for XGB tree
        X_train = df.loc[train.index.intersection(df.index), features]
        print(X_train.columns)
        y_train = df.loc[train.index.intersection(df.index), self.target_col]
        X_test = df.loc[test.index.intersection(df.index), features]
        
        #print to debug
        #rint(X_train.head(),'\n Train \n',y_train.head(),'---\n Test \n',X_test.head())
        #print(X_train.columns,'\n Train',y_train,'---\n Test',X_test.columns)
        
        # GRID search - RUN once 

        #         model = XGBRegressor(n_estimators=100,
        #                              max_depth=3,
        #                              learning_rate=0.05,
        #                              objective='reg:squarederror',
        #                              subsample=0.8,
        #                              colsample_bytree=0.8,
        #                              random_state=42)

        #         param_grid_xgb = {
        #             'n_estimators': [30,50,100, 200,300],
        #             'max_depth': [3, 5, 6],
        #             'learning_rate': [0.0001,0.001,0.01,0.05, 0.1],
        #             'subsample':[0.8],
        #         #'colsample_bytree':[0.8],
        #         #'gamma':[1,2],
        #         'reg_alpha':[0.4,0.5],
        #         'reg_lambda':[5,6],
        #         }

        #         tscv = TimeSeriesSplit(n_splits=4)
        #         model = XGBRegressor(objective='reg:squarederror', random_state=42)
        #         grid_search = GridSearchCV(model, param_grid_xgb, cv=tscv, scoring='neg_mean_absolute_percentage_error')

        #         grid_search.fit(X_train, y_train)
        #         model=grid_search.best_estimator_
        
        # train best XGB model based on grid search
        #estimator=300
        
        model=XGBRegressor(objective='reg:absoluteerror',booster='gbtree',
                           learning_rate=0.1, max_depth=6, n_estimators=700, n_jobs=-1, 
                           random_state=42,verbose=-1)
        #print(model)
        # train XGB
        model.fit(X_train, y_train)
        # make prediction
        preds = pd.Series(model.predict(X_test), index=X_test.index)

        #print(preds.shape, y_train.shape)
    
        # show feature importance for XGB
        feat_imp = pd.DataFrame(data=model.feature_importances_,
        index=model.feature_names_in_,
        columns=['feature importance'])
        feat_imp.sort_values('feature importance').plot(kind='barh', title='XGB Feature Importance plot')
        plt.show()
        
        #return self.average_predictions_by_doy(preds)
        return preds

        


    def _lightgbm_model(self, train, test):
        df = pd.concat([train, test])
#         df['dayofweek'] = df.index.dayofweek
#         df['dayofyear'] = df.index.dayofyear
#         df['month'] = df.index.month
#         df['weekofyear'] = df.index.isocalendar().week.astype(int)

        # seasonal pattern from training (like base model)

        
        seasonal = train.groupby(train.index.dayofyear)[self.target_col].mean().to_dict()
        df['seasonal_base'] = df.index.dayofyear.map(seasonal)

        # Lag features
        df['lag7'] = df[self.target_col].shift(7)
        df['lag3'] = df[self.target_col].shift(3)
        df['lag28'] = df[self.target_col].shift(30)
        df['Sales_lag1'] = df['Sales Count'].shift(1)


        # Drop NA rows due to lag
        df.dropna(inplace=True)
        

        features = ['Sales_lag1','dayofweek', 'dayofyear', 'monthly', 'season','weekofyear',
                    'seasonal_base','lag3', 'lag7', 'lag28']
        X_train = df.loc[train.index.intersection(df.index), features]
        y_train = df.loc[train.index.intersection(df.index), self.target_col]
        X_test = df.loc[test.index.intersection(df.index), features]
        print(X_train.columns)
        model = lgb.LGBMRegressor(n_estimators=200,learning_rate=0.03,max_depth=5,subsample=0.8,colsample_bytree=0.8,n_jobs=-1,random_state=42,verbose=-1)
                        
        
        model.fit(X_train, y_train)
        

        preds = pd.Series(model.predict(X_test), index=X_test.index)
        
        return preds



    def _rf_model(self, train, test):
        df = pd.concat([train, test]).copy()

        # Calendar features
#         df['dayofweek'] = df.index.dayofweek
#         df['dayofyear'] = df.index.dayofyear
#         df['month'] = df.index.month
#         df['weekofyear'] = df.index.isocalendar().week.astype(int)

        # Season from month
        #df['season'] = df.index.month.map(month_to_season)
        #df['season_code'] = LabelEncoder().fit_transform(df['season'])

        # Observed seasonal pattern from training
        seasonal = train.groupby(train.index.dayofyear)[self.target_col].mean().to_dict()
        df['seasonal_base'] = df.index.dayofyear.map(seasonal)

        # Lag features
        df['lag7'] = df[self.target_col].shift(7)
        df['lag28'] = df[self.target_col].shift(28)
        df['Sales_lag1'] = df['Sales Count'].shift(1)

        # Drop NA due to lag
        df.dropna(inplace=True)
        #print(df)
        features = ['Sales_lag1','dayofweek', 'dayofyear', 'monthly', 'season',
                    'weekofyear','seasonal_base', 'lag7', 'lag28']

        X_train = df.loc[train.index.intersection(df.index), features]
        y_train = df.loc[train.index.intersection(df.index), self.target_col]
        X_test = df.loc[test.index.intersection(df.index), features]

        # Random Forest
        model = RandomForestRegressor(n_estimators=200,max_depth=6,min_samples_split=10,min_samples_leaf=5)               
        # regularization: limit depth
        # regularization: minimum samples to split
        # regularization: minimum samples at l
        
        model.fit(X_train, y_train)
        
        #feat imp
        feat_imp = pd.DataFrame(data=model.feature_importances_,
        index=model.feature_names_in_,
        columns=['feature importance'])
        feat_imp.sort_values('feature importance').plot(kind='barh', title='RF Feature Importance')
        
        plt.show()
        preds = pd.Series(model.predict(X_test), index=X_test.index)

        return preds
    
    def _ensemble_model(self, train, test):
        df = pd.concat([train, test])



        # Add seasonal trend from data
        seasonal = train.groupby(train.index.dayofyear)[self.target_col].mean().to_dict()
        df['seasonal_base'] = df.index.dayofyear.map(seasonal)
        
        # Add lags 
        df['lag3'] = df[self.target_col].shift(3)
        df['lag7'] = df[self.target_col].shift(7)
        df['lag30'] = df[self.target_col].shift(30)
        df['lag14'] = df[self.target_col].shift(14)
        df['lag28'] = df[self.target_col].shift(28)
        df['lag56'] = df[self.target_col].shift(56)
        df['Sales_lag1'] = df['Sales Count'].shift(1)

        # Drop rows with NA as some intial lags might be 0 
        df.dropna(inplace=True)
        

        features = ['Sales_lag1','dayofweek', 'dayofyear', 'monthly', 'season','weekofyear', 
                    'seasonal_base','lag3', 'lag7', 'lag14', 'lag28','lag30','lag56']

        X_train = df.loc[train.index.intersection(df.index), features]
        y_train = df.loc[train.index.intersection(df.index), self.target_col]
        X_test = df.loc[test.index.intersection(df.index), features]

        print(X_train.columns)
        model = EnsembleModelRedemption()
        model.fit(X_train, y_train)
        #print(X_test.info())
        preds= model.predict(X_test)

        preds = pd.Series(preds, index=X_test.index)


        importances = model.get_feature_importances() 
        feature_names =['Sales_lag1','dayofweek', 'dayofyear', 'monthly', 'season','weekofyear', 
                    'seasonal_base','lag3', 'lag7', 'lag14', 'lag28','lag30','lag56']
        
       
        # feature imortance plot
        # Stack all importances into an array (models x features)
        importance_matrix = np.vstack([
            importances[m] for m in importances if m in importances
        ])
        
        average_importance = np.mean(importance_matrix, axis=0)
        std_importance = np.std(importance_matrix, axis=0)
        
        sorted_idx = np.argsort(average_importance)[::-1]
        plt.figure(figsize=(8,5))
        plt.bar(
            np.array(feature_names)[sorted_idx],
            average_importance[sorted_idx],
            yerr=std_importance[sorted_idx],
            alpha=0.8,
            color="skyblue"
        )
        plt.title("Average Feature Importance of Ensemble model")
        plt.ylabel("Mean Importance with SD")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()



        
        #print(preds.shape, y_train.shape)


        return preds


    def plot(self, preds, truth, label, conf_int=None):
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(truth.index, truth.values, label='Observed', color='grey')
        ax.plot(preds.index, preds.values, label=f'Forecast: {label}', color='blue')
        if conf_int is not None:
            ax.fill_between(preds.index, 
                            conf_int.iloc[:, 0], 
                            conf_int.iloc[:, 1], 
                            color='blue', alpha=0.2, label='95% CI')
        plt.title(f"{label} Forecast vs Actual")
        plt.legend()
        plt.show()
