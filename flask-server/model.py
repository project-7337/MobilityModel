#!flask/bin/python
from json import loads, dumps
import pandas as pd
import numpy as np
import datetime
import warnings
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from flask import Flask
app = Flask(__name__)

def sarimax(train,exog_train,p,d,q,P,D,Q,M,summary=True):
    arima = sm.tsa.statespace.SARIMAX(train,exog=exog_train,order=(p,d,q),seasonal_order=(P,D,Q,M),enforce_stationarity=False, enforce_invertibility=False).fit()
    if summary==True:
        arima.summary()
    return(arima)

def fcast_sarimax(arima,exog_forecast,n):
    #forecasts
    pred_uc = arima.get_forecast(steps=n,exog=exog_forecast)
    forecast = pred_uc.predicted_mean
    return(forecast.tail(n))

def get_sarimax(trend,exog_train,exog_test,max_range=3,seasonal_factor=12):
    #sarima modelling on trend with period = 12
    train,test = data_creation_sarima(trend,0.9)
    param_order = []
    param_seas = []
    p = d = q = range(0, max_range)
    pdq = list(itertools.product(p, d, q))

    #seasonal parameter = seasonal_factor
    seasonal_pdq = [(x[0], x[1], x[2], seasonal_factor) for x in list(itertools.product(p, d, q))]
#     print(pdq,seasonal_pdq)
    
    min_rmse=9999999
    min_model_mape = 9999999
    min_model_aic = 9999999
    final_predictions=-1 
    result_param=-1 
    result_param_seasonal=-1
    for i in [(1,1,1)]:#range(0,len(pdq)):
        for j in [(1,0,0,12)]:#range(0,len(seasonal_pdq)):
#             try:

            param = pdq[i]
            param_seasonal = seasonal_pdq[j]
            mod = sm.tsa.statespace.SARIMAX(train,exog=exog_train,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
            results = mod.fit()
            aic_value = results.aic
#                 if math.isnan(temp) or math.isinf(temp):
#                     temp = 999.99
            pred = results.predict(len(train),len(trend)-1,exog = exog_test)
            rmse = scoring_rmse(pred,test)
            mape = scoring_mape(pred,test)
#                 print(pred,rmse,mape)
            if rmse < min_rmse:
                min_rmse = rmse
                min_model_aic = aic_value
                min_model_mape = mape
                result_param = param
                result_param_seasonal = param_seasonal
                final_predictions = pred

    return {'rmse':min_rmse, 'mape':min_model_mape, 'aic':min_model_aic, 'prediction':final_predictions, 'parameter':result_param, 'seasonal paramater':result_param_seasonal}

def sarimax_modelling2(data,exog,div=0.9,max_range=1):
    seas_values = [12]
    result_list = []
    train_size = int(len(data) * div)
    min_rmse = 999999
    mape = 99999
    param = None
    seas_param = None
    train_exog,test_exog = exog[0:train_size], exog[train_size:len(data)]
    for i in seas_values:
#         try:
#             result_list.append(get_sarimax(data,train_exog,test_exog,max_range,seasonal_factor=i))
        temp_res = get_sarimax(data,train_exog,test_exog,max_range,seasonal_factor=i)
        temp_rmse = temp_res['rmse']
        if temp_rmse<min_rmse:
            min_rmse = temp_rmse
            param = temp_res['parameter']
            seas_param = temp_res['seasonal paramater']
            mape = temp_res['mape']
#         except:
#             pass
    return {'param':param,'seas_param':seas_param,'rmse':min_rmse,'mape':mape}

def data_creation_sarima(data,division):
    """
    return
    -----------------
    split[0]: training set
    split[1]: test set
    """
    split = []
    div = division
    data.fillna(method='ffill',inplace=True)
    train_size = int(len(data) * div)
    train, test = data[0:train_size], data[train_size:len(data)]
    split.append(train)
    split.append(test)
    return(split)

def scoring_rmse(y_pred, y_test):
    print(y_pred)
    print(y_test)
    rmse = np.sqrt(np.mean(np.square(y_test.values - y_pred.values)))
    return rmse

def scoring_mape(pred,test):
    try:
        mape = np.mean(np.abs((test - pred) / test))
    except:
        mape = -1
    return mape

def predict_data(country, region):
    cdata= pd.read_csv('finalDataset.csv', index_col=0)
    cdata['date'] =  pd.to_datetime(cdata['date'], infer_datetime_format=True)
    cdata = cdata.set_index('date')

    data_country = country
    data_sub_region = region
    data_req_input = cdata.loc[cdata['country'] == data_country]
    data_req_in2 = data_req_input.loc[data_req_input['sub-region'] == data_sub_region]
    data = data_req_in2['Confirmed'].astype(float)
    # exog = cdata[['driving','transit','walking'],'date']
    exog = data_req_in2[['driving','transit','walking']].fillna(method='ffill').fillna(0).astype(float)

    p,d,q,P,D,Q,M = 1,1,1,1,0,0,12
    model = sarimax(data,None,p,d,q,P,D,Q,M,summary=False)

    #need to create the exog_forecast dataframe which contains exogenous variables for future time
    # forecast = fcast_sarimax(model,None,10)
    exog_list=['driving','walking','transit']
    exog_df = pd.DataFrame()
    for var in exog_list:
        data_ = cdata[var]
        model_ = sarimax(data_,None,p,d,q,P,D,Q,M,summary=False)
        forecast_ = fcast_sarimax(model_,None,30)
        exog_df[var] = forecast_

    # loop ended here
    #replace exog in last line with exog_df and 10 by 30

    forecast_drive = fcast_sarimax(model,exog_df,30)
    return forecast_drive

@app.route("/<string:country>/<string:region>", methods=['GET'])
def runModel(country, region):
    print(country)
    print(region)
    result = predict_data(country, region)
    return {
        'statusCode': 200,
        'body': dumps({
            'result': result
        }),
        'headers' : {
            'Access-Control-Allow-Origin' : '*'
        }
    }

if __name__ == "__main__":
    app.run()