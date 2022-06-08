import os
import csv
import json
import pandas as pd
import ml_xgboost
import ml_lightgbm

def load_data_from_csv_to_dataframe(filename):
    #load data
    data_df = pd.read_csv(filename)
    data_df.drop("Unnamed: 0", axis=1, inplace=True)

    return data_df

if __name__ == '__main__':
    #read config file
    with open("config.json", "r", encoding="utf-8") as conf:
        conf_json = json.load(conf)
    
    #load data
    data_df = load_data_from_csv_to_dataframe(conf_json["data_filename"])
    print(data_df)

    static_df = pd.DataFrame(columns=["MAPE(%)", "RMSE", "MAE"])
    #train every cdn and predict
    for cdn in conf_json["cdn_name_list"]:
        if "xgboost" in conf_json["ml_model_list"]:
            xgb = ml_xgboost.xgb_regression(cdn, data_df[cdn].values, conf_json["model_params"], "xgboost")
            xgb.model_run()

            static_df.loc[cdn + "_xgboost"] = [xgb.mape, xgb.rmse, xgb.mae]
        
        if "svm" in conf_json["ml_model_list"]:
            xgb = ml_xgboost.xgb_regression(cdn, data_df[cdn].values, conf_json["model_params"], "svm")
            xgb.model_run()

            static_df.loc[cdn + "_svm"] = [xgb.mape, xgb.rmse, xgb.mae]
        
        if "lightgbm" in conf_json["ml_model_list"]:
            lgb = ml_lightgbm.lightgbm_regression(cdn, data_df[cdn].values, conf_json["model_params"], "lightgbm")
            lgb.model_run()

            static_df.loc[cdn + "_lightgbm"] = [lgb.mape, lgb.rmse, lgb.mae]
    
    print(static_df)
    static_df.to_csv("./static.csv", float_format="%.4f")

