import os
import csv
import json
import pandas as pd
import ml_xgboost

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

    #train every cdn and predict
    for cdn in conf_json["cdn_name_list"]:
        if "xgboost" in conf_json["ml_model_list"]:
            xgb = ml_xgboost.xgb_regression(cdn, data_df[cdn].values, conf_json["model_params"])
            xgb.model_run()
        
        if "svm" in conf_json["ml_model_list"]:
            print("using svm")

