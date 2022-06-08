import logging
import math
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.svm import SVR

class xgb_regression:
    def __init__(self, cdn_name, data_series, model_params, learner):
        logging.info("initing object xgb_regression...")
        self.cdn_name     = cdn_name
        self.data_series  = data_series
        self.model_params = model_params
        self.learner      = learner

    def construct_data_set(self):
        logging.info("construct to train data set.")
        data_df = pd.DataFrame(self.data_series, columns=["x_0"])
        feature_points = self.model_params["feature_points_length"]

        for i in range(1, feature_points):
            data_df["x_%s" %(i)] = data_df["x_0"].shift(-i)
        
        data_df["y"] = data_df["x_0"].shift(feature_points)
        data_df.dropna(inplace=True)

        X = data_df.values[:, : -1]
        y = data_df.values[:, -1]

        return X, y
    
    def train_data_set(self):
        logging.info("train data set")
        X, y = self.construct_data_set()

        #split X, y to train and test
        test_size = (int)(X.shape[0] * self.model_params["test_size_ration"])

        X_train, X_test = X[: -test_size], X[-test_size :]
        y_train, y_test = y[: -test_size], y[-test_size :]

        #create and train model
        if self.learner == "xgboost":
            xgb_regressor = xgb.XGBRegressor()
            xgb_regressor.fit(X_train, y_train)

            #predict on test data
            prediction = xgb_regressor.predict(X_test)
        elif self.learner == "svm":
            poly_svr = SVR()
            poly_svr.fit(X_train, y_train)

            prediction = poly_svr.predict(X_test)

        self.y_train = y_train
        self.y_test  = y_test
        self.predict = prediction
    
    def plot_train_result(self):
        logging.info("ploting train result")
        test_len = self.y_test.shape[0]
        
        #train
        x1 = np.arange(test_len)
        y1 = self.y_train[-test_len :]
        df_train = pd.DataFrame(y1, index=x1, columns=["train"])

        #test and predict
        x23 = np.arange(test_len, 2*test_len)
        y2  = self.y_test
        y3  = self.predict
        df_test     = pd.DataFrame(y2, index=x23, columns=["test"])
        df_predict  = pd.DataFrame(y3, index=x23, columns=["predict"])

        #concat train, test, prediction
        df_result = pd.concat([df_train, df_test, df_predict], axis = 0)

        #plot
        df_result.plot()
        plt.title("%s_prediction" %(self.cdn_name))
        plt.xlabel("timeline")
        plt.ylabel("traffic")
        plt.grid()
        #plt.show()
        plt.savefig(self.cdn_name + "_" + self.learner)

    def calculate_error(self):
        logging.info("calculating model prediction error")

        self.mape   = np.mean(np.abs((self.y_test - self.predict) / self.y_test)) * 100
        self.mae    = metrics.mean_absolute_error(self.y_test, self.predict)
        self.rmse   = math.sqrt(metrics.mean_squared_error(self.y_test, self.predict))

        print("MAPE   : %s\nMAE    : %s\nRMSE   : %s" %(self.mape, self.mae, self.rmse))
    
    def model_run(self):
        self.train_data_set()
        self.plot_train_result()
        self.calculate_error()

if __name__ == '__main__':
    print("============== XGBoost training process ==============")


