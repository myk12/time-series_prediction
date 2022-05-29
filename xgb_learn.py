import csv
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def load_seq_data(file_list):
    # load data from file
    seq_data = []
    for file_path in file_list:
        print("#reading sequence data from file :", file_path)
        with open(file_path) as f:
            csv_f = csv.reader(f)
            for row in csv_f:
                seq_data.extend(list(map(int, row)))

    return seq_data

def show_data_set(data_list, figure = False):
    # display data set
    if figure:
        plt.title("data")
        plt.xlabel("timeline")
        plt.ylabel("aliyun_req")
        plt.plot(data_list)
        plt.show()
    
    else:
        print(data_list)

def construct_data_set(data_list, shift = 7):
    print("# construct to train data set")
    data_set_df = pd.DataFrame(data_list, columns=["x_0"])
    for i in range(shift):
        d = i + 1
        data_set_df["x_%s" %d] =  data_set_df["x_0"].shift(-d)

    data_set_df["y"] = data_set_df["x_0"].shift(shift)
    data_set_df.dropna(inplace=True)

    print(data_set_df)
    data_array = data_set_df.values

    # split to X, y
    X = data_array[:, : -1]
    y = data_array[:, -1]

    return X, y

def split_data_set(X, y, ratio=0.1, random = False, predict = 10):
    print("# spliting data set")

    if random:
        return train_test_split(X, y, test_size=ratio, random_state=0)
    else:
        test_size = predict
        X_train = X[: -test_size]
        X_test = X[-test_size:]
        y_train = y[: -test_size]
        y_test = y[-test_size:]
    return X_train, X_test, y_train, y_test

def xgb_regression(X_train, y_train, X_test, y_test):
    # init model and train
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)

    # predict on test data
    prediction = model.predict(X_test)

    # mae
    mae = mean_absolute_error(y_test, prediction)
    print("mae: ", mae)

    return prediction

def plot_train_result(y_train, y_test, predictions, feature_points, predict_points):
    train_len = y_train.shape[0]
    test_len = y_test.shape[0]
    pre_show = 4

    # train data
    x1 = np.arange(train_len)[- pre_show* test_len : ]
    y1 = y_train[- pre_show* test_len : ]
    df_train = pd.DataFrame(y1, index = x1, columns = ["train"])

    # test data
    x2 = np.arange(train_len, train_len + y_test.shape[0])
    y2 = y_test
    df_test = pd.DataFrame(y2, index = x2, columns = ["test"])

    # prediction
    x3 = np.arange(train_len, train_len + len(predictions))
    y3 = predictions
    df_predict = pd.DataFrame(y3, index = x3, columns=["predict"])

    # concat train, test, prediction
    df_result = pd.concat([df_train, df_test, df_predict], axis=0)

    # plot
    df_result.plot()
    plt.title("aliyun_prediction: n=%s, predict_points=%s" %(feature_points, predict_points))
    plt.xlabel("timeline")
    plt.ylabel("traffic")
    plt.grid()
    plt.show()


if __name__ == '__main__':
    print("============== XGBoost training process ==============")

    # load config from file config.json
    with open("config.json", "r", encoding="utf-8") as config:
        config_json = json.load(config)
    model_param = config_json["model_param"]
    data_file_list = ["./data/aliyun_hour_46.csv", "./data/aliyun_hour_47.csv", "./data/aliyun_hour_48.csv"]
    #data_file_list = ["./data/aliyun_hour_46.csv"]

    # read data
    data_list = load_seq_data(data_file_list)

    # show data
    show_data_set(data_list, figure=False)

    # construct train data set
    X, y = construct_data_set(data_list, model_param["feature_points"])

    # split data
    X_train, X_test, y_train, y_test = split_data_set(X, y, ratio = 0.01, random = False, predict = model_param["predict_points"])

    print("X_train   : ", X_train.shape)
    print("X_test    : ", X_test.shape)
    print("y_train   : ", y_train.shape)
    print("y_test    : ", y_test.shape)

    prediction = xgb_regression(X_train, y_train, X_test, y_test)

    plot_train_result(y_train, y_test, prediction, model_param["feature_points"], model_param["predict_points"])


