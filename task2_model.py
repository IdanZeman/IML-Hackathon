import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from taske2_preprocess import preprocess

# todo: excuter to pdf the prediction of agoda test 2
# make sure that you remove the cancltion from the preprocess 2
# take first task 2 predict then

def model():
    df = pd.read_csv("Data/agoda_cancellation_train.csv")
    X, y = df.drop(columns=["original_selling_amount"]), df["original_selling_amount"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=42)  # TODO: check random_state

    (preprocessed_X_train, preprocessed_y_train), (preprocessed_X_test, preprocessed_y_test) = \
        preprocess(X_train.drop(columns=["h_booking_id"]), y_train), preprocess(X_test.drop(columns=["h_booking_id"]),
                                                                                None)

    all_dummis = set(preprocessed_X_train.columns).union(preprocessed_X_test.columns)
    preprocessed_X_train = preprocessed_X_train.reindex(columns=all_dummis, fill_value=0)
    preprocessed_X_test = preprocessed_X_test.reindex(columns=all_dummis, fill_value=0)

    #
    # X_agoda_test = pd.read_csv("Data/Agoda_Test_1.csv")
    # preprocessed_X_agoda_test, _ = preprocess(X_agoda_test.drop(columns=["h_booking_id"]), None)
    # preprocessed_X_agoda_test = preprocessed_X_agoda_test.reindex(columns=all_dummis, fill_value=0)
    models = [
        linear_model.LinearRegression(),
        linear_model.Ridge(alpha=300000),
        linear_model.Lasso(alpha=1000)
    ]
    models_name = ['Linear Regression','Ridge','Lasso']
    X_test, _ = preprocess(X_test, None)
    for i in range(len(models)):
        lg = models[i]
        lg.fit(preprocessed_X_train, preprocessed_y_train)
        print(models_name[i] + '----------------------------------------------------------------------------')
        print('train losse ' + str(np.sqrt(mean_squared_error(preprocessed_y_train[preprocessed_X_train.index],lg.predict(preprocessed_X_train)))))
        s = mean_squared_error(y_test.loc[preprocessed_X_test.index], lg.predict(preprocessed_X_test))
        print('test losse ' + str(np.sqrt(s)))
        print(models_name[i] + '----------------------------------------------------------------------------\n')


model()
