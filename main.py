import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from preprocess import preprocess


def soft_svm_model():
    # Assuming you have your features stored in X and labels in y
    # ###
    # c = ['hotel_live_date', 'accommadation_type_name', 'original_payment_method',
    #            'original_payment_type', 'is_user_logged_in', 'cancellation_policy_code',
    #            'is_first_booking', 'delta_cancel_before_checking', 'charge_option']
    # X = X.drop(columns=c)
    # y = y.drop(columns=c)
    # ###
    models = [
        LogisticRegression(penalty="none"),
        DecisionTreeClassifier(max_depth=5),
        KNeighborsClassifier(n_neighbors=5),
        # SVC(kernel='linear', probability=True),
        LinearDiscriminantAnalysis(store_covariance=True),
        QuadraticDiscriminantAnalysis(store_covariance=True),
        RandomForestClassifier(max_depth=5,random_state=0)

    ]
    ## "Linear SVM"
    model_names = ["Logistic regression", "Desicion Tree (Depth 5)", "KNN", "LDA", "QDA","Random Forest Classifier"]
    df = pd.read_csv("Data/agoda_cancellation_train.csv")
    X, y = df.drop(columns=["cancellation_datetime"]), df["cancellation_datetime"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=42)  # TODO: check random_state
    y_test = y_test.notnull().astype(int)

    (preprocessed_X_train, preprocessed_y_train), (preprocessed_X_test, preprocessed_y_test) = \
        preprocess(X_train.drop(columns=["h_booking_id"]), y_train), preprocess(X_test.drop(columns=["h_booking_id"]), None)
    # all_dummis = set(preprocessed_X_train.columns).union(preprocessed_X_test.columns)
    # preprocessed_X_train = preprocessed_X_train.reindex(columns=all_dummis, fill_value=0)
    # preprocessed_X_test = preprocessed_X_test.reindex(columns=all_dummis, fill_value=0)


    y_pred_arr = []
    accuracy_scores = []
    F1_arr = []
    common_columns = preprocessed_X_train.columns.intersection(preprocessed_X_test.columns).tolist()
    # Add missing columns from preprocessed_X_train to preprocessed_X_test
    missing_columns = preprocessed_X_train.columns.difference(preprocessed_X_test.columns).tolist()
    preprocessed_X_test = preprocessed_X_test.reindex(columns=common_columns+missing_columns, fill_value=0)
    for i, model in enumerate(models):
        model.fit(preprocessed_X_train.to_numpy(), preprocessed_y_train.to_numpy())
        print("Model - " + model_names[i]+": ")
        y_pred = model.predict(preprocessed_X_test.to_numpy())
        y_pred_arr.append(y_pred)
        accuracy = accuracy_score(y_test.loc[preprocessed_X_test.index], y_pred)
        Tp = np.sum(y_pred == y_test.loc[preprocessed_X_test.index].to_numpy())
        Fp = np.sum((y_pred != y_test.loc[preprocessed_X_test.index].to_numpy()) & (y_pred == 1))
        Fn = np.sum((y_pred != y_test.loc[preprocessed_X_test.index].to_numpy()) & (y_pred == 0))
        precision = Tp / (Tp + Fp)
        recall = Tp / ( (Tp + Fn))
        # F1 = 2 * (precision * recall) / (precision + recall)
        F1 = f1_score(y_test.loc[preprocessed_X_test.index], y_pred, average="macro")
        F1_arr.append(F1)
        accuracy_scores.append(accuracy)
        print("Accuracy:", accuracy)
        print("F1:", F1)
    X_agoda_test = pd.read_csv("Data/Agoda_Test_1.csv")
    preprocessed_X_agoda_test, _ = preprocess(X_agoda_test.drop(columns=["h_booking_id"]), None)
    preprocessed_X_agoda_test = preprocessed_X_agoda_test.reindex(columns=common_columns+missing_columns, fill_value=0)
    max_index = np.argmax(F1_arr)
    y_pred_agoda = models[max_index].predict(preprocessed_X_agoda_test.to_numpy())
    # Create the output DataFrame with id and cancellation columns
    output_df = pd.DataFrame({"id": X_agoda_test.loc[preprocessed_X_agoda_test.index]['h_booking_id'],
                              "cancellation": y_pred_agoda})

    # Save the output DataFrame to a CSV file
    output_df.to_csv("agoda_cancellation_prediction.csv", index=False)


def task3(df:  DataFrame):
    cancellation_datetime = df.drop(columns=["cancellation_datetime"]).notnull().astype(int)
    X, y = preprocess(df.drop(columns=["cancellation_datetime"]), df["cancellation_datetime"])
    corr_arr = []
    corr_col = []
    for i, col in enumerate(X.columns):
        corr = np.corrcoef(X[col], y)[0][1]
        corr_arr.append((np.abs(corr), col))
        print(corr)
        corr_col.append(col)
        print(col)
    corr_arr = sorted(corr_arr, key=lambda x: x[0], reverse=True)
    print(corr_arr)
    # Bar plot of correlation coefficients
    corr_values = [corr[0] for corr in corr_arr[1:10]]
    feature_names = [corr[1] for corr in corr_arr[1:10]]
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, 10), corr_values)
    # plt.text(range(1, 10), corr_values,)
    plt.xlabel("Features")
    plt.ylabel("Absolute Correlation")
    plt.title("Correlation Coefficients")
    plt.xticks(rotation=90)
    plt.savefig("correlation_plot.png")  # Save the figure as a PNG file
    plt.show()


if __name__ == '__main__':
    soft_svm_model()
    df = pd.read_csv("Data/agoda_cancellation_train.csv")
    # task3(df)

