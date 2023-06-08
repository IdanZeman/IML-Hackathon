import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.tree import DecisionTreeClassifier

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
        QuadraticDiscriminantAnalysis(store_covariance=True)
    ]
    ## "Linear SVM"
    model_names = ["Logistic regression", "Desicion Tree (Depth 5)", "KNN", "LDA", "QDA"]
    df = pd.read_csv("Data/agoda_cancellation_train.csv")
    X, y = df.drop(columns=["cancellation_datetime"]), df["cancellation_datetime"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=42)  # TODO: check random_state
    preprocessed_X_train, preprocessed_y_train = preprocess_train(X_train, y_train)
    for i, model in enumerate(models):
        model.fit(X_train.drop(columns=["h_booking_id"]).iloc[:, 1:].to_numpy(), y_train.to_numpy())
        print("Model - " + model_names[i]+": ")
        y_pred = model.predict(X_test.drop(columns=["h_booking_id"]).iloc[:, 1:].to_numpy())
        print(y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)


if __name__ == '__main__':
    soft_svm_model()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
