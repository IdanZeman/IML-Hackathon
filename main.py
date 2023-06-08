from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from preprocess import preprocess


def soft_svm_model():
    # Assuming you have your features stored in X and labels in y
    X, y = preprocess(filename="Data/agoda_cancellation_train.csv")
    # ###
    # c = ['hotel_live_date', 'accommadation_type_name', 'original_payment_method',
    #            'original_payment_type', 'is_user_logged_in', 'cancellation_policy_code',
    #            'is_first_booking', 'delta_cancel_before_checking', 'charge_option']
    # X = X.drop(columns=c)
    # y = y.drop(columns=c)
    # ###
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99,
                                                        random_state=42)  # TODO: check random_state
    svm = SVC(kernel='linear', probability=True)
    i =2
    headers = X_train.columns
    svm.fit(X_train.drop(columns=["h_booking_id"]).iloc[:, 1:9].to_numpy(), y_train.to_numpy())

    while(i):
        print(i)
        print(headers[i])
        svm.fit(X_train.drop(columns=["h_booking_id"]).iloc[:, 1:i].to_numpy(), y_train.to_numpy())
        i+=1
    print("finish fit")
    y_pred = svm.predict(X_test)
    print(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)


if __name__ == '__main__':
    soft_svm_model()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
