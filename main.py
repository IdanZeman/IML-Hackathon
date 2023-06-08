from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from preprocess import preprocess


def soft_svm_model():
    # Assuming you have your features stored in X and labels in y
    X, y = preprocess(filename="Data/agoda_cancellation_train.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=42)  # TODO: check random_state
    svm = SVC(C=1.0, kernel='linear')
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)


if __name__ == '__main__':
    soft_svm_model()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
