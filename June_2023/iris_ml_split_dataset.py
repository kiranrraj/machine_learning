from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris_dataset = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], 
    iris_dataset['target'],
    random_state=0)

print("X train shape {}".format(X_train.shape))
print("y train shape {}".format(y_train.shape))

print("X test shape {}".format(X_test.shape))
print("y test shape {}".format(y_test.shape))