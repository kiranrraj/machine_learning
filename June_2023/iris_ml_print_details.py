from sklearn.datasets import load_iris

iris_dataset = load_iris()
print("IRIS Target Names {}".format(iris_dataset['target_names']))
print("IRIS Feature Names {}".format(iris_dataset['feature_names']))
# print("IRIS Dataset Description:\n",iris_dataset["DESCR"][:1208])
print("IRIS Dataset shape: {}".format(iris_dataset['data'].shape))
print("IRIS First 5 columns: {}".format(iris_dataset['data'][:5]))
print()