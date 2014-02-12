from pprint import pprint

# read more: http://www.numpy.org/
import numpy

# read more: http://pandas.pydata.org/
import pandas

# read more: http://wiki.scipy.org/PyLab
# import pylab  # uncomment for plots

# read more: http://scikit-learn.org/stable/
from sklearn.neighbors import KNeighborsClassifier


def train_k_neighbours(n, features_set, predict_set):
    model = KNeighborsClassifier(n_neighbors=n)
    # http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    model.fit(features_set, predict_set)
    return model


def split_set(dataframe):
    uniform_array = numpy.random.uniform(0, 1, len(dataframe))
    train_set = dataframe[uniform_array > 0.3]
    test_set = dataframe[uniform_array <= 0.3]
    return train_set, test_set


def n_fold_set(n, dataframe):
    uniform_array = numpy.random.uniform(0, 1, len(dataframe))
    sets = []
    for i in range(1, n):
        sets.append(
            dataframe[uniform_array > 0.3]
        )

csv = "iris.csv"

data_frame = pandas.read_csv(csv)
# >>> data_frame.info()
# <class 'pandas.core.frame.DataFrame'>
# Int64Index: 150 entries, 0 to 149
#
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html

features = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']
predict = 'class'

train_set, test_set = split_set(data_frame)

results = []
best = None
for n in range(1, len(train_set), 1):
    model = train_k_neighbours(n, train_set[features], train_set[predict])
    preds = model.predict(test_set[features])

    accuracy = numpy.where(
        preds == test_set[predict], 1, 0
    ).sum() / float(len(test_set))

    if not best or best['accuracy'] < accuracy:
        best = {
            'accuracy': accuracy,
            'parameters': model.get_params()
        }
    results.append([n, accuracy])
pprint(best)

# uncomment for plots
# results = pandas.DataFrame(results, columns=["n", "accuracy"])
# pylab.plot(results.n, results.accuracy)
# pylab.title("Accuracy with Increasing K")
# pylab.show()
