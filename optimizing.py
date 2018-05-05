
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tpot import TPOTRegressor


dengue = pd.read_csv('d.csv')
dengue['Dengue_Cases'] = np.log1p(dengue['Dengue_Cases'])
y = dengue['Dengue_Cases']
x = dengue.drop('Dengue_Cases', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01)


tpot = TPOTRegressor(generations=20, population_size=50, verbosity=2)
tpot.fit(x_train, y_train)
print(tpot.score(x_test, y_test))
tpot.export('pipeline.py')

 # GradientBoostingRegressor(OneHotEncoder(OneHotEncoder(SelectPercentile(percentile=79), minimum_fraction=0.05, sparse=False), minimum_fraction=0.05, sparse=False), alpha=0.99, learning_rate=0.1, max_depth=6, max_features=1.0, min_samples_leaf=13, min_samples_split=9, n_estimators=100, subsample=0.8)