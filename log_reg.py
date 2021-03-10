import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%matplotlib inline
import pandas as pd
import os

import pickle

df = pd.read_csv(os.path.join('Resources', 'cardio_train.csv'))
df.head()


def load_data():
    # assume data file will always be the same per training
    data = pickle.load(open('./df.pkl', 'rb'))
    return data

def load_model(model_file_name):
    loaded_model = pickle.load(open(model_file_name, 'rb'))
    return loaded_model

def runModel():
    """
    this function has mode logic that works on data
    """

df=df.drop('id', axis=1)
df.age = df.age.apply(lambda x: x / 365, 2)
# combining height and weight into 1 -- BMI
df['bmi'] = df['weight'] / (df['height']/100)** 2
df = df[['age','gender','height','weight','bmi','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active','cardio']]

df.round().astype('int')
df.head()

# Assign X (data) and y (target)
X = df.drop("cardio", axis=1)
y = df["cardio"]
print(X.shape, y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(max_iter=1000)
classifier

classifier.fit(X_train, y_train)

#print(f"Training Data Score: {classifier.score(X_train, y_train)}")
#print(f"Testing Data Score: {classifier.score(X_test, y_test)}")

predictions = classifier.predict(X_test)
#print(f"First 10 Predictions:   {predictions[:10]}")
#print(f"First 10 Actual labels: {y_test[:10].tolist()}")

pred_actual = pd.DataFrame({"Prediction": predictions, "Actual": y_test}).reset_index(drop=True)

# Create the GridSearch estimator along with a parameter object containing the values to adjust
# from sklearn.model_selection import GridSearchCV
# param_grid = {'C': [1, 5, 10],
#               'penalty': ["l1","l2"]}
model=LogisticRegression(solver="liblinear")
# grid = GridSearchCV(model, param_grid, verbose=3)

pickle.dump(model, open('df.pkl', 'wb'))
pickle.dump(model, open('cardio.pkl', 'wb'))
# Fit the model using the grid search estimator. 
# This will take the SVC model and try each combination of parameters
# grid.fit(X_train,y_train)

# print(grid.best_params_)
# print(grid.best_score_)

#  # Calculate classification report
# from sklearn.metrics import classification_report
# print(classification_report(y_test, predictions,
#                             target_names=["blue", "red"]))

cardio = df.copy()

cardio.loc[(df["age"] < 40), 'AGE_RANGE'] = 30
cardio.loc[(df["age"] >= 40) & (df['age'] <50), 'AGE_RANGE'] =40
cardio.loc[(df["age"] >= 50) & (df['age'] <60), 'AGE_RANGE'] =50
cardio.loc[(df["age"] >= 60) & (df['age'] <70), 'AGE_RANGE'] =60
cardio.head()

def load_data():
    # assume data file will always be the same per training
    data = pickle.load(open('./cardio.pkl', 'rb'))
    return data

