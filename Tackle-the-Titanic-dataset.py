import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.preprocessing import minmax_scale
import numbers


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]


# Inspired from stackoverflow.com/questions/25239958
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X], index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)


train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

train_data.head()
train_data.info()
train_data.columns
train_data.describe()
train_data["Survived"].value_counts()
# plt.figure()
# train_data.hist(figsize=(15, 15))
# plt.show()

num_pipeline = Pipeline([("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),
                         ("imputer", SimpleImputer(strategy="median")), ])

cat_pipeline = Pipeline(
    [("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])), ("imputer", MostFrequentImputer()),
     ("cat_encoder", OneHotEncoder(sparse=False)), ])

preprocess_pipeline = FeatureUnion(transformer_list=[("num_pipeline", num_pipeline), ("cat_pipeline", cat_pipeline), ])

X_train = preprocess_pipeline.fit_transform(train_data)
y_train = train_data["Survived"]

svm_clf = SVC(gamma="auto")
svm_clf.fit(X_train, y_train)

X_test = preprocess_pipeline.transform(test_data)
y_pred = svm_clf.predict(X_test)

svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
svm_scores.mean()

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
forest_scores.mean()

plt.figure(figsize=(8, 4))
plt.plot([1] * 10, svm_scores, "r.")
plt.plot([2] * 10, forest_scores, "g.")
plt.boxplot([svm_scores, forest_scores], labels=("SVM", "Random Forest"))
plt.ylabel("Accuracy", fontsize=14)
plt.show()

train_data["AgeBucket"] = train_data["Age"] // 15 * 15
train_data[["AgeBucket", "Survived"]].groupby(['AgeBucket']).mean()

train_data["RelativesOnboard"] = train_data["SibSp"] + train_data["Parch"]
train_data[["RelativesOnboard", "Survived"]].groupby(['RelativesOnboard']).mean()



########################################################################################################################

from tensorflow.keras.models import Sequential
from  tensorflow.keras.layers import *

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")
train_data.dropna(how="all",inplace=True)
features = ["Age", "SibSp", "Parch", "Fare","Pclass", "Sex", "Embarked"]

# Pandas can issue a SettingWithCopyWarning when you try to modify the copy of data instead of the original.
# This often follows chained indexing.

for feature in  features:
    if train_data[feature].dtype == np.dtype("O"):
        train_data[feature] = pd.factorize(train_data[feature])[0]

X_train = train_data[features]
X_train = SimpleImputer(strategy="median").fit_transform(X_train)


X_train = minmax_scale(X_train, axis=0)
y_train = train_data[["Survived"]].to_numpy()

idx = np.arange(X_train.shape[0])
np.random.shuffle(idx)
X_train = X_train[idx]
y_train = y_train[idx]

model = Sequential([
    Dense(64,activation = "relu",input_shape=(X_train.shape[-1],)),
    Dropout(0.5),
    Dense(32, activation = "relu"),
    Dropout(0.5),
    Dense(1, activation ="sigmoid")
])
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
model.fit(X_train,y_train,validation_split=0.2,epochs=200)