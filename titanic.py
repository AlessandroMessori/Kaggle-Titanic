import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, \
    ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from src.data_cleaning import DataCleaner
from src.data_formatter import save_preds_tocsv
from src.ann import NetClassifier

ann = NetClassifier()
ann.build()

classifiers = [
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    AdaBoostClassifier(),
    BaggingClassifier(),
    ExtraTreesClassifier(),
    GaussianNB(),
    LogisticRegression(),
    KNeighborsClassifier(),
    ann
]

X_train = pd.read_csv("data/train.csv")
X_test = pd.read_csv("data/test.csv")

X_train['famCount'] = X_train['SibSp'] + X_train['Parch']
X_test['famCount'] = X_test['SibSp'] + X_test['Parch']

Y_train = X_train.iloc[:, 1]
X_train = X_train.drop(columns='Survived')

train_cleaner = DataCleaner(X_train)
test_cleaner = DataCleaner(X_test)

train_cleaner.fill("Cabin")
train_cleaner.encode_and_scale()

for classifier in classifiers:
    classifier.fit(train_cleaner.df, Y_train)

test_cleaner.fill("Cabin")
test_cleaner.fill(column="Age", mean=True)
test_cleaner.encode_and_scale()

preds = [cls.predict(test_cleaner.df) for cls in classifiers]

save_preds_tocsv(X_test["PassengerId"], preds)
