import pandas as pd # data processing
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # statistical data visualization
sns.set_style('dark')

# machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load loan payment data
loan_data = pd.read_csv('loan_data.csv')

# clean up data from unused criteria
to_delete = ['Loan_ID', 'effective_date', 'due_date', 'paid_off_time']
for criterion in to_delete:
    del loan_data[criterion]

# principal amount and loan status
plt.figure(figsize=(10,5))
plt.title("Principle Amount vs Loan Status")
sns.countplot(y = 'Principal', hue = "loan_status", data = loan_data, palette = "hls")
plt.show()

# number of data points for past due payments that were sent to collection
collection = loan_data['past_due_days'][loan_data['loan_status'] == 'COLLECTION']
# number of data points for past due payments that were sent to collection and were paid off
collection_paidoff = loan_data['past_due_days'][loan_data['loan_status'] == 'COLLECTION_PAIDOFF']

# get averages
avg_days = DataFrame([collection.mean(), collection_paidoff.mean()])
std_days = DataFrame([collection.std(), collection_paidoff.std()])
avg_age = loan_data["age"].mean()
std_age = loan_data["age"].std()
# delete column for past due days
loan_data = loan_data.drop(['past_due_days'], axis = 1)

# age and loan status
plt.figure(figsize=(10,5))
plt.title("Age vs Loan Status")
sns.countplot(x = "age", hue = "loan_status", data = loan_data, palette = "dark")
plt.show()

# gender and loan status
plt.figure(figsize=(10,5))
plt.title("Gender vs Loan Status")
sns.countplot(x = "loan_status", hue = "Gender", data = loan_data, palette = "cubehelix")
loan_data['Gender'] = loan_data['Gender'].map({'female' : 0, 'male' : 1}).astype(int)
plt.show()

# education and loan status
plt.figure(figsize=(10,5))
sns.countplot(x = "loan_status", hue = "education", data = loan_data, palette = "Set2")
title_mapping = {"High School or Below": 1, "college": 2, "Bechalor": 3, "Master or Above": 4 }
loan_data['education'] = loan_data['education'].map(title_mapping)
plt.show()

# divide the data for testing
collection_data = loan_data.loc[(loan_data["loan_status"] == 'COLLECTION') | (loan_data["loan_status"]
                                                                             == 'COLLECTION_PAIDOFF')]
collection_paidoff_data = loan_data.loc[(loan_data["loan_status"] == 'PAIDOFF') | (loan_data["loan_status"]
                                                                             == 'COLLECTION')]

# set up train variables
X_train = collection_paidoff_data.drop("loan_status", axis = 1)
Y_train = collection_paidoff_data['loan_status']

# train the data with Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
print("Random Forrest Classification Result: ")
print(random_forest.score(X_train, Y_train))

# output results of Random Forrest Classification
feature_names = collection_data.columns.delete(0)
plt.figure(figsize=(10, 5))
index = np.arange(len(feature_names))
bar_width = 0.35
plt.bar(index, random_forest.feature_importances_, align="center" )
plt.xlabel('Features')
plt.ylabel('Weight')
plt.title("Random Forrest's Feature Weight")
plt.xticks(index + bar_width, feature_names)
plt.tight_layout()
plt.show()

# train the data with K-nearest Classifier
classifier = KNeighborsClassifier(n_neighbors=len(feature_names))
classifier.fit(X_train, Y_train)
print("K-Nearest Classification Result: ")
print(classifier.score(X_train, Y_train))