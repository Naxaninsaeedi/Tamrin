import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



# Read the CSV file
DataSet = pd.read_csv('1.csv')

DataSet['class'] = DataSet['class'].map({'tested_positive': 1, 'tested_negative': 0})
DataSet.set_index(pd.RangeIndex(start=0, stop=len(DataSet)), inplace=True)
pd.set_option('display.max_columns', None)

print(DataSet.describe())

# Split the dataset into features (X) and target variable (y)
X = DataSet.drop('class', axis=1)
y = DataSet['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# C4.5 Decision Tree
c45_tree = DecisionTreeClassifier(criterion='entropy', random_state=42)
c45_tree.fit(X_train, y_train)
c45_pred = c45_tree.predict(X_test)

# Visualize C4.5 Decision Tree
plt.figure(figsize=(256, 128))
plot_tree(c45_tree, filled=True, feature_names=X.columns, class_names=c45_tree.classes_.astype(str))
plt.savefig('c45_tree.png')

# CART Decision Tree
cart_tree = DecisionTreeClassifier(criterion='gini', random_state=42)
cart_tree.fit(X_train, y_train)
cart_pred = cart_tree.predict(X_test)

# Visualize CART Decision Tree
plt.figure(figsize=(256, 128))
plot_tree(cart_tree, filled=True, feature_names=X.columns, class_names=cart_tree.classes_.astype(str))
plt.savefig('cart_tree.png')



# ID3 Decision Tree
id3_tree = DecisionTreeClassifier(criterion='entropy', splitter='best', random_state=42)
id3_tree.fit(X_train, y_train)
id3_pred = id3_tree.predict(X_test)

# Visualize ID3 Decision Tree
plt.figure(figsize=(256, 128))
plot_tree(id3_tree, filled=True, feature_names=X.columns, class_names=id3_tree.classes_.astype(str))
plt.savefig('id3_tree.png')


# Linear SVM
linear_svm = SVC(kernel='linear', random_state=42)
linear_svm.fit(X_train, y_train)
linear_svm_pred = linear_svm.predict(X_test)


# RBF Kernel SVM
rbf_svm = SVC(kernel='rbf', random_state=42)
rbf_svm.fit(X_train, y_train)
rbf_svm_pred = rbf_svm.predict(X_test)

# Polynomial Kernel SVM
poly_svm = SVC(kernel='poly', degree=3, random_state=42)
poly_svm.fit(X_train, y_train)
poly_svm_pred = poly_svm.predict(X_test)

# Sigmoid Kernel SVM
sigmoid_svm = SVC(kernel='sigmoid', random_state=42)
sigmoid_svm.fit(X_train, y_train)
sigmoid_svm_pred = sigmoid_svm.predict(X_test)

# Evaluate the models
models = {
    'C4.5 Decision Tree': c45_pred,
    'CART Decision Tree': cart_pred,
    'Linear SVM': linear_svm_pred,
    'RBF Kernel SVM': rbf_svm_pred,
    'Polynomial Kernel SVM': poly_svm_pred,
    'Sigmoid Kernel SVM': sigmoid_svm_pred
}

for model_name, predictions in models.items():
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)

    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)
    print("\n" + "="*40 + "\n")

