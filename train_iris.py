from sklearn import datasets
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# Load Data
iris = datasets.load_iris()

# Seperating the data into dependent and independent variables
X = iris.data
y = iris.target

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


print("Training with iris dataset using SVC...\n")
classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print("CLASSIFICATION REPORT")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
print(confusion_matrix(y_test, y_pred))

# Accuracy score
print('Overall accuracy is',accuracy_score(y_pred, y_test)*100, '%')


