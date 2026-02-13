import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


kidney_disease_csv = pd.read_csv(r"C:\DATA221\Assignments\Assignment3\kidney_disease.csv")
kidney_disease_df = kidney_disease_csv.dropna()
x = kidney_disease_df.drop(columns=["classification"])
y = kidney_disease_df["classification"]
x = pd.get_dummies(x)

split = train_test_split(x, y, test_size=0.3, random_state=42)
x_train = split[0]
x_test = split[1]
y_train = split[2]
y_test = split[3]

#print(x_train.dtypes)
#print(x_train.isnull().sum())


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_prediction = knn.predict(x_test)

new_confusion_matrix = confusion_matrix(y_test, y_prediction)
print(new_confusion_matrix)
print("accuracy: ", accuracy_score(y_test, y_prediction))

new_precision = precision_score(y_test, y_prediction, pos_label="ckd")
print("precision: ", new_precision)

new_recall = recall_score(y_test, y_prediction, pos_label="ckd")
print("recall: ", new_recall)

f_score = f1_score(y_test, y_prediction, pos_label="ckd")
print("f1 score: ", f_score)


"""
Negatives and Positive:
- In context of this exercise, a true positive occurs when
someone affected with the disease is predicted to have the
disease, and a false positive is when a person is predicted
to have the disease but doesnt actually have it. Similarly, 
a true negative is when someone is predicted to not have
the disease and does not have it, and a false negative is
when someone is predicted to not have the disease but actually
does have it.

Accuracy:
- Accuracy is simply represented by the equation: 
            (All true predictions)/(All predictions)
If the sample has an imbalanced amount of features,
the accuracy can simply take the accuracy of the m
feature that represents the majority.

Important metric:
- If missing a kidney disease case is very serious,
the most important metric would be false positives.
With a high recall value, the model flags most people
that actually have the disease, leaving out all false
positives. In context, this could lead to serious if
not fatal consequences.


"""