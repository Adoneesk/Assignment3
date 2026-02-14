import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

kidney_disease_csv = pd.read_csv(r"C:\DATA221\Assignments\Assignment3\kidney_disease.csv")
kidney_disease_df = kidney_disease_csv.replace(["?", "NA","N/A","None", "nan", ""], np.nan)
kidney_disease_df = kidney_disease_df.dropna()

x = kidney_disease_df.drop("classification", axis = 1)
y = kidney_disease_df["classification"]
x = pd.get_dummies(x)

split = train_test_split(x, y, test_size=0.3, random_state=42)
x_train = split[0]
x_test = split[1]
y_train = split[2]
y_test = split[3]

values = [1, 3, 5, 7, 9]
results = []
for i in values:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    y_predicted = knn.predict(x_test)
    new_accuracy = accuracy_score(y_test, y_predicted)
    results.append((i, new_accuracy))

for i, j in results:
    print(f"{i:7}|{j:.4f}")
most_valuable_value = max(results, key = lambda x:x[1])
best_accuracy = max(results, key = lambda x:x[1])
print("Best K value: ", most_valuable_value)
print("Best accuracy: ", best_accuracy)

