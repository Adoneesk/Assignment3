import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

kidney_disease_csv = pd.read_csv(r"C:\DATA221\Assignments\Assignment3\kidney_disease.csv")
x = kidney_disease_csv.drop(columns=["classification"])
y = kidney_disease_csv["classification"]

split = train_test_split(x, y, test_size=0.3, random_state=42)
x_test = split[1]
x_train = split[0]
y_test = split[3]
y_train = split[2]

"""
Training and Testing on the Same Model:
- We should not train and split on the same model
because the model can accidentally use the training
examples instead of actually learning general
patterns. In general, this will lead to overfitting.

Test Set:
- The purpose of the test set is to evaluate the
performance of the trained model. Helps direct the
model to more accurate predictions.


"""

