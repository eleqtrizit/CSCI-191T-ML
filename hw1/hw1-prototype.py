# %%
from sklearn.model_selection import train_test_split
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

dataPath = "d:/Dropbox (Personal)/Documents/School/CSCI ML/CSCI-191T-ML/handson-ml/datasets/housing/"

# %%
housing = pd.read_csv(dataPath + "housing.csv")
print(housing.info())

# %%
print(housing.head(4))
print(housing.tail(4))


# %%
housing["median_house_value"].hist()


# %%
print(housing["ocean_proximity"].value_counts())


# %%
train_set, test_set = train_test_split(
    housing, test_size=0.2, random_state=42)

print(train_set.describe())
print("\nTraining Percentages:\n")
print(train_set.count()/housing.count())
