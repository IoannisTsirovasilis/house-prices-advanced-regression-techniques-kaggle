from pandas import read_csv, concat
import matplotlib.pyplot as plt
import seaborn as sn

# Task A
train = read_csv("files/train.csv")
test = read_csv("files/test.csv")
train = train.drop(["SalePrice"], axis=1)
merge = concat([train, test])
print(merge.describe())

# Task B
