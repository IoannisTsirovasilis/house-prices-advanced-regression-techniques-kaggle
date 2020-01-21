from pandas import read_csv, concat
import matplotlib.pyplot as plt
import seaborn as sn

# Task A
# train = read_csv("files/train.csv")
# test = read_csv("files/test.csv")
# train = train.drop(["SalePrice"], axis=1)
# merge = concat([train, test])
merge = read_csv("files/merge.csv")

# Task B
# count missing values per column
missing_values = merge.apply(lambda x: x.count()/2919, axis=0)
sorted_mv = missing_values.sort_values()
print(sorted_mv[:])
print(sorted_mv[sorted_mv[0] < 1])
