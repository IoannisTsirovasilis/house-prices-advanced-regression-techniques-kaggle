from pandas import read_csv, concat, get_dummies, DataFrame
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
#import seaborn as sn

# Task A
# train = read_csv("files/train.csv")
# test = read_csv("files/test.csv")
# train = train.drop(["SalePrice"], axis=1)
# merge = concat([train, test])
merge = read_csv("files/merge.csv")
merge.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence'])
print(merge)
# Task B
# count missing values per column

# Task C
processed_data_set = merge
print(processed_data_set.describe())

# Rank A
correlation = processed_data_set.corr()
print(correlation)
sum_correlation = correlation.sum()
print(sum_correlation.sort_values(ascending=False))

# Rank B
stds = processed_data_set.std()
print(stds.sort_values(ascending=False))

# PCA
# https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
pca = PCA(.95)
pca.fit(processed_data_set)
print(pca.components_)

# Normalization
min_max_scaler = MinMaxScaler()
processed_data_set = min_max_scaler.fit_transform(processed_data_set)
processed_data_set = DataFrame(processed_data_set)

# Rank C
correlation = processed_data_set.corr()
print(correlation)
sum_correlation = correlation.sum()
print(sum_correlation.sort_values(ascending=False))

# Rank D
stds = processed_data_set.std()
print(stds.sort_values(ascending=False))

# PCA
pca = PCA(.95)
pca.fit(processed_data_set)
print(pca.components_)



