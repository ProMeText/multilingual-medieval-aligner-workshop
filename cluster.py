from sklearn.cluster import AgglomerativeClustering
import pandas as pd

# https://stackoverflow.com/a/47333211
distance_matrix = pd.read_csv('result_dir/absence_matrix.csv', index_col=0)
data_matrix = distance_matrix.values
print(data_matrix)
model = AgglomerativeClustering(affinity='precomputed', n_clusters=4, linkage='complete').fit(data_matrix)
print(model.labels_)