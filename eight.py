import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#Loading data
data = pd.read_csv('creditcard.csv')

#Check for null values
print(data.isnull().sum())

#Normalizing data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

#Creating Model
kmeans = KMeans(n_clusters=3, init='k-means++')
kmeans.fit(data_scaled)

#Predicting clusters
data['cluster_id'] = kmeans.fit_predict(data_scaled)

#Visualizing clusters
plt.scatter(data_scaled[data['cluster_id'] == 0, 0], data_scaled[data['cluster_id'] == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(data_scaled[data['cluster_id'] == 1, 0], data_scaled[data['cluster_id'] == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(data_scaled[data['cluster_id'] == 2, 0], data_scaled[data['cluster_id'] == 2, 1], s = 100, c = 'green', label = 'Cluster 3')

#Plotting the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
