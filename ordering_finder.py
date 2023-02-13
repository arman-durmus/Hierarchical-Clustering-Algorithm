import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial import distance
import random
import scipy.cluster.hierarchy as sch

def sch_ordering(data):
    z = sch.linkage(data, method="ward")
    dendrogram = sch.dendrogram(z, no_plot=True)
    indices = dendrogram["leaves"]
    return data.iloc[indices, :]

def rand_ord(df):
    return df.sample(frac=1).reset_index(drop=True)

def create_ordering(data, n_clusters = 3):
    """"
    Creates a ordering by finding a clustering , then sorting values according to that clustering.
    Credit: created with the help of ChatGPT.
    """
    # Use the k-means algorithm to identify clusters of similar data points
    kmeans = KMeans(n_clusters=n_clusters).fit(data)
    
    # Create an ordering for the data points by sorting them according to their cluster assignment
    data['cluster'] = kmeans.labels_
    centroids = kmeans.cluster_centers_
    # To order the clusters against each other, sort the clusters by the mean or median value of the points in each cluster
    def get_dist(row):
        return distance.euclidean(row.drop("cluster"), centroids[int(row['cluster'])])
    data['distance_to_centroid'] = data.apply(lambda row: get_dist(row), axis=1)
    
    # To order the points within each cluster, sort the points within each cluster by their distance from the cluster centroid
    data = data.sort_values(by=['cluster', 'distance_to_centroid'])
    data = data.drop(columns=['cluster', 'distance_to_centroid'])

    return data

def pca_ordering(data):
    """
    Performs PCA to reduce Data to 1 dimension, then returns the data sorted by that dimension.
    """
    pca = PCA(n_components=1)
    data["order"] = pca.fit_transform(data.drop('target', axis=1))
    data = data.sort_values(axis=0, by="order")
    data = data.drop(columns=["order"])
    return data

def tsne_ordering(data):
    """
    Performs tSNE to reduce Data to 1 dimension, then returns the data sorted by that dimension.
    """
    tsne = TSNE(n_components=1)
    data["order"] = tsne.fit_transform(data.drop('target', axis=1))
    data = data.sort_values(axis=0, by="order")
    data = data.drop(columns=["order"])
    return data

def greedy_ordering(data):
    """
    Take first randomly, then take the closest point in the dataset as the next.
    """
    start_idx = random.randint(0, len(data)-1)
    temp = data.copy()
    prev = temp.iloc[start_idx]
    data["order"] = [0 for _ in range(len(data))]
    data.iloc[start_idx, -1] = 1
    temp = temp.drop([start_idx]).reset_index(drop=True)

    for i in range(2, len(data)+1):
        min_dist = float('inf')
        for j in range(len(temp)):
            d = distance.euclidean(temp.iloc[j], prev)
            if d < min_dist:
                min_dist = d
                min_dist_idx = j
        data.iloc[min_dist_idx+1-i, -1] = i
        temp = temp.drop([min_dist_idx,], axis = 0).reset_index(drop=True)
    
    data = data.sort_values(axis=0, by="order")
    data = data.drop(columns=["order"])
    
    return data
