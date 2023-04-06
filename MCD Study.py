import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.utils import resample

df=pd.read_csv(r'D:/Users/AW0811/Desktop/Feynn Labs Internship/Project 2 study task/Code replication/mcdonalds.csv')
print(list(df.columns.values))
print(df.head())
print(df.size)
MDB = df.iloc[:, 0:11].values.astype(str)
MDB1 = (MDB == "Yes").astype(int)
A=np.round(np.mean(MDB1, axis=0), 2)
print(A)
pca = PCA(n_components=11)
pca1 = pca.fit_transform(MDB1)
cols = ['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10','pc11']
pca2 = pd.DataFrame(data = pca1, columns = cols)
print(pca2)

#plotting data in scatter plot
plt.scatter(pca1[:, 0], pca1[:, 1], c='blue')
plt.xlabel('PC1')
plt.ylabel('PC2')
#plotting projection axes
plt.plot([0, pca.components_[0][0]], [0, pca.components_[0][1]], color='red')
plt.plot([0, pca.components_[1][0]], [0, pca.components_[1][1]], color='green')
plt.show()

np.random.seed(1234)

# Perform clustering 
n_clusters_range = range(2, 9)
scores = []
for n_clusters in n_clusters_range:
    km = KMeans(n_clusters=n_clusters, random_state=0)
    km.fit(pca1)
    scores.append(km.inertia_)

# Plotting the results
plt.plot(n_clusters_range, scores, marker='o')
plt.xlabel('Number of segments')
plt.ylabel('sum of within cluster distances')
plt.show()

# Define the number of clusters and bootstrap iterations
n_clusters_range = range(2, 9)
n_bootstrap_iterations = 100

# Perform bootstrapped clustering
adjusted_rand_scores = []
for n_clusters in n_clusters_range:
    cluster_labels = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(pca1)
    bootstrap_adjusted_rand_scores = []
    for i in range(n_bootstrap_iterations):
        X_resampled, cluster_labels_resampled = resample(pca1, cluster_labels, random_state=i)
        bootstrap_adjusted_rand_scores.append(adjusted_rand_score(cluster_labels, cluster_labels_resampled))
    adjusted_rand_scores.append(np.mean(bootstrap_adjusted_rand_scores))

# Plot the results
plt.plot(n_clusters_range, adjusted_rand_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Adjusted Rand Index')
plt.show()

# Reverse order of table
print(df["Like"].value_counts(sort=False)[::-1])

# Create new variable Like.n
df["Like.n"] = 6 - pd.to_numeric(df["Like"], errors='coerce')

# Tabulate Like.n
print(df["Like.n"].value_counts())

f = pd.value_counts(df['Like.n']).sort_index()

print(f)