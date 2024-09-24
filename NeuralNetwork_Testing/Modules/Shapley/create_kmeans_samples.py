from sklearn.cluster import KMeans

def create_kmeans_samples(X, clusters=10, random_state=0):
    # Use KMeans to create a background dataset which represents the whole dataset
    kmeans = KMeans(n_clusters=clusters, random_state=random_state).fit(X)
    background = kmeans.cluster_centers_
    print(background.shape)
    return background