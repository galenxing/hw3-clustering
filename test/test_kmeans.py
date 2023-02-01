# Write your k-means unit tests here
# write tests for bfs
import pytest
import sklearn.metrics
import numpy as np
from numpy.testing import assert_almost_equal

from cluster import (
        KMeans, 
        Silhouette, 
        make_clusters)

def test_kmeans():
    """
    Testing Kmeans class
    """
    clusters, labels = make_clusters(k=3, scale=1)
    km = KMeans(k=3)
    km.fit(clusters)
    pred = km.predict(clusters)
    
    assert len(np.unique(pred)) == 3
    
    km = KMeans(k=2)
    km.fit(clusters)
    pred = km.predict(clusters)
    
    assert len(np.unique(pred)) == 2
    
    
        
    assert km.get_error() > 0 
    assert km.get_error() == km.error
    
    assert km.get_centroids().shape == (2, clusters.shape[1])

    with pytest.raises(Exception):
        km = KMeans(k=0)

def test_silhouette():
    clusters, labels = make_clusters(k=3, scale=1)
    km = KMeans(k=3)
    km.fit(clusters)
    pred = km.predict(clusters)
    
    scores = Silhouette().score(clusters, pred)
    
    sk_sil = sklearn.metrics.silhouette_score(clusters, pred)
    
    assert_almost_equal(np.mean(scores), np.mean(sk_sil), 2)
    
    