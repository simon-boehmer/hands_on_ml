# Third-party
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils.validation import check_array, check_is_fitted


class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True):
        # Store initialization parameter (like sklearn: controls centering)
        self.with_mean = with_mean

    def fit(self, X, y=None):
        # y must be accepted for sklearn compatibility, even if unused
        X = check_array(X)  # ensure X is a valid array of floats (no NaNs, infs)

        # Compute column-wise statistics
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)

        # Convention: estimators must store n_features_in_ during fit()
        self.n_features_in_ = X.shape[1]

        # Always return self to allow chaining (fit_transform pattern)
        return self

    def transform(self, X):
        # Verify that fit() has been called (looks for attributes with trailing "_")
        check_is_fitted(self)

        # Validate new data
        X = check_array(X)

        # Sanity check: number of features must match training
        assert self.n_features_in_ == X.shape[1]

        # Optionally center by mean, then scale by std deviation
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        # n_clusters: how many centroids to learn with KMeans
        # gamma: RBF kernel width (larger = narrower bumps)
        # random_state: reproducibility for KMeans
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        # Validate input and fit KMeans to learn cluster centers
        X = check_array(X)
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            n_init=10,
            random_state=self.random_state,
        )
        self.kmeans_.fit(X, sample_weight=sample_weight)
        self.n_features_in_ = X.shape[1]  # track number of features
        return self  # sklearn convention

    def transform(self, X):
        # Check that fit() was called and input has right shape
        check_is_fitted(self, attributes=["kmeans_", "n_features_in_"])
        X = check_array(X)
        assert X.shape[1] == self.n_features_in_

        # Compute similarity of each sample to each cluster center
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, input_features=None):
        # Generate stable column names for transformed features
        return [f"cluster_{i}_similarity" for i in range(self.n_clusters)]
