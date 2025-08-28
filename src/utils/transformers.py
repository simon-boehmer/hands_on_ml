# Third-party
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
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
