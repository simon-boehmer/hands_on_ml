# Standard library
from pathlib import Path
from zlib import crc32

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.compose import (
    TransformedTargetRegressor,
    ColumnTransformer,
    make_column_transformer,
    make_column_selector,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

# Local
from utils import plot_housing_map, plot_scatter, StandardScalerClone, ClusterSimilarity


# Resolve repo paths
repo_root = Path(__file__).resolve().parents[2]
data_dir = repo_root / "data" / "sample"
csv_path = data_dir / "housing.csv"


# Split option 1: sklearn utility
def split_train_test(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(
        df, test_size=test_size, stratify=df["income_cat"], random_state=random_state
    )


# Split option 2: manual random shuffle
def split_train_test_manual(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(seed)
    shuffled = np.random.permutation(len(df))
    test_size_abs = int(len(df) * test_size)
    test_indices = shuffled[:test_size_abs]
    train_indices = shuffled[test_size_abs:]
    return df.iloc[train_indices], df.iloc[test_indices]


# Split option 3: stable split by ID (avoids reshuffling issues)
def is_in_test_set(identifier: int | float, test_size: float, hash=crc32) -> bool:
    return hash(np.int64(identifier)) & 0xFFFFFFFF < test_size * 2**32


def split_train_test_by_id(
    df: pd.DataFrame, test_size: float, id_column: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = df[id_column]
    in_test_set = ids.apply(lambda id_: is_in_test_set(id_, test_size))
    return df.loc[~in_test_set], df.loc[in_test_set]


# Split option 4: stratified split by income category (preferred)
def split_train_test_stratified(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_index, test_index = next(splitter.split(df, df["income_cat"]))
    strat_train = df.iloc[train_index].copy()
    strat_test = df.iloc[test_index].copy()

    # print(strat_test["income_cat"].value_counts() / len(strat_test))

    # Drop helper column to avoid leakage
    for split in (strat_train, strat_test):
        split.drop(columns="income_cat", inplace=True)

    return strat_train, strat_test


# Return a fitted imputer for numerical attributes
def get_imputer(strategy: str = "median") -> SimpleImputer:
    return SimpleImputer(strategy=strategy)


# Fit ordinal encoder on categorical columns
def encode_ordinal(df: pd.DataFrame) -> OrdinalEncoder:
    encoder = OrdinalEncoder()
    encoder.fit(df)
    return encoder


# Fit one-hot encoder on categorical columns
def encode_onehot(df: pd.DataFrame) -> OneHotEncoder:
    encoder = OneHotEncoder(sparse_output=True)  # keep sparse for efficiency
    encoder.fit(df)
    return encoder


# Fit a MinMaxScaler on the DataFrame
def scale_minmax(df: pd.DataFrame) -> MinMaxScaler:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(df)
    return scaler


# Fit a StandardScaler on the DataFrame
def scale_std(df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df)
    return scaler


def main() -> None:
    # Load dataset
    housing = pd.read_csv(csv_path)

    # Inspect dataset structure + stats
    print(housing.head())  # first rows
    print(housing.info())  # column info + dtypes + nulls
    print(housing.describe())  # summary stats

    # Discretize median_income into categories for stratified sampling
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0, 1.5, 3, 4.5, 6, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    # Split option 3: stable split by ID (avoids reshuffling issues)
    # housing_with_id = housing.copy()
    # housing_with_id["id"] = housing_with_id["longitude"] * 1000 + housing_with_id["latitude"]
    # train, test = split_train_test_by_id(housing_with_id, 0.2, "id")

    # Default: stratified split (preferred)
    train, test = split_train_test_stratified(housing)
    print(f"strat train: {len(train)}, strat test: {len(test)}")

    # Copy train set for feature engineering
    housing = train.copy()

    # Add engineered features
    housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["people_per_house"] = housing["population"] / housing["households"]

    # Correlation matrix (numerical only)
    corr_matrix = housing.corr(numeric_only=True)
    print(corr_matrix["median_house_value"].sort_values(ascending=False))

    # Visual exploration examples (commented out)
    # attributes = [
    #     "median_house_value",
    #     "median_income",
    #     "total_rooms",
    #     "housing_median_age",
    # ]
    # scatter_matrix(housing[attributes], figsize=(20, 20))
    # plt.show()

    # plot_scatter(
    #     housing["median_income"],
    #     housing["median_house_value"],
    #     x_label="Median Income",
    #     y_label="Median House Value",
    #     title="?",
    # )

    # Separate predictors and labels
    housing = train.drop("median_house_value", axis=1)
    housing_labels = train["median_house_value"].copy()

    # Handling missing values (different options below)
    # Option 1: drop rows with missing values
    # housing.dropna(subset=["total_bedrooms"], inplace=True)
    # Option 2: drop the whole column
    # housing.drop("total_bedrooms", axis=1)
    # Option 3: fill with median value
    # total_bedrooms_median = housing["total_bedrooms"].median()
    # housing["total_bedrooms"].fillna(total_bedrooms_median, inplace=True)

    # Preferred: imputation with sklearn
    imputer = get_imputer("median")
    housing_num = housing.select_dtypes(include=[np.number])
    imputer.fit(housing_num)

    print(imputer.statistics_)  # learned medians for each column

    # Transform the dataset (replace missing values with medians)
    X = imputer.transform(housing_num)

    # Back to DataFrame with original index + column names
    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
    print(housing_tr.info())

    # Categorical attributes (non-numeric columns)
    housing_cat = housing.select_dtypes(include=["object"])

    # Ordinal encoding
    ordinal_encoder = encode_ordinal(housing_cat)
    housing_cat_encoded = ordinal_encoder.transform(housing_cat)
    # print(ordinal_encoder.categories_)

    # One-hot encoding
    onehot_encoder = encode_onehot(housing_cat)
    housing_cat_onehot = onehot_encoder.transform(housing_cat)
    # print(housing_cat_onehot[:5])
    # print(housing_cat_onehot.toarray())  # toarray only for inspection

    scaler = scale_std(housing_num)
    housing_scaled = scaler.transform(housing_num)
    print(housing_scaled)

    # Transform a multimodal distributions using RBF (TEST)
    age_similarity_35 = rbf_kernel(housing[["housing_median_age"]], [[15]], gamma=0.1)

    # Scatter: age vs similarity
    plt.figure(figsize=(8, 5))
    plt.scatter(housing["housing_median_age"], age_similarity_35, alpha=0.1)
    plt.xlabel("Housing Median Age")
    plt.ylabel("Similarity to Age=35")
    plt.title("RBF Similarity to Age=35")
    plt.grid(True, linestyle=":")
    plt.show()

    # Scale target values (labels) before training
    target_scaler = StandardScaler()
    scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())

    model = LinearRegression()
    model.fit(housing[["median_income"]], scaled_labels)

    # Take some new data (first 5 rows of median_income)
    some_new_data = housing[["median_income"]].iloc[:5]

    # Predict in scaled target space
    scaled_predictions = model.predict(some_new_data)
    print(scaled_predictions)

    # Inverse-transform predictions back to original scale
    predictions = target_scaler.inverse_transform(scaled_predictions)
    print(predictions)

    # Alternative: let sklearn handle target scaling internally
    model = TransformedTargetRegressor(LinearRegression(), transformer=StandardScaler())
    model.fit(housing[["median_income"]], housing_labels)
    predictions = model.predict(some_new_data)
    print(predictions)

    # Custom log transformer (with inverse = exp)
    log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
    log_population = log_transformer.transform(housing[["population"]])

    # Custom RBF similarity transformer
    # rbf_kernel(X, Y) needs both arguments; we fix Y=[[15.]], gamma=0.1 via kw_args
    rbf_transformer = FunctionTransformer(
        rbf_kernel, kw_args=dict(Y=[[15.0]], gamma=0.1)
    )
    age_similarity_35 = rbf_transformer.transform(housing[["housing_median_age"]])

    # Fit and apply custom StandardScalerClone on median_income
    scaler = StandardScalerClone(with_mean=True)
    scaler.fit(housing[["median_income"]])
    scaled = scaler.transform(housing[["median_income"]])

    print(scaled[:5])  # show first 5 scaled values

    # Numerical pipeline: impute missing values, then standardize
    num_pipeline = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    # Equivalent shorthand using make_pipeline (auto-names steps)
    num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    # Fit + transform numerical features
    housing_num_prepared = num_pipeline.fit_transform(housing_num)
    print(housing_num_prepared[:2].round(2))

    # Back to DataFrame with feature names + original index
    df_housing_num_prepared = pd.DataFrame(
        housing_num_prepared,
        columns=num_pipeline.get_feature_names_out(),
        index=housing_num.index,
    )
    print(df_housing_num_prepared.head(2).round(2))

    # Define numerical and categorical attribute lists
    num_attribs = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
    ]
    cat_attribs = ["ocean_proximity"]

    # Categorical pipeline: impute missing with most_frequent, then one-hot encode
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")
    )

    # Combine numerical + categorical pipelines
    preprocessing = ColumnTransformer(
        [("num", num_pipeline, num_attribs), ("cat", cat_pipeline, cat_attribs)]
    )

    # Easier approach: select columns by dtype instead of listing names
    preprocessing = make_column_transformer(
        (num_pipeline, make_column_selector(dtype_include=np.number)),
        (cat_pipeline, make_column_selector(dtype_include=object)),
    )

    # Fit + transform full dataset (numerical + categorical)
    housing_prepared = preprocessing.fit_transform(housing)

    # Get feature names from the preprocessing pipeline
    feature_names = preprocessing.get_feature_names_out()

    # Wrap transformed data back into DataFrame
    df_housing_prepared = pd.DataFrame(
        housing_prepared,
        columns=feature_names,
        index=housing.index,  # keep original row index
    )

    print(df_housing_prepared.head())

    # Full preprocessing pipeline
    # Includes: engineered ratios, log transforms, cluster-based geo features,
    # categorical encoding, and default numeric impute+scale for leftovers

    # Custom transformer: compute ratio of two columns
    def column_ratio(X):
        return X[:, [0]] / X[:, [1]]

    # Provide a feature name for the ratio output (sklearn requires this)
    def ratio_name(function_transformer, feature_names_in):
        return ["ratio"]

    # Pipeline to create ratio features: impute missing → compute ratio → scale
    def ratio_pipeline():
        return make_pipeline(
            SimpleImputer(strategy="median"),
            FunctionTransformer(column_ratio, feature_names_out=ratio_name),
            StandardScaler(),
        )

    # Pipeline for log-transformed features: impute missing → log-transform → scale
    log_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler(),
    )

    # Geo-feature transformer: replace lat/long with cluster similarities
    cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1.0, random_state=42)

    # Default pipeline for any remaining numeric columns:
    # impute missing values with median, then standard scale
    default_num_pipeline = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler()
    )

    # Combine all transformations into one ColumnTransformer
    preprocessing = ColumnTransformer(
        [
            # Ratios: new features engineered from column pairs
            ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
            ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
            ("people_per_house", ratio_pipeline(), ["population", "households"]),
            # Log transformations: applied to skewed numeric columns
            (
                "log",
                log_pipeline,
                [
                    "total_bedrooms",
                    "total_rooms",
                    "population",
                    "households",
                    "median_income",
                ],
            ),
            # Geographic features: cluster similarity from latitude/longitude
            ("geo", cluster_simil, ["latitude", "longitude"]),
            # Categorical pipeline: impute missing with most_frequent, then one-hot encode
            ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
        ],
        # Any other numeric column not listed (e.g. housing_median_age) → default pipeline
        remainder=default_num_pipeline,
    )

    housing_prepared = preprocessing.fit_transform(housing)
    print(housing_prepared.shape)
    print(preprocessing.get_feature_names_out())


if __name__ == "__main__":
    main()
