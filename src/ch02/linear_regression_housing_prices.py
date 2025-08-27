# Standard library
from pathlib import Path
from zlib import crc32

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Local
from utils import plot_housing_map, plot_scatter


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


if __name__ == "__main__":
    main()
