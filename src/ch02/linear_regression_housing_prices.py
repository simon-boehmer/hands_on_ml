from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from zlib import crc32

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


def main() -> None:
    # Load dataset
    housing = pd.read_csv(csv_path)

    # Inspect dataset
    print(housing.head())
    print(housing.info())
    print(housing.describe())

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

    # Default: stratified split
    train, test = split_train_test_stratified(housing)
    print(f"strat train: {len(train)}, strat test: {len(test)}")


if __name__ == "__main__":
    main()
