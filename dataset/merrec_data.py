# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import os
from itertools import chain
from typing import Dict, List, Tuple

import dask
import dask.dataframe as dd

import duckdb
import numpy as np
import pandas as pd
import polars as pl
from dask.diagnostics import ProgressBar

from datasets import load_dataset

dask.config.set({"dataframe.convert-string": False})


def init_duckdb_conn(threads: int = 64) -> duckdb.DuckDBPyConnection:
    # Create a connection to DuckDB in memory
    conn = duckdb.connect(database=":memory:")
    # Set number of threads for parallel processing
    conn.execute(f"PRAGMA threads = {threads}")
    conn.execute("PRAGMA enable_progress_bar")
    return conn


feature_cols: Dict[str, str] = {
    "name": "text",
    "category_name": "text",
    "brand_name": "text",
    "category_id": "int64",
    "brand_id": "int64",
    "item_id": "int64",
    "product_id": "text",
    "stime": "int64",
    "event_id": "text",
    "price": "float8",
    "item_condition_name": "text",
    "size_name": "text",
    "color": "text",
    "shipper_name": "text",
}


def rebuild_sequence_dataset_duck(
    output_parquet_file: str,
    hf_dataset_name: str = "mercari-us/merrec",
    split: str = "train",
    user_id_col: str = "user_id",
    sequence_id_col: str = "sequence_id",
    timestamp_col: str = "stime",
    c0_name_col: str = "c0_name",
    c1_name_col: str = "c1_name",
    c2_name_col: str = "c2_name",
    c0_id_col: str = "c0_id",
    c1_id_col: str = "c1_id",
    c2_id_col: str = "c2_id",
) -> None:
    """
    Rebuilds a sequence dataset using DuckDB.

    Args:
        output_parquet_file (str): The path to the output Parquet file.
        hf_dataset_name (str, optional): The name of the Hugging Face dataset. Defaults to "mercari-us/merrec".
        split (str, optional): The split of the dataset. Defaults to "train".
        user_id_col (str, optional): The column name for the user ID. Defaults to "user_id".
        sequence_id_col (str, optional): The column name for the sequence ID. Defaults to "sequence_id".
        timestamp_col (str, optional): The column name for the timestamp. Defaults to "stime".
        c0_name_col (str, optional): The column name for category level 0. Defaults to "c0_name".
        c1_name_col (str, optional): The column name for category level 1. Defaults to "c1_name".
        c2_name_col (str, optional): The column name for category level 2. Defaults to "c2_name".
        c0_id_col (str, optional): The column name for category ID level 0. Defaults to "c0_id".
        c1_id_col (str, optional): The column name for category ID level 1. Defaults to "c1_id".
        c2_id_col (str, optional): The column name for category ID level 2. Defaults to "c2_id".

    Returns:
        None
    """
    # Load the Hugging Face dataset and get the Arrow table directly
    dataset = load_dataset(hf_dataset_name, split=split)
    table = dataset.data.table  # pyre-ignore[16]

    # Create a connection to DuckDB in memory
    conn = init_duckdb_conn(threads=64)
    conn.register("sequence_table", table)

    original_cols: List[str] = [
        user_id_col,
        "name",
        "brand_name",
        "brand_id",
        "item_id",
        "product_id",
        "event_id",
        "price",
        "item_condition_name",
        "size_name",
        "color",
        "shipper_name",
    ]

    conn.execute(
        f"""
        CREATE VIEW ordered_sequence_table AS 
        SELECT {', '.join(original_cols)}, 
                coalesce({c2_name_col}, {c1_name_col}, {c0_name_col}) AS category_name,
                coalesce({c2_id_col}, {c1_id_col}, {c0_id_col}) AS category_id,
                epoch({timestamp_col}) AS {timestamp_col}
        FROM sequence_table 
        """
    )

    row = f"ROW({', '.join(feature_cols.keys())})"
    feature_def = [f"{key} {value}" for key, value in feature_cols.items()]
    struct = f"ROW({', '.join(feature_def)})"
    query = f"""
        SELECT {user_id_col}, ARRAY_AGG(CAST({row} AS {struct}) ORDER BY {timestamp_col} ASC) AS uih
        FROM ordered_sequence_table 
        GROUP BY {user_id_col}
        """
    print(f"query: {query}")
    output_parquet_file = os.path.expanduser(output_parquet_file)
    conn.sql(query).to_parquet(output_parquet_file, compression=None)


def split_user_sequence_dataset_duck(
    src_parquet_file: str,
    output_parquet_file: str,
    user_id_col: str = "user_id",
    uih_col: str = "uih",
    split_length: int = 500,
) -> None:
    """
    Splits a user sequence dataset into chunks of a specified length.

    Args:
        src_parquet_file (str): The path to the source Parquet file.
        output_parquet_file (str): The path to the output Parquet file.
        user_id_col (str, optional): The column name for the user ID. Defaults to "user_id".
        uih_col (str, optional): The column name for the user interaction history. Defaults to "uih".
        split_length (int, optional): The length of each chunk. Defaults to 500.

    Returns:
        None
    """
    parquet_file = os.path.expanduser(src_parquet_file)
    query = f"""
        WITH filtered AS (
            SELECT 
                {user_id_col},
                {uih_col},
                ARRAY_LENGTH({uih_col}) AS uih_len
            FROM read_parquet('{parquet_file}')
            WHERE ARRAY_LENGTH({uih_col}) >= {split_length} -- Filter rows with fewer items upfront
        ),
        range_values AS (
            SELECT 
                {user_id_col},
                {uih_col},
                uih_len,
                UNNEST(RANGE(1, CAST(FLOOR(uih_len / {split_length}) AS INTEGER) + 1)) AS i -- Generate range of indices
            FROM filtered
        ),
        chunked AS (
            SELECT 
                {user_id_col},
                list_slice(
                    {uih_col},
                    uih_len - (i * {split_length}) + 1,
                    uih_len - ((i - 1) * {split_length})
                ) AS uih
            FROM range_values
        )
        SELECT *
        FROM chunked
        WHERE ARRAY_LENGTH(uih) = {split_length}    
    """
    print(f"query: {query}")
    output_parquet_file = os.path.expanduser(output_parquet_file)

    conn = init_duckdb_conn(threads=64)
    conn.sql(query).to_parquet(output_parquet_file, compression=None)


def build_llm_dataset(
    interaction_output: str | None = None,
    item_output: str | None = None,
    hf_dataset_name: str = "mercari-us/merrec",
    split: str = "train",
    user_id_col: str = "user_id",
    event_time_col: str = "stime",
    event_id_col: str = "event_id",
    event_id_mapping: Dict[str, int] | None = None,
    object_id_col: str = "product_id",
    text_cols: Tuple[str, ...] = ("brand_name", "category_name"),
    c0_name_col: str = "c0_name",
    c1_name_col: str = "c1_name",
    c2_name_col: str = "c2_name",
    min_seq_len: int = 1024,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    # Load the Hugging Face dataset and get the Arrow table directly
    dataset = load_dataset(hf_dataset_name, split=split)
    table = dataset.data.table  # pyre-ignore[16]

    # Create a connection to DuckDB in memory
    conn = init_duckdb_conn(threads=64)
    conn.register("sequence_table", table)

    cols = [col for col in text_cols if col != "category_name"]
    final_cols = text_cols + ("event_time", "event_id")
    query = f"""
    WITH sorted_data AS (
        SELECT {user_id_col} AS user_id, {object_id_col} AS object_id, {', '.join(cols)}, 
                coalesce({c2_name_col}, {c1_name_col}, {c0_name_col}) AS category_name,
                CAST(epoch({event_time_col}) AS INT) AS event_time,
                {event_id_col} AS event_id
        FROM sequence_table
        ORDER BY user_id, event_time ASC
    ),
    user_counts AS (
        SELECT user_id, COUNT(*) AS user_count
        FROM sorted_data
        GROUP BY user_id
    )
    SELECT sd.user_id AS user_id, object_id, {', '.join(final_cols)}
    FROM sorted_data sd
    INNER JOIN user_counts uc
    ON sd.user_id = uc.user_id
    WHERE uc.user_count >= {min_seq_len}
    ORDER BY user_id, event_time ASC
    """
    print(f"query: {query}")

    pdf = conn.execute(query).pl()
    conn.close()

    pdf = pdf.with_columns(
        [
            pdf["user_id"].rank(method="dense").cast(pl.Int32).alias("user_id"),
            pdf["object_id"].rank(method="dense").cast(pl.Int32).alias("object_id"),
        ]
    )
    if event_id_mapping is not None:
        pdf = pdf.with_columns(
            pl.col("event_id")
            .replace(event_id_mapping)
            .cast(pl.Int32)
            .alias("event_id")
        )

    interactions = (
        pdf.select(["user_id", "object_id", "event_time", "event_id"])
        .group_by("user_id", maintain_order=True)
        .agg([pl.col("object_id"), pl.col("event_time"), pl.col("event_id")])
    )
    if interaction_output is not None:
        interaction_output = os.path.expanduser(interaction_output)
        if interaction_output.endswith(".parquet"):
            interactions.write_parquet(interaction_output)
        elif interaction_output.endswith(".csv"):
            interactions.write_csv(interaction_output)
        else:
            raise ValueError(
                f"Unsupported file format for filename: {interaction_output}"
            )

    items = (
        pdf.group_by("object_id")
        .agg([pl.col(col).first().alias(col) for col in text_cols])
        .select(["object_id", *text_cols])
    )
    if item_output is not None:
        item_output = os.path.expanduser(item_output)
        if item_output.endswith(".parquet"):
            items.write_parquet(item_output)
        elif item_output.endswith(".csv"):
            items.write_csv(item_output)
        else:
            raise ValueError(f"Unsupported file format for filename: {item_output}")
    return interactions, items


def split_train_valid_test(
    parquet_file: str,
    output_parquet_file_prefix: str,
    train_frac: float = 0.8,
    valid_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> None:
    """
    Splits a Parquet file into three separate Parquet files for training, validation, and testing.

    Args:
        parquet_file (str): The path to the input Parquet file.
        output_parquet_file_prefix (str): The prefix for the output Parquet files.
        train_frac (float, optional): The fraction of rows to use for training. Defaults to 0.8.
        valid_frac (float, optional): The fraction of rows to use for validation. Defaults to 0.1.
        test_frac (float, optional): The fraction of rows to use for testing. Defaults to 0.1.
        seed (int, optional): The seed for shuffling the data. Defaults to 42.

    Returns:
        None
    """
    parquet_file = os.path.expanduser(parquet_file)
    df = pl.read_parquet(parquet_file)
    # Shuffle the DataFrame to ensure randomness
    df = df.sample(
        fraction=1,
        with_replacement=False,
        seed=42,
        shuffle=True,
    )  # `seed` ensures reproducibility

    # Calculate split indices
    n = len(df)
    train_end = int(n * train_frac)
    valid_end = train_end + int(n * valid_frac)
    # Perform the splits
    train_df = df[:train_end]
    valid_df = df[train_end:valid_end]
    test_df = df[valid_end:]

    # Save each split to separate Parquet files
    output_parquet_file_prefix = os.path.expanduser(output_parquet_file_prefix)
    train_df.write_parquet(file=f"{output_parquet_file_prefix}_train.parquet")
    valid_df.write_parquet(file=f"{output_parquet_file_prefix}_valid.parquet")
    test_df.write_parquet(file=f"{output_parquet_file_prefix}_test.parquet")


def fast_read_parquet(parquet_file: str) -> pd.DataFrame:
    """
    Reads a Parquet file into a Pandas DataFrame using Polars.

    Args:
        parquet_file (str): The path to the Parquet file.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the data from the Parquet file.
    """
    parquet_file = os.path.expanduser(parquet_file)
    # Read the Parquet file into a Polars DataFrame
    pdf = pl.read_parquet(parquet_file)
    # Convert the Polars DataFrame to a Pandas DataFrame, using PyArrow extension arrays
    return pdf.to_pandas(use_pyarrow_extension_array=True)


def sequence_dataset_duck(
    hf_dataset_name: str = "mercari-us/merrec",
    split: str = "train",
    user_id_col: str = "user_id",
    sequence_id_col: str = "sequence_id",
    timestamp_col: str = "stime",
    c0_name_col: str = "c0_name",
    c1_name_col: str = "c1_name",
    c2_name_col: str = "c2_name",
    c0_id_col: str = "c0_id",
    c1_id_col: str = "c1_id",
    c2_id_col: str = "c2_id",
) -> pd.DataFrame:
    """
    This function loads a Hugging Face dataset, processes the data in sequences (ordered by tiimestamp_col),
    and returns a DataFrame containing the processed sequences.

    Parameters:
    hf_dataset_name (str): The name of the Hugging Face dataset to load. Defaults to "mercari-us/merrec".
    split (str): The split of the dataset to use (e.g., 'train', 'test'). Defaults to 'train'.
    user_id_col (str): The column name for identifying users. Defaults to "user_id".
    sequence_id_col (str): The column name for identifying sequences. Defaults to "sequence_id".
    timestamp_col (str): The column name for the timestamp. Defaults to "stime".
    c0_name_col (str): The column name for category level 0. Defaults to "c0_name".
    c1_name_col (str): The column name for category level 1. Defaults to "c1_name".
    c2_name_col (str): The column name for category level 2. Defaults to "c2_name".
    c0_id_col (str): The column name for category ID level 0. Defaults to "c0_id".
    c1_id_col (str): The column name for category ID level 1. Defaults to "c1_id".
    c2_id_col (str): The column name for category ID level 2. Defaults to "c2_id".

    Returns:
    pd.Dataframe: A dataframe containing the processed sequences.
    """
    # Load the Hugging Face dataset and get the Arrow table directly
    dataset = load_dataset(hf_dataset_name, split=split)
    table = dataset.data.table  # pyre-ignore[16]

    # Create a connection to DuckDB in memory
    conn = init_duckdb_conn(threads=64)
    conn.register("sequence_table", table)

    # Create `seq_user_id` and backfill category columns
    sequences = conn.execute(
        f"""
        SELECT *, 
                {user_id_col} || '_' || {sequence_id_col} AS seq_user_id,
                coalesce({c2_name_col}, {c1_name_col}, {c0_name_col}) AS category_name,
                coalesce({c2_id_col}, {c1_id_col}, {c0_id_col}) AS category_id
        FROM sequence_table 
        ORDER BY seq_user_id, {timestamp_col}
        """
    ).fetch_arrow_table()

    # It uses less memory than fetchdf()
    dask_sorted_df = dd.from_pandas(sequences.to_pandas(), npartitions=64)
    # Sepecify the index `seq_user_id` is sorted
    dask_sorted_df = dask_sorted_df.set_index("seq_user_id", sorted=True)

    # Group by 'seq_user_id' and aggregate feature columns
    agg_dict = {col: "list" for col in feature_cols.keys()}
    group_columns = ["seq_user_id", user_id_col, sequence_id_col]
    with ProgressBar():
        grouped_dask_df = (
            dask_sorted_df.groupby(group_columns, sort=False)  # no need to sort
            .agg(agg_dict)  # order preserved
            .compute()
        )
    return grouped_dask_df


def user_sequence_dataset_duck(
    src_parquet_file: str,
    output_split_ratio: float = 0.8,
    user_id_col: str = "user_id",
    sequence_id_col: str = "sequence_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function takes a Parquet file as input, sorts it by user ID and sequence ID,
    groups the data by user ID, concatenates the feature columns for each group,
    shuffles the user IDs, and splits them into two subsets based on the output split ratio.

    Args:
        src_parquet_file (str): The path to the input Parquet file.
        output_split_ratio (float, optional): The ratio of user IDs to include in the first subset. Defaults to 0.8.
        user_id_col (str, optional): The column name for the user ID. Defaults to "user_id".
        sequence_id_col (str, optional): The column name for the sequence ID. Defaults to "sequence_id".

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames, one for each subset of user IDs.
    """

    parquet_file = os.path.expanduser(src_parquet_file)
    conn = init_duckdb_conn(threads=64)

    features: List[str] = [user_id_col] + list(feature_cols.keys())

    query = f"""
        SELECT {', '.join(features)}
        FROM '{parquet_file}'
        ORDER BY {user_id_col}, {sequence_id_col}
    """

    # Execute the query and keep the result as an Arrow Table
    ordered_table = conn.execute(query).fetch_arrow_table()
    # It uses less memory than fetchdf()
    dask_df = dd.from_pandas(ordered_table.to_pandas())
    # Sepecify the index `seq_user_id` is sorted
    dask_sorted_df = dask_df.set_index(user_id_col, sorted=True)
    dask_sorted_df = dask_sorted_df.repartition(npartitions=64)

    def concat(group: pd.DataFrame) -> pd.Series:
        """
        Concatenates lists for the specified feature columns using chain.from_iterable for better performance.

        Args:
            group (pd.DataFrame): A DataFrame group.

        Returns:
            pd.Series: A Series containing the concatenated feature columns.
        """
        return pd.Series(
            {col: list(chain.from_iterable(group[col])) for col in feature_cols.keys()}
        )

    grouped_df = dask_sorted_df.groupby(dask_sorted_df.index).apply(
        concat, meta={col: object for col in feature_cols.keys()}
    )
    result = grouped_df.compute()

    # Shuffle user_ids and split into two subsets
    unique_user_ids = np.array(result.index.unique())
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(unique_user_ids)

    split_idx = int(len(unique_user_ids) * output_split_ratio)

    user_ids_1 = unique_user_ids[:split_idx]
    user_seqs_1 = result.loc[user_ids_1]

    user_ids_2 = unique_user_ids[split_idx:]
    user_seqs_2 = result.loc[user_ids_2]

    return user_seqs_1, user_seqs_2


def seq_len_filter_pd(df: pd.DataFrame, min_seq_len: int = 10) -> pd.DataFrame:
    """
    Filters a Pandas DataFrame based on the length of the 'name' column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        min_seq_len (int, optional): The minimum sequence length to keep. Defaults to 10.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    # Calculate the length of each sequence in the 'name' column.
    lengths = np.array([len(x) for x in df["name"]])
    # Filter the DataFrame to only include rows where the sequence length is greater than or equal to the minimum sequence length.
    df_filtered = df[lengths >= min_seq_len]
    return df_filtered


def seq_len_filter_duck(parquet_file: str, min_seq_len: int = 10) -> pd.DataFrame:
    """
    Filters a Parquet file based on the length of the 'name' column.

    Args:
        parquet_file (str): The path to the Parquet file.
        min_seq_len (int, optional): The minimum sequence length to keep. Defaults to 10.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    parquet_file = os.path.expanduser(parquet_file)
    conn = init_duckdb_conn(threads=64)
    # Execute the query to filter rows based on the length of the 'name' column
    result = conn.execute(
        f"""
        SELECT *
        FROM read_parquet('{parquet_file}')
        WHERE LENGTH(name) >= {min_seq_len}
        """
    ).fetch_df()
    return result
