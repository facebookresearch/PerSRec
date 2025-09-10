# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

# install miniconda3
# conda create --name pytorch310 python=3.10
# conda activate pytorch310
# conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
# pip3 install -r requirements.txt
# python3 -m build_hllm_data_from_merrec
# python3 -m build_hllm_data_from_merrec -t name -t category_name -t brand_name -t price -t item_condition_name -t size_name -t color -t shipper_name --min_seq_len 2048 --dataset_name merrec_item_id --object_id_col item_id
# python3 -m build_hllm_data_from_merrec -t name -t category_name -t brand_name -t price --min_seq_len 3000 --dataset_name merrec_item_id_4t --object_id_col item_id --output_type parquet

import os
from typing import Tuple

import click

from .merrec_data import build_llm_dataset


@click.command()
@click.option("--hf_dataset_name", type=str, default="mercari-us/merrec")
@click.option(
    "--output_dir",
    type=str,
    default="~/local/data/merrec/",
)
@click.option("--min_seq_len", type=int, default=1024)
@click.option("--output_type", type=str, default="parquet")
@click.option(
    "--dataset_name",
    type=str,
    default="merrec",
)
@click.option("--object_id_col", type=str, default="product_id")
@click.option(
    "-t",
    "--text_cols",
    type=str,
    multiple=True,
    default=["brand_name", "category_name"],
)
def main(
    hf_dataset_name: str,
    output_dir: str,
    output_type: str,
    dataset_name: str,
    min_seq_len: int,
    object_id_col: str,
    text_cols: Tuple[str, ...],
) -> None:
    output_dir = os.path.expanduser(output_dir)

    interaction_output_dir = os.path.join(output_dir, "interaction")
    os.makedirs(interaction_output_dir, exist_ok=True)
    interaction_output = os.path.join(
        interaction_output_dir, f"{dataset_name}_{min_seq_len}.{output_type}"
    )
    print(f"interaction_output: {interaction_output}")

    item_output_dir = os.path.join(output_dir, "information")
    os.makedirs(item_output_dir, exist_ok=True)
    item_output = os.path.join(
        item_output_dir, f"{dataset_name}_{min_seq_len}.{output_type}"
    )
    print(f"item_output: {item_output}")

    event_id_mapping = {
        "item_view": 0,
        "item_like": 1,
        "item_add_to_cart_tap": 2,
        "offer_make": 3,
        "buy_start": 4,
        "buy_comp": 5,
    }
    build_llm_dataset(
        interaction_output=interaction_output,
        item_output=item_output,
        hf_dataset_name=hf_dataset_name,
        min_seq_len=min_seq_len,
        object_id_col=object_id_col,
        text_cols=text_cols,
        event_id_mapping=event_id_mapping,
    )


if __name__ == "__main__":
    main()
