Please store interaction files in this path. Each file should look like:

| item_id | user_id | timestamp |
|---------|---------|-----------|
| item_i  | user_j  | time_k    |

item_id should start from **1**, as 0 would be treated as a dummy item.

For information, please follow this format:

| user_id | field_1 | field_2 | ... |
|---------|---------|-----------| --- |
| user_j  | some string  | other string    | ... |

You would refer those fields in `text_keys` of launch command.

# Examples
- EB-NeRD.ipynb: process [EB-NeRD data](https://recsys.eb.dk/), please download the parquet files first;
- build_hllm_data_from_merrec.py: process [mrerec](https://huggingface.co/datasets/mercari-us/merrec), please download the dataset from huggingface. It is very huge!
