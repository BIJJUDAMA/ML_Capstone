# Machine Learning Capstone

## Source dataset and preprocessing

Raw input: [`laptop_price_uncleaned.csv`](laptop_price_uncleaned.csv). Original source: [Kaggle Uncleaned Laptop Price Dataset](https://www.kaggle.com/datasets/ehtishamsadiq/uncleaned-laptop-price-dataset).

[`pre_processing.ipynb`](pre_processing.ipynb) runs once on the raw dataset, then writes one CSV for each track. It does not create a train/test split and does not fit `StandardScaler`. Each track notebook performs its own split and scaling after loading its generated CSV.

### Common raw-data processing in `pre_processing.ipynb`

| Step                | Processing performed                                                                                                                                  |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Column names        | Strips spaces, converts to lowercase, replaces spaces with`_`, removes special characters, drops `unnamed_0` when present                         |
| Numeric parsing     | Converts`price`, `ram`, `weight`, and `inches` from text to numeric values                                                                    |
| Missing values      | Fills missing`gpu` and `opsys` with `Unknown`; fills missing `weight` with its median; removes rows missing `price`, `ram`, or `inches` |
| Duplicates          | Removes duplicate rows                                                                                                                                |
| Category cleanup    | Standardizes`company`, `typename`, and `opsys` text values                                                                                      |
| Outliers            | Caps`price` and `weight` by the IQR bounds, `Q1 - 1.5 × IQR` and `Q3 + 1.5 × IQR`                                                           |
| Display features    | Extracts`res_width` and `res_height`; creates `is_ips` and `is_touchscreen`                                                                   |
| Storage features    | Extracts`storage_gb`; creates `has_ssd` and `has_hdd`                                                                                           |
| PPI feature         | Calculates`ppi = sqrt(res_width² + res_height²) / inches`                                                                                         |
| Encoding            | One-hot encodes`company`, `typename`, and `opsys`; converts boolean features to 0/1                                                             |
| Excluded raw fields | Excludes free-text`cpu`, `gpu`, `screenresolution`, and `memory` from the final model feature list                                            |

### Separate dataset creation per track

| Track          | Code path in`pre_processing.ipynb`                                                                          | Generated file              | Feature columns                          | Target handling                                                       |
| -------------- | ------------------------------------------------------------------------------------------------------------- | --------------------------- | ---------------------------------------- | --------------------------------------------------------------------- |
| Regression     | `regression_df = df_encoded[encoded_feature_cols + ['price']]`                                              | `data/regression.csv`     | `encoded_feature_cols`, all 42 columns | Keeps continuous`price` as `y`                                    |
| Classification | `price_category = pd.qcut(df_encoded['price'], q=4, labels=['Budget', 'Mid-range', 'Premium', 'Flagship'])` | `data/classification.csv` | `encoded_feature_cols`, all 42 columns | Adds`price_category` as `y`; removes `price` to prevent leakage |
| Clustering     | `clustering_df = df_encoded[encoded_feature_cols]`                                                          | `data/clustering.csv`     | `encoded_feature_cols`, all 42 columns | Removes both`price` and `price_category`; no `y` exists         |

Run `pre_processing.ipynb` first. Then run the three track notebooks independently from their own generated CSV files.

## Dataset reference

| Track          | CSV                         | `X`                                 | `y`              |
| -------------- | --------------------------- | ------------------------------------- | ------------------ |
| Regression     | `data/regression.csv`     | Every column except`price`          | `price`          |
| Classification | `data/classification.csv` | Every column except`price_category` | `price_category` |
| Clustering     | `data/clustering.csv`     | Every column                          | None               |

Rows in each CSV: `1,243`

The supplied CSVs contain `42` feature columns. The assignment text refers to 29 features, but one-hot encoding expands the supplied data to 42 columns. Use all 42 columns together as the feature matrix. Do not train one model per column.
