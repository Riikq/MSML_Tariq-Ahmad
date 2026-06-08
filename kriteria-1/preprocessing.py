import os
import pandas as pd
from preprocessing/automate_TariqAhmad import preprocess_data


if __name__ == "__main__":
    raw_path = "./pilgrimage_raw.csv"
    save_pipeline_path = "preprocessing/preprocessor.joblib"
    save_header_path = "preprocessing/pilgrimage_preprocessing/columns.csv"
    save_dataset_path = "preprocessing/pilgrimage_preprocessing"

    os.makedirs(save_dataset_path, exist_ok=True)

    df = pd.read_csv(raw_path)

    X_train, X_test, y_train, y_test = preprocess_data(
        data=df,
        target_column="charges",
        save_path=save_pipeline_path,
        file_path=save_header_path,
    )

    pd.DataFrame(X_train).assign(target=y_train.reset_index(drop=True)).to_csv(
        f"{save_dataset_path}/pilgrimage_train_preprocessed.csv", index=False
    )

    pd.DataFrame(X_test).assign(target=y_test.reset_index(drop=True)).to_csv(
        f"{save_dataset_path}/pilgrimage_test_preprocessed.csv", index=False
    )
