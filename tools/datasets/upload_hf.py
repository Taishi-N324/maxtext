import argparse
import json
import logging
import os
from glob import glob
from typing import List, Union

import pandas as pd
from huggingface_hub import create_repo

from datasets import Dataset, concatenate_datasets


class HuggingFaceDatasetUploader:
    def __init__(self, token: str):
        self.token = token
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_multiple_files(
        self, data_paths: Union[str, List[str]], file_format: str = None
    ) -> Dataset:
        if isinstance(data_paths, str):
            file_list = glob(data_paths)
            if not file_list:
                raise ValueError(
                    f"パターン '{data_paths}' に一致するファイルが見つかりませんでした"
                )
            self.logger.info(f"{len(file_list)}個のファイルが見つかりました")
        else:
            file_list = data_paths

        datasets = []
        total_rows = 0

        for file_path in file_list:
            try:
                # ファイル形式が指定されていない場合は、拡張子から判断
                if file_format is None:
                    extension = os.path.splitext(file_path)[1].lower()
                    curr_format = extension[1:] if extension else "csv"
                else:
                    curr_format = file_format

                # JSONLファイルの読み込み
                if curr_format in ["jsonl", ".jsonl"]:
                    data = []
                    with open(file_path, "r", encoding="utf-8") as f:
                        for i, line in enumerate(f):
                            try:
                                data.append(json.loads(line.strip()))
                            except json.JSONDecodeError as e:
                                self.logger.error(f"行 {i+1} のJSONパースに失敗: {e}")
                                raise
                    df = pd.DataFrame(data)
                # その他の形式の読み込み
                elif curr_format in ["csv", ".csv"]:
                    df = pd.read_csv(file_path)
                elif curr_format in ["json", ".json"]:
                    df = pd.read_json(file_path)
                elif curr_format in ["parquet", ".parquet"]:
                    df = pd.read_parquet(file_path)
                else:
                    raise ValueError(f"未対応のファイル形式です: {curr_format}")

                dataset = Dataset.from_pandas(df)
                datasets.append(dataset)
                total_rows += len(dataset)
                self.logger.info(
                    f"ファイル '{file_path}' から {len(dataset)} 行を読み込みました"
                )
            except Exception as e:
                self.logger.error(f"ファイル '{file_path}' の読み込みに失敗: {str(e)}")
                raise

        if not datasets:
            raise ValueError("読み込み可能なデータがありませんでした")

        combined_dataset = concatenate_datasets(datasets)
        self.logger.info(f"合計 {total_rows} 行のデータを読み込みました")
        return combined_dataset

    def upload(self, repo_name: str, dataset: Dataset) -> None:
        try:
            # プライベートリポジトリとして作成
            create_repo(
                repo_id=repo_name, token=self.token, repo_type="dataset", private=True
            )
            self.logger.info(f"プライベートリポジトリ '{repo_name}' を作成しました")

            # データセットをアップロード（プライベート設定）
            dataset.push_to_hub(repo_name, token=self.token, private=True)
            self.logger.info(f"データセットを '{repo_name}' にアップロードしました")

        except Exception as e:
            self.logger.error(f"アップロードに失敗しました: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Hugging Faceにデータセットをアップロードします"
    )
    parser.add_argument(
        "--files", required=True, help="アップロードするファイルのパスまたはパターン"
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Hugging Faceのリポジトリ名 (username/dataset_name)",
    )
    parser.add_argument(
        "--format", help="ファイル形式 (csv, json, jsonl, parquet)", default=None
    )
    args = parser.parse_args()
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError("HUGGINGFACE_TOKENが設定されていません")
    uploader = HuggingFaceDatasetUploader(token)
    dataset = uploader.load_multiple_files(args.files, args.format)
    uploader.upload(args.repo, dataset)


if __name__ == "__main__":
    main()
