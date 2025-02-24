import sys
import os
from huggingface_hub import HfApi, create_repo


def upload_checkpoint(checkpoint_path, repo_id, checkpoint_num):
    """指定されたチェックポイントをHugging Faceにアップロードする"""
    print(
        f"チェックポイント {checkpoint_num} をアップロード中: {checkpoint_path} -> {repo_id}"
    )

    # HfApiインスタンスの作成
    api = HfApi()

    try:
        # リポジトリが存在するか確認し、存在しなければ作成
        try:
            api.repo_info(repo_id=repo_id)
            print(f"リポジトリ {repo_id} は既に存在します")
        except Exception:
            print(f"リポジトリ {repo_id} を作成します")
            create_repo(repo_id=repo_id, private=True, repo_type="model")

        # フォルダをアップロード
        api.upload_folder(
            folder_path=checkpoint_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload checkpoint {checkpoint_num}",
        )
        print(f"チェックポイント {checkpoint_num} のアップロードが完了しました")
        return True
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        return False


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "使用法: python upload_to_hf.py <チェックポイントパス> <リポジトリID> <チェックポイント番号>"
        )
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    repo_id = sys.argv[2]
    checkpoint_num = sys.argv[3]

    success = upload_checkpoint(checkpoint_path, repo_id, checkpoint_num)
    sys.exit(0 if success else 1)
