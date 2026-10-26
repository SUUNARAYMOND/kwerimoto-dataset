"""Upload the built dataset to Hugging Face.

    uv run kwerimoto-upload --repo rayyyyy007/kwerimoto-engine-sounds

Uploads manifest.csv and prepared/ from --data, plus the dataset card
(dataset_card.md, shown as the dataset's README). Raw source files are not
uploaded. Needs a write token: run `uv run hf auth login` once first.
"""
import argparse

from huggingface_hub import HfApi


def upload(repo_id, data_dir="data", card="dataset_card.md", private=False):
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=repo_id,
                    repo_type="dataset", commit_message="Update dataset card")
    api.upload_folder(folder_path=str(data_dir), repo_id=repo_id, repo_type="dataset",
                      allow_patterns=["manifest.csv", "prepared/**"],
                      delete_patterns=["prepared/**"],  # clips gone from a rebuild are removed too
                      commit_message="Upload prepared clips and manifest")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="Hugging Face dataset id, e.g. user/name")
    parser.add_argument("--data", default="data", help="built dataset folder")
    parser.add_argument("--card", default="dataset_card.md", help="dataset card to show as the README")
    parser.add_argument("--private", action="store_true", help="create the repo as private")
    args = parser.parse_args()
    upload(args.repo, args.data, args.card, args.private)


if __name__ == "__main__":
    main()
