"""Publish the layoff-risk model and portfolio demo Space to Hugging Face.

Requires: `pip install huggingface_hub`, then `huggingface-cli login`
(paste a token with write access from https://huggingface.co/settings/tokens).

Usage:
  python scripts/push_to_huggingface.py --username YOUR_HF_USERNAME
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def push_model(api: HfApi, repo_id: str) -> None:
    print(f"Creating/updating model repo: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(PROJECT_ROOT / "model_card" / "README.md"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=str(PROJECT_ROOT / "data" / "models" / "xgboost.pkl"),
        path_in_repo="xgboost.pkl",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Model published: https://huggingface.co/{repo_id}")


def push_space(api: HfApi, repo_id: str) -> None:
    print(f"Creating/updating Space repo: {repo_id}")
    # Static Spaces (plain HTML/JS, no Python server) are free on any tier —
    # Gradio/Docker/Streamlit-class Spaces require an HF PRO subscription for
    # free-tier cpu-basic hardware. The actual runtime is determined by the
    # `sdk:` field in huggingface_space/README.md's frontmatter once uploaded.
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(
        folder_path=str(PROJECT_ROOT / "huggingface_space"),
        repo_id=repo_id,
        repo_type="space",
    )
    print(f"Space published: https://huggingface.co/spaces/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True, help="Your Hugging Face username")
    parser.add_argument("--model-repo-name", default="layoff-risk-xgboost")
    parser.add_argument("--space-repo-name", default="layoff-risk-monitor-demo")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--skip-space", action="store_true")
    args = parser.parse_args()

    api = HfApi()

    if not args.skip_model:
        push_model(api, f"{args.username}/{args.model_repo_name}")
    if not args.skip_space:
        push_space(api, f"{args.username}/{args.space_repo_name}")


if __name__ == "__main__":
    main()
