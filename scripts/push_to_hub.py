"""CLI to push trained SO-101 bowling actors or collected datasets to Hugging Face Hub."""

from __future__ import annotations

import argparse
import json

from so101_bowling.hub import push_dataset_to_hub, push_model_to_hub


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="kind", required=True)

    m_parser = sub.add_parser("model", help="Push a trained actor run directory to HF Hub.")
    m_parser.add_argument("--run-dir", required=True, help="Local training run directory.")
    m_parser.add_argument("--repo-id", required=True, help="HF repo id, e.g. user/so101-bowling-sac.")
    m_parser.add_argument("--private", action="store_true")
    m_parser.add_argument("--message", default=None)

    d_parser = sub.add_parser("dataset", help="Push a collected bowling dataset to HF Hub.")
    d_parser.add_argument("--dataset-dir", required=True, help="Local dataset directory.")
    d_parser.add_argument("--repo-id", required=True, help="HF dataset repo id, e.g. user/so101-bowling-demos.")
    d_parser.add_argument("--private", action="store_true")
    d_parser.add_argument("--message", default=None)

    args = parser.parse_args()
    if args.kind == "model":
        url = push_model_to_hub(
            run_dir=args.run_dir,
            repo_id=args.repo_id,
            private=args.private,
            commit_message=args.message,
        )
    else:
        url = push_dataset_to_hub(
            dataset_dir=args.dataset_dir,
            repo_id=args.repo_id,
            private=args.private,
            commit_message=args.message,
        )
    print(json.dumps({"kind": args.kind, "repo_id": args.repo_id, "url": url}, indent=2))


if __name__ == "__main__":
    main()
