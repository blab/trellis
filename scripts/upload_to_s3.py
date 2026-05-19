"""Upload trellis results to S3."""

import argparse
import os
import sys

import boto3


EXCLUDED_FILES = {".DS_Store", ".snakemake_timestamp"}


def _format_size(n_bytes: int) -> str:
    if n_bytes >= 1e9:
        return f"{n_bytes / 1e9:.1f} GB"
    if n_bytes >= 1e6:
        return f"{n_bytes / 1e6:.0f} MB"
    if n_bytes >= 1e3:
        return f"{n_bytes / 1e3:.1f} KB"
    return f"{n_bytes} B"


def _list_s3_objects(s3, bucket: str, prefix: str) -> list[dict]:
    """List all objects under an S3 prefix."""
    objects = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" in page:
            objects.extend(page["Contents"])
    return objects


def _delete_s3_objects(s3, bucket: str, objects: list[dict]) -> int:
    """Delete a list of S3 objects. Returns count deleted."""
    deleted = 0
    for i in range(0, len(objects), 1000):
        batch = [{"Key": obj["Key"]} for obj in objects[i : i + 1000]]
        s3.delete_objects(Bucket=bucket, Delete={"Objects": batch})
        deleted += len(batch)
    return deleted


def _local_files(dataset_path: str) -> list[str]:
    """List uploadable files in a dataset directory."""
    files = []
    for root, _dirs, filenames in os.walk(dataset_path):
        for f in filenames:
            if f not in EXCLUDED_FILES:
                files.append(os.path.join(root, f))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets", nargs="*",
        help="Dataset directory names to upload (default: all subdirs of --upload-dir)",
    )
    parser.add_argument("--upload-dir", default="results")
    parser.add_argument("--prefix", default="trellis-trajectories")
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Skip confirmation prompts for overwrites",
    )
    args = parser.parse_args()

    required_vars = ["S3_BUCKET", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        sys.exit(f"error: missing environment variables: {', '.join(missing)}")

    bucket = os.environ["S3_BUCKET"]
    s3 = boto3.client("s3")

    if args.datasets:
        datasets = []
        for name in args.datasets:
            path = os.path.join(args.upload_dir, name)
            if not os.path.isdir(path):
                print(f"warning: {path} not found, skipping")
            else:
                datasets.append(name)
    else:
        datasets = sorted(
            d for d in os.listdir(args.upload_dir)
            if os.path.isdir(os.path.join(args.upload_dir, d))
        )

    if not datasets:
        sys.exit(f"no datasets found in {args.upload_dir}/")

    uploaded = 0
    for dataset in datasets:
        dataset_path = os.path.join(args.upload_dir, dataset)
        files = _local_files(dataset_path)
        total_size = sum(os.path.getsize(f) for f in files)
        print(f"\n{dataset}: {len(files)} files ({_format_size(total_size)})")

        s3_prefix = f"{args.prefix}/{dataset}/"
        existing = _list_s3_objects(s3, bucket, s3_prefix)
        if existing:
            existing_size = sum(obj["Size"] for obj in existing)
            print(
                f"  s3://{bucket}/{s3_prefix} exists "
                f"({len(existing)} files, {_format_size(existing_size)})"
            )
            if not args.force:
                answer = input("  overwrite? [y/N] ").strip().lower()
                if answer != "y":
                    print("  skipped")
                    continue
            deleted = _delete_s3_objects(s3, bucket, existing)
            print(f"  deleted {deleted} existing files")

        for filepath in files:
            relative = os.path.relpath(filepath, dataset_path)
            s3_key = f"{args.prefix}/{dataset}/{relative}"
            size = os.path.getsize(filepath)
            print(f"  uploading {relative} ({_format_size(size)})...")
            s3.upload_file(filepath, bucket, s3_key)

        print("  done")
        uploaded += 1

    print(f"\nuploaded {uploaded} dataset(s) to s3://{bucket}/{args.prefix}/")


if __name__ == "__main__":
    main()
