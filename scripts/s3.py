"""Push and pull trellis results to/from S3."""

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


def _discover_datasets(s3, bucket: str, prefix: str) -> list[str]:
    """List dataset names available under the S3 prefix."""
    objects = _list_s3_objects(s3, bucket, prefix + "/")
    datasets = set()
    for obj in objects:
        relative = obj["Key"][len(prefix) + 1:]
        parts = relative.split("/", 1)
        if parts[0]:
            datasets.add(parts[0])
    return sorted(datasets)


def _local_files(dataset_path: str) -> list[str]:
    """List uploadable files in a dataset directory."""
    files = []
    for root, _dirs, filenames in os.walk(dataset_path):
        for f in filenames:
            if f not in EXCLUDED_FILES:
                files.append(os.path.join(root, f))
    return sorted(files)


def _local_file_stats(dataset_path: str) -> tuple[int, int]:
    """Return (file_count, total_bytes) for a local directory."""
    count = 0
    total = 0
    for root, _dirs, filenames in os.walk(dataset_path):
        for f in filenames:
            count += 1
            total += os.path.getsize(os.path.join(root, f))
    return count, total


def _delete_s3_objects(s3, bucket: str, objects: list[dict]) -> int:
    """Delete a list of S3 objects. Returns count deleted."""
    deleted = 0
    for i in range(0, len(objects), 1000):
        batch = [{"Key": obj["Key"]} for obj in objects[i : i + 1000]]
        s3.delete_objects(Bucket=bucket, Delete={"Objects": batch})
        deleted += len(batch)
    return deleted


def cmd_push(args, s3, bucket: str) -> None:
    local_dir = args.local_dir

    if args.datasets:
        datasets = []
        for name in args.datasets:
            path = os.path.join(local_dir, name)
            if not os.path.isdir(path):
                print(f"warning: {path} not found, skipping")
            else:
                datasets.append(name)
    else:
        datasets = sorted(
            d for d in os.listdir(local_dir)
            if os.path.isdir(os.path.join(local_dir, d))
        )

    if not datasets:
        sys.exit(f"no datasets found in {local_dir}/")

    uploaded = 0
    for dataset in datasets:
        dataset_path = os.path.join(local_dir, dataset)
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


def cmd_list(args, s3, bucket: str) -> None:
    datasets = _discover_datasets(s3, bucket, args.prefix)
    if not datasets:
        print(f"no datasets found under s3://{bucket}/{args.prefix}/")
        return
    print(f"available datasets in s3://{bucket}/{args.prefix}/:")
    for name in datasets:
        objects = _list_s3_objects(s3, bucket, f"{args.prefix}/{name}/")
        total_size = sum(obj["Size"] for obj in objects)
        print(f"  {name} ({len(objects)} files, {_format_size(total_size)})")


def cmd_pull(args, s3, bucket: str) -> None:
    local_dir = args.local_dir

    if args.all:
        args.datasets = _discover_datasets(s3, bucket, args.prefix)
        if not args.datasets:
            sys.exit(f"no datasets found under s3://{bucket}/{args.prefix}/")
    elif not args.datasets:
        sys.exit("error: specify dataset names or use --all")

    downloaded = 0
    for dataset in args.datasets:
        s3_prefix = f"{args.prefix}/{dataset}/"
        objects = _list_s3_objects(s3, bucket, s3_prefix)
        if not objects:
            print(f"warning: no objects at s3://{bucket}/{s3_prefix}, skipping")
            continue

        remote_size = sum(obj["Size"] for obj in objects)
        print(f"\n{dataset}: {len(objects)} files ({_format_size(remote_size)})")

        local_path = os.path.join(local_dir, dataset)
        if os.path.isdir(local_path):
            local_count, local_size = _local_file_stats(local_path)
            print(
                f"  {local_path}/ exists "
                f"({local_count} files, {_format_size(local_size)})"
            )
            if not args.force:
                answer = input("  overwrite? [y/N] ").strip().lower()
                if answer != "y":
                    print("  skipped")
                    continue

        for obj in objects:
            relative = obj["Key"][len(s3_prefix):]
            dest = os.path.join(local_path, relative)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            size = obj["Size"]
            print(f"  downloading {relative} ({_format_size(size)})...")
            s3.download_file(bucket, obj["Key"], dest)

        print("  done")
        downloaded += 1

    print(f"\ndownloaded {downloaded} dataset(s) from s3://{bucket}/{args.prefix}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="trellis-trajectories")
    parser.add_argument("--local-dir", default="results")
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Skip confirmation prompts for overwrites",
    )

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List available datasets on S3")

    push = sub.add_parser("push", help="Upload datasets to S3")
    push.add_argument(
        "datasets", nargs="*",
        help="Dataset names to upload (default: all in local-dir)",
    )

    pull = sub.add_parser("pull", help="Download datasets from S3")
    pull.add_argument(
        "datasets", nargs="*",
        help="Dataset names to download",
    )
    pull.add_argument(
        "--all", "-a", action="store_true",
        help="Download all available datasets",
    )

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    required_vars = ["S3_BUCKET", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        sys.exit(f"error: missing environment variables: {', '.join(missing)}")

    bucket = os.environ["S3_BUCKET"]
    s3 = boto3.client("s3")

    if args.command == "list":
        cmd_list(args, s3, bucket)
    elif args.command == "push":
        cmd_push(args, s3, bucket)
    elif args.command == "pull":
        cmd_pull(args, s3, bucket)


if __name__ == "__main__":
    main()
