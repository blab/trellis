"""Inspect a tar.zst shard: print FASTA contents, list files, or extract."""

import argparse
import sys
import tarfile
from pathlib import Path

import zstandard as zstd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("shard", type=str, help="path to a .tar.zst shard")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--list", "-l", action="store_true",
                      help="list filenames only")
    mode.add_argument("--extract", "-x", type=str, metavar="DIR",
                      help="extract FASTA files to DIR")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dctx = zstd.ZstdDecompressor()
    with open(args.shard, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            with tarfile.open(mode="r|", fileobj=reader) as tar:
                if args.extract:
                    out = Path(args.extract)
                    out.mkdir(parents=True, exist_ok=True)
                    for member in tar:
                        f = tar.extractfile(member)
                        if f is None:
                            continue
                        (out / member.name).write_bytes(f.read())
                        print(member.name)
                elif args.list:
                    for member in tar:
                        print(member.name)
                else:
                    for member in tar:
                        f = tar.extractfile(member)
                        if f is None:
                            continue
                        print(f"==> {member.name} <==")
                        sys.stdout.write(f.read().decode())
                        print()


if __name__ == "__main__":
    main()
