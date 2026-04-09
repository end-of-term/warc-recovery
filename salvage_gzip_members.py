#!/usr/bin/env python3
"""
Recover valid gzip members from a corrupted concatenated gzip file.

This version is stricter than the earlier one:
- It validates full gzip members, including CRC and ISIZE.
- Members that would fail `gzip -t` are discarded.
- Valid members are copied out byte-for-byte into the output file.

Usage:
    python salvage_gzip_members.py input.gz output.gz
"""

from __future__ import annotations

import argparse
import os
import sys
import zlib

GZIP_HEADER = b"\x1f\x8b\x08"


def find_next_header(buf: memoryview, start: int) -> int:
    idx = buf[start:].tobytes().find(GZIP_HEADER)
    return -1 if idx < 0 else start + idx


def try_recover_member(buf: memoryview, start: int, chunk_size: int = 1024 * 1024):
    """
    Attempt to validate and recover one gzip member starting at `start`.

    Returns:
        (member_end, uncompressed_size) on success
        None on failure

    member_end is the absolute offset just after the valid gzip member.
    """
    decomp = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)  # gzip mode with CRC validation
    pos = start
    n = len(buf)
    total_out = 0

    try:
        while pos < n:
            block_end = min(pos + chunk_size, n)
            chunk = buf[pos:block_end].tobytes()
            out = decomp.decompress(chunk)
            total_out += len(out)
            pos = block_end

            if decomp.eof:
                # Bytes after the end of this gzip member that were present in the last chunk
                unused = decomp.unused_data
                member_end = pos - len(unused)
                return member_end, total_out

        return None

    except zlib.error:
        return None


def salvage(input_path: str, output_path: str, verbose: bool = True):
    with open(input_path, "rb") as f:
        data = f.read()

    buf = memoryview(data)
    n = len(buf)

    recovered = 0
    discarded = 0
    pos = 0

    with open(output_path, "wb") as out:
        pos = find_next_header(buf, 0)

        while pos != -1 and pos < n:
            result = try_recover_member(buf, pos)

            if result is not None:
                member_end, usize = result
                out.write(buf[pos:member_end])
                recovered += 1
                if verbose:
                    print(
                        f"[ok]  member #{recovered}  offset={pos}  compressed={member_end - pos}  uncompressed={usize}",
                        file=sys.stderr,
                    )
                pos = find_next_header(buf, member_end)
            else:
                next_pos = find_next_header(buf, pos + 1)
                if next_pos == -1:
                    discarded += n - pos
                    if verbose:
                        print(f"[bad] offset={pos} -> EOF  discarded={n - pos}", file=sys.stderr)
                    break
                discarded += next_pos - pos
                if verbose:
                    print(
                        f"[bad] offset={pos}  discarded={next_pos - pos}  resync={next_pos}",
                        file=sys.stderr,
                    )
                pos = next_pos

    if verbose:
        print(
            f"\nDone. Recovered {recovered} valid gzip member(s). "
            f"Discarded approximately {discarded} byte(s).",
            file=sys.stderr,
        )


def main():
    ap = argparse.ArgumentParser(description="Recover only fully valid gzip members from a corrupted concatenated gzip file.")
    ap.add_argument("input", help="Input concatenated gzip file")
    ap.add_argument("output", help="Output recovered gzip file")
    ap.add_argument("-q", "--quiet", action="store_true", help="Suppress progress messages")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input file not found: {args.input}")

    salvage(args.input, args.output, verbose=not args.quiet)


if __name__ == "__main__":
    main()
