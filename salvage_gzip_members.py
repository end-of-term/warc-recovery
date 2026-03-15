#!/usr/bin/env python3
"""
Recover valid gzip members from a file that is a concatenation of many gzip chunks,
where some chunks are corrupted.

- Reads the input as a byte stream.
- Finds gzip member headers (1f 8b 08).
- Attempts to decompress each member.
- If decompression succeeds, writes the *original compressed bytes* of that member
  to the output (so output is a concatenation of valid gzip members).
- If decompression fails, skips forward to the next plausible gzip header.

This is a best-effort recovery tool. It does not guarantee perfect chunk boundaries
in extremely adversarial corruption scenarios.

Usage:
  python salvage_gzip_members.py input.gz output.gz
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import zlib
from typing import Optional, Tuple

GZ_MAGIC = b"\x1f\x8b"
GZ_METHOD_DEFLATE = 8

# Small utility: find next gzip header (1f 8b 08) from a given offset.
def find_next_gzip_header(mm: memoryview, start: int) -> int:
    """
    Return the offset of the next occurrence of gzip signature+method: 1f 8b 08
    starting at 'start'. Returns -1 if not found.
    """
    needle = b"\x1f\x8b\x08"
    data = mm[start:].tobytes()
    idx = data.find(needle)
    return -1 if idx < 0 else start + idx


def parse_gzip_header(mm: memoryview, off: int) -> Optional[int]:
    """
    Parse the gzip header at offset 'off'. If valid enough, return the offset
    where compressed deflate stream begins. Otherwise return None.
    """
    # Need at least 10 bytes for base header.
    if off < 0 or off + 10 > len(mm):
        return None
    if mm[off:off+2].tobytes() != GZ_MAGIC:
        return None
    method = mm[off + 2]
    if method != GZ_METHOD_DEFLATE:
        return None

    flg = mm[off + 3]
    # Bytes 4..9: mtime(4), xfl, os
    pos = off + 10

    # Optional fields based on FLG.
    # https://www.rfc-editor.org/rfc/rfc1952
    FTEXT   = 0x01
    FHCRC   = 0x02
    FEXTRA  = 0x04
    FNAME   = 0x08
    FCOMMENT= 0x10

    if flg & FEXTRA:
        if pos + 2 > len(mm):
            return None
        xlen = int.from_bytes(mm[pos:pos+2].tobytes(), "little")
        pos += 2
        if pos + xlen > len(mm):
            return None
        pos += xlen

    if flg & FNAME:
        # NUL-terminated string
        while pos < len(mm) and mm[pos] != 0:
            pos += 1
        if pos >= len(mm):
            return None
        pos += 1

    if flg & FCOMMENT:
        while pos < len(mm) and mm[pos] != 0:
            pos += 1
        if pos >= len(mm):
            return None
        pos += 1

    if flg & FHCRC:
        if pos + 2 > len(mm):
            return None
        pos += 2

    # pos now points at start of deflate data
    return pos


def try_decompress_member(mm: memoryview, hdr_off: int, deflate_off: int) -> Optional[int]:
    """
    Try to decompress a gzip member starting at hdr_off with deflate stream at deflate_off.

    If successful, returns the offset just after the end of this gzip member
    (i.e., end of compressed data + 8-byte gzip trailer).
    If unsuccessful, returns None.

    Strategy:
    - Use raw DEFLATE decompressor (wbits=-15).
    - Feed it bytes from deflate_off onward.
    - When it reaches EOF, zlib exposes unused_data = bytes after the DEFLATE stream.
    - Then we require at least 8 bytes after the DEFLATE stream for gzip trailer.
    - We do not validate ISIZE/CRC ourselves here; zlib doesn't provide gzip CRC
      check for raw streams. However, corruption usually prevents reaching EOF cleanly.
      (You can optionally add stricter checks if you want.)
    """
    decomp = zlib.decompressobj(wbits=-zlib.MAX_WBITS)

    i = deflate_off
    n = len(mm)

    # Read in moderately sized blocks to avoid huge memory spikes.
    # (Doesn't matter much since we're not keeping decompressed output.)
    CHUNK = 1 << 20  # 1 MiB

    try:
        while True:
            if i >= n:
                return None  # ran out of input before reaching end-of-stream
            block = mm[i:min(i + CHUNK, n)].tobytes()
            _ = decomp.decompress(block)  # discard output
            i += len(block)

            if decomp.eof:
                # decomp.unused_data begins at the first byte after the deflate stream,
                # but it's only what was in the last 'block' past the end.
                unused = decomp.unused_data
                # Compute where the deflate stream ended in absolute file offsets.
                # We consumed len(block) bytes but some tail 'unused' wasn't part of deflate.
                deflate_end = i - len(unused)
                # Need gzip trailer (CRC32 + ISIZE), 8 bytes
                if deflate_end + 8 > n:
                    return None
                member_end = deflate_end + 8
                return member_end

            # If decompressor needs more input, loop.

    except zlib.error:
        return None


def salvage(input_path: str, output_path: str, *, verbose: bool = True) -> None:
    with open(input_path, "rb") as f:
        data = f.read()

    mm = memoryview(data)
    n = len(mm)

    out = open(output_path, "wb")

    recovered = 0
    discarded_bytes = 0

    off = 0
    # Find first plausible header
    off = find_next_gzip_header(mm, off)

    while off != -1 and off < n:
        deflate_off = parse_gzip_header(mm, off)
        if deflate_off is None:
            # Not a usable header; move forward by 1 byte and keep searching.
            off += 1
            off = find_next_gzip_header(mm, off)
            continue

        member_end = try_decompress_member(mm, off, deflate_off)
        if member_end is not None:
            # Write original compressed bytes for the member.
            out.write(mm[off:member_end])
            recovered += 1
            if verbose:
                sys.stderr.write(f"[ok] member #{recovered}  bytes={member_end - off}  offset={off}\n")
            off = member_end
            off = find_next_gzip_header(mm, off)
        else:
            # Corrupt member; skip forward to next header after current one.
            # We count discarded bytes as "from this header to next found header".
            next_off = find_next_gzip_header(mm, off + 1)
            if next_off == -1:
                # nothing more to salvage
                discarded_bytes += (n - off)
                if verbose:
                    sys.stderr.write(f"[bad] corruption from offset={off} to EOF (discard {n - off} bytes)\n")
                break
            discarded_bytes += (next_off - off)
            if verbose:
                sys.stderr.write(f"[bad] corruption at offset={off} (discard {next_off - off} bytes), resync -> {next_off}\n")
            off = next_off

    out.close()

    if verbose:
        sys.stderr.write(
            f"\nDone. Recovered members: {recovered}. Discarded bytes (approx): {discarded_bytes}.\n"
            f"Output written to: {output_path}\n"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Recover valid gzip members from a concatenated/corrupted gzip stream.")
    ap.add_argument("input", help="Input file (concatenated gzip, possibly corrupted)")
    ap.add_argument("output", help="Output file (concatenation of recovered gzip members)")
    ap.add_argument("-q", "--quiet", action="store_true", help="Suppress progress logging")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input not found: {args.input}")

    salvage(args.input, args.output, verbose=not args.quiet)


if __name__ == "__main__":
    main()
