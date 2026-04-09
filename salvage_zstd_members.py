#!/usr/bin/env python3

import argparse
import ctypes
import ctypes.util
import io
import mmap
import os
import struct
import sys
import time

try:
    import zstandard as zstd
except ImportError:
    print("ERROR: This script requires the 'zstandard' Python package.", file=sys.stderr)
    print("Install it with: pip install zstandard", file=sys.stderr)
    sys.exit(1)


# Regular Zstd frame magic: 0xFD2FB528, little-endian bytes on disk
ZSTD_FRAME_MAGIC = b"\x28\xb5\x2f\xfd"

# Zstd dictionary magic: 0xEC30A437, little-endian bytes on disk
ZSTD_DICT_MAGIC = b"\x37\xa4\x30\xec"

# Custom dictionary wrapper at the beginning of the file:
# skippable frame magic 0x184D2A5D -> little-endian bytes:
CUSTOM_DICT_MAGIC = b"\x5D\x2A\x4D\x18"


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def fmt_bytes(n):
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    v = float(n)
    for u in units:
        if v < 1024.0 or u == units[-1]:
            return f"{v:.1f} {u}"
        v /= 1024.0
    return f"{n} B"


class NullWriter(io.RawIOBase):
    def writable(self):
        return True

    def write(self, b):
        return len(b)


class BytesReader(io.RawIOBase):
    def __init__(self, data):
        self._data = data
        self._pos = 0

    def readable(self):
        return True

    def readinto(self, b):
        if self._pos >= len(self._data):
            return 0
        n = min(len(b), len(self._data) - self._pos)
        b[:n] = self._data[self._pos:self._pos + n]
        self._pos += n
        return n


class LibZstd:
    def __init__(self):
        libname = ctypes.util.find_library("zstd")
        if not libname:
            raise RuntimeError("Could not find libzstd on this system.")
        self.lib = ctypes.CDLL(libname)

        self.lib.ZSTD_findFrameCompressedSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.lib.ZSTD_findFrameCompressedSize.restype = ctypes.c_size_t

        self.lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
        self.lib.ZSTD_isError.restype = ctypes.c_uint

        self.lib.ZSTD_getErrorName.argtypes = [ctypes.c_size_t]
        self.lib.ZSTD_getErrorName.restype = ctypes.c_char_p

    def _ptr_at(self, mm, offset):
        return ctypes.addressof((ctypes.c_ubyte * 1).from_buffer(mm, offset))

    def find_frame_compressed_size(self, mm, offset, total_size):
        remaining = total_size - offset
        if remaining <= 0:
            return None, "no remaining bytes"

        ptr = self._ptr_at(mm, offset)
        res = self.lib.ZSTD_findFrameCompressedSize(ptr, remaining)
        if self.lib.ZSTD_isError(res):
            err = self.lib.ZSTD_getErrorName(res).decode("utf-8", errors="replace")
            return None, err
        return int(res), None


def read_exact_prefix(mm, offset, length, file_size, label):
    end = offset + length
    if end > file_size:
        raise RuntimeError(f"Truncated {label}: need {length} bytes at offset {offset}, file too short")
    return mm[offset:end]


def extract_embedded_dictionary(mm, file_size, verbose=False):
    """
    Parse the custom embedded dictionary container at the start of the file.

    Layout:
      0..3   : custom skippable-frame magic = b'\\x5D\\x2A\\x4D\\x18'
      4..7   : dict size, uint32 little-endian
      8..... : dict payload of exactly dict_size bytes

    The payload is either:
      - a compressed zstd blob (starts with ZSTD_FRAME_MAGIC), or
      - a zstd dictionary blob (starts with ZSTD_DICT_MAGIC)

    Returns:
      prefix_bytes_to_preserve_verbatim,
      usable_dictionary_bytes,
      first_member_offset
    """
    if file_size < 8:
        raise RuntimeError("Input file too short to contain embedded dictionary header")

    magic = bytes(mm[0:4])
    if magic != CUSTOM_DICT_MAGIC:
        raise RuntimeError(
            f"not a valid custom-dictionary zstd file: expected initial magic "
            f"{CUSTOM_DICT_MAGIC!r}, got {magic!r}"
        )

    dict_size_bytes = read_exact_prefix(mm, 4, 4, file_size, "dict size")
    dict_size = struct.unpack("<I", dict_size_bytes)[0]

    if dict_size < 4:
        raise RuntimeError(f"dict too small: {dict_size}")
    if dict_size >= 100 * 1024**2:
        raise RuntimeError(f"dict too large: {dict_size}")

    dict_payload = read_exact_prefix(mm, 8, dict_size, file_size, "dictionary payload")
    dict_payload = bytes(dict_payload)

    prefix_end = 8 + dict_size
    prefix_bytes = bytes(mm[:prefix_end])

    if dict_payload.startswith(ZSTD_FRAME_MAGIC):
        if verbose:
            eprint("[info] Embedded dictionary payload is itself zstd-compressed; decompressing it")
        try:
            usable_dict = zstd.ZstdDecompressor().decompress(dict_payload)
        except Exception as e:
            raise RuntimeError(f"Failed to decompress embedded dictionary payload: {e}")
    elif dict_payload.startswith(ZSTD_DICT_MAGIC):
        if verbose:
            eprint("[info] Embedded dictionary payload is already a zstd dictionary")
        usable_dict = dict_payload
    else:
        raise RuntimeError(
            "Embedded dictionary payload does not look valid: "
            "it does not start with zstd frame magic or zstd dictionary magic"
        )

    return prefix_bytes, usable_dict, prefix_end


def make_dict_candidates(dict_bytes):
    """
    Build candidate dictionary interpretations.
    Trying multiple dict types helps if the embedded content is valid but the
    constructor would otherwise infer the wrong type.
    """
    candidates = []

    seen = set()

    def add(name, ctor):
        try:
            zd = ctor()
            key = (name, getattr(zd, "dict_id", lambda: None)())
            if key not in seen:
                candidates.append((name, zd))
                seen.add(key)
        except Exception:
            pass

    add("auto", lambda: zstd.ZstdCompressionDict(dict_bytes))
    add("full", lambda: zstd.ZstdCompressionDict(dict_bytes, dict_type=zstd.DICT_TYPE_FULLDICT))
    add("raw", lambda: zstd.ZstdCompressionDict(dict_bytes, dict_type=zstd.DICT_TYPE_RAWCONTENT))

    if not candidates:
        raise RuntimeError("Could not construct any usable ZstdCompressionDict from extracted dictionary bytes")

    return candidates


def get_frame_dict_id(frame_bytes):
    try:
        fp = zstd.get_frame_parameters(frame_bytes)
        return fp.dict_id
    except Exception:
        return None


def verify_frame_with_dict_candidates(frame_bytes, dict_candidates, verbose=False):
    """
    Try to fully stream-decompress one frame using candidate dictionary modes.
    Output is discarded. On success, return the mode that worked.
    """
    frame_dict_id = get_frame_dict_id(frame_bytes)
    last_err = None

    for mode_name, zd in dict_candidates:
        try:
            dict_id = None
            try:
                dict_id = zd.dict_id()
            except Exception:
                pass

            if verbose:
                eprint(
                    f"[dict-try] mode={mode_name}, "
                    f"frame_dict_id={frame_dict_id}, candidate_dict_id={dict_id}"
                )

            dctx = zstd.ZstdDecompressor(dict_data=zd)
            reader = dctx.stream_reader(io.BytesIO(frame_bytes))
            try:
                while reader.read(131072):
                    pass
            finally:
                reader.close()

            return True, mode_name, dict_id, None

        except zstd.ZstdError as e:
            last_err = str(e)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"

    return False, None, None, last_err


def recover_file(input_path, output_path, verbose=False, progress_every_mb=256):
    libz = LibZstd()

    file_size = os.path.getsize(input_path)
    if file_size == 0:
        raise RuntimeError("Input file is empty.")

    start_time = time.time()
    next_progress_at = 0

    with open(input_path, "rb") as inf:
        with mmap.mmap(inf.fileno(), 0, access=mmap.ACCESS_COPY) as mm:
            prefix_bytes, usable_dict_bytes, pos = extract_embedded_dictionary(mm, file_size, verbose=verbose)
            dict_candidates = make_dict_candidates(usable_dict_bytes)

            recovered_members = 0
            rejected_candidates = 0
            skipped_bytes = 0
            scanned_bytes = pos

            if verbose:
                eprint(f"Input: {input_path}")
                eprint(f"Output: {output_path}")
                eprint(f"Input size: {fmt_bytes(file_size)}")
                eprint(f"Preserved prefix size: {fmt_bytes(len(prefix_bytes))}")
                eprint(f"Usable dictionary size: {fmt_bytes(len(usable_dict_bytes))}")
                eprint(f"First member scan offset: {pos}")
                for mode_name, zd in dict_candidates:
                    try:
                        did = zd.dict_id()
                    except Exception:
                        did = None
                    eprint(f"[dict] mode={mode_name}, dict_id={did}")
                eprint("")

            with open(output_path, "wb") as outf:
                outf.write(prefix_bytes)

                while True:
                    if verbose and scanned_bytes >= next_progress_at:
                        elapsed = max(time.time() - start_time, 1e-9)
                        rate = scanned_bytes / elapsed
                        pct = (scanned_bytes / file_size) * 100 if file_size else 100.0
                        eprint(
                            f"[progress] scanned={fmt_bytes(scanned_bytes)} / {fmt_bytes(file_size)} "
                            f"({pct:.2f}%), recovered={recovered_members}, rejected={rejected_candidates}, "
                            f"rate={fmt_bytes(rate)}/s"
                        )
                        next_progress_at = scanned_bytes + progress_every_mb * 1024 * 1024

                    candidate = mm.find(ZSTD_FRAME_MAGIC, pos)
                    if candidate == -1:
                        skipped_bytes += file_size - pos
                        scanned_bytes = file_size
                        break

                    if candidate > pos:
                        gap = candidate - pos
                        skipped_bytes += gap
                        scanned_bytes = candidate
                        if verbose:
                            eprint(f"[skip] gap={fmt_bytes(gap)} at offsets [{pos}, {candidate})")

                    frame_size, err = libz.find_frame_compressed_size(mm, candidate, file_size)
                    if frame_size is None or frame_size <= 0:
                        rejected_candidates += 1
                        if verbose:
                            eprint(f"[reject] offset={candidate}: could not determine compressed size ({err})")
                        pos = candidate + 1
                        scanned_bytes = pos
                        continue

                    frame_end = candidate + frame_size
                    if frame_end > file_size:
                        rejected_candidates += 1
                        if verbose:
                            eprint(f"[reject] offset={candidate}: frame overruns file end (size={frame_size})")
                        pos = candidate + 1
                        scanned_bytes = pos
                        continue

                    frame_bytes = bytes(mm[candidate:frame_end])

                    if verbose:
                        frame_dict_id = get_frame_dict_id(frame_bytes)
                        eprint(
                            f"[candidate] offset={candidate}, "
                            f"compressed_size={fmt_bytes(frame_size)}, frame_dict_id={frame_dict_id}"
                        )

                    ok, mode_name, dict_id, verify_err = verify_frame_with_dict_candidates(
                        frame_bytes, dict_candidates, verbose=verbose
                    )

                    if ok:
                        outf.write(frame_bytes)
                        recovered_members += 1
                        pos = frame_end
                        scanned_bytes = pos
                        if verbose:
                            eprint(
                                f"[recover] member=#{recovered_members}, offset={candidate}, "
                                f"size={fmt_bytes(frame_size)}, dict_mode={mode_name}, dict_id={dict_id}"
                            )
                    else:
                        rejected_candidates += 1
                        pos = candidate + 1
                        scanned_bytes = pos
                        if verbose:
                            eprint(f"[reject] offset={candidate}: decompression test failed ({verify_err})")

    elapsed = time.time() - start_time
    return {
        "input": input_path,
        "output": output_path,
        "input_size": file_size,
        "prefix_size": len(prefix_bytes),
        "usable_dict_size": len(usable_dict_bytes),
        "recovered_members": recovered_members,
        "rejected_candidates": rejected_candidates,
        "skipped_bytes": skipped_bytes,
        "elapsed_sec": elapsed,
    }


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Recover valid concatenated Zstd members from a damaged file whose beginning "
            "contains an embedded custom dictionary container."
        )
    )
    ap.add_argument("input", help="Input damaged Zstd file")
    ap.add_argument("output", help="Recovered output file")
    ap.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed progress and recovery logs to stderr",
    )
    ap.add_argument(
        "--progress-every-mb",
        type=int,
        default=256,
        help="Emit periodic progress logs every N MiB scanned when --verbose is enabled (default: 256)",
    )
    args = ap.parse_args()

    try:
        stats = recover_file(
            input_path=args.input,
            output_path=args.output,
            verbose=args.verbose,
            progress_every_mb=args.progress_every_mb,
        )
    except Exception as e:
        eprint(f"ERROR: {e}")
        sys.exit(1)

    print(
        f"Recovered {stats['recovered_members']} member(s); "
        f"preserved prefix={stats['prefix_size']} bytes; "
        f"usable dictionary={stats['usable_dict_size']} bytes; "
        f"rejected candidates={stats['rejected_candidates']}; "
        f"skipped damaged/noise bytes={stats['skipped_bytes']}; "
        f"elapsed={stats['elapsed_sec']:.2f}s"
    )


if __name__ == "__main__":
    main()
