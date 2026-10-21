"""Pure-Python ISO/IEC 18004 QR Code SVG generator for TOTP MFA provisioning.

Generates crisp, standards-compliant SVG QR codes (Versions 1-10, Error Correction
Level L, Byte Mode) with zero external third-party dependencies.
"""

from __future__ import annotations

import base64
from typing import List, Optional, Tuple

# Version 1..10 capabilities for Error Correction Level L:
# (total_codewords, ec_codewords_per_block, num_blocks_group1, data_codewords_group1, num_blocks_group2, data_codewords_group2)
_VERSION_TABLE_L: List[Tuple[int, int, int, int, int, int]] = [
    (0, 0, 0, 0, 0, 0),  # dummy index 0
    (26, 7, 1, 19, 0, 0),  # v1: 17 bytes max
    (44, 10, 1, 34, 0, 0),  # v2: 32 bytes max
    (70, 15, 1, 55, 0, 0),  # v3: 53 bytes max
    (100, 20, 1, 80, 0, 0),  # v4: 78 bytes max
    (134, 26, 1, 108, 0, 0),  # v5: 106 bytes max
    (172, 18, 2, 68, 0, 0),  # v6: 134 bytes max
    (196, 20, 2, 78, 0, 0),  # v7: 154 bytes max
    (242, 24, 2, 97, 0, 0),  # v8: 192 bytes max
    (292, 30, 2, 116, 0, 0),  # v9: 230 bytes max
    (346, 18, 2, 68, 2, 69),  # v10: 271 bytes max
]

# Alignment pattern center coordinates for Versions 1..10
_ALIGNMENT_COORDS: List[List[int]] = [
    [],
    [],  # v1
    [6, 18],  # v2
    [6, 22],  # v3
    [6, 26],  # v4
    [6, 30],  # v5
    [6, 34],  # v6
    [6, 22, 38],  # v7
    [6, 24, 42],  # v8
    [6, 26, 46],  # v9
    [6, 28, 50],  # v10
]

# Remainder bits required after final codeword for Versions 1..10
_REMAINDER_BITS = [0, 0, 7, 7, 7, 7, 7, 0, 0, 0, 0]

# Precompute GF(256) exp and log tables with primitive polynomial 0x11D
_GF_EXP = [0] * 512
_GF_LOG = [0] * 256


def _init_gf() -> None:
    x = 1
    for i in range(255):
        _GF_EXP[i] = x
        _GF_LOG[x] = i
        x <<= 1
        if x & 0x100:
            x ^= 0x11D
    for i in range(255, 512):
        _GF_EXP[i] = _GF_EXP[i - 255]


_init_gf()


def _gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _GF_EXP[_GF_LOG[a] + _GF_LOG[b]]


def _rs_generator_poly(degree: int) -> List[int]:
    """Compute Reed-Solomon generator polynomial of given degree."""
    g = [1]
    for i in range(degree):
        nxt = [0] * (len(g) + 1)
        factor = _GF_EXP[i]
        for j, coeff in enumerate(g):
            nxt[j] ^= coeff
            nxt[j + 1] ^= _gf_mul(coeff, factor)
        g = nxt
    return g


def _rs_encode(data: List[int], ec_len: int) -> List[int]:
    """Compute `ec_len` Reed-Solomon error-correction bytes for `data`."""
    gen = _rs_generator_poly(ec_len)
    rem = [0] * ec_len
    for b in data:
        factor = b ^ rem[0]
        rem = rem[1:] + [0]
        if factor != 0:
            for i in range(ec_len):
                rem[i] ^= _gf_mul(gen[i + 1], factor)
    return rem


def _select_version(byte_len: int) -> int:
    for v in range(1, len(_VERSION_TABLE_L)):
        _, _, b1, d1, b2, d2 = _VERSION_TABLE_L[v]
        total_data_cw = b1 * d1 + b2 * d2
        # Mode (4 bits) + char count (8 bits for v1-9, 16 bits for v10)
        header_bytes = 2 if v <= 9 else 3
        if byte_len + header_bytes <= total_data_cw:
            return v
    raise ValueError(f"Payload too long for QR versions 1-10 ({byte_len} bytes)")


def _encode_data_codewords(payload: bytes, version: int) -> List[int]:
    _, _, b1, d1, b2, d2 = _VERSION_TABLE_L[version]
    total_data_cw = b1 * d1 + b2 * d2
    bits: List[int] = []

    def append_bits(val: int, length: int) -> None:
        for i in range(length - 1, -1, -1):
            bits.append((val >> i) & 1)

    # Byte mode indicator: 0100
    append_bits(0b0100, 4)
    # Character count indicator: 8 bits for v1-9, 16 bits for v10
    count_bits = 8 if version <= 9 else 16
    append_bits(len(payload), count_bits)

    for b in payload:
        append_bits(b, 8)

    # Terminator: up to 4 zero bits
    max_bits = total_data_cw * 8
    term_len = min(4, max_bits - len(bits))
    append_bits(0, term_len)

    # Pad to byte boundary
    while len(bits) % 8 != 0:
        bits.append(0)

    codewords: List[int] = []
    for i in range(0, len(bits), 8):
        byte_val = 0
        for bit in bits[i : i + 8]:
            byte_val = (byte_val << 1) | bit
        codewords.append(byte_val)

    # Pad bytes 0xEC, 0x11
    pad_seq = [0xEC, 0x11]
    pad_idx = 0
    while len(codewords) < total_data_cw:
        codewords.append(pad_seq[pad_idx % 2])
        pad_idx += 1

    return codewords


def _interleave_codewords(data_cw: List[int], version: int) -> List[int]:
    _, ec_len, b1, d1, b2, d2 = _VERSION_TABLE_L[version]
    blocks_data: List[List[int]] = []
    blocks_ec: List[List[int]] = []

    idx = 0
    for _ in range(b1):
        blk = data_cw[idx : idx + d1]
        idx += d1
        blocks_data.append(blk)
        blocks_ec.append(_rs_encode(blk, ec_len))

    for _ in range(b2):
        blk = data_cw[idx : idx + d2]
        idx += d2
        blocks_data.append(blk)
        blocks_ec.append(_rs_encode(blk, ec_len))

    result: List[int] = []
    max_data_len = max(d1, d2)
    for i in range(max_data_len):
        for blk in blocks_data:
            if i < len(blk):
                result.append(blk[i])

    for i in range(ec_len):
        for ec_blk in blocks_ec:
            result.append(ec_blk[i])

    return result


def _format_bits_ec_l(mask: int) -> int:
    """Compute 15-bit BCH format info for EC Level L (01) and `mask` (0..7)."""
    data = (0b01 << 3) | (mask & 0b111)
    rem = data << 10
    for i in range(4, -1, -1):
        if rem & (1 << (i + 10)):
            rem ^= 0x537 << i
    bits = ((data << 10) | rem) ^ 0x5412
    return bits


def _version_bits(version: int) -> int:
    """Compute 18-bit BCH version info for Versions 7..10."""
    rem = version << 12
    for i in range(5, -1, -1):
        if rem & (1 << (i + 12)):
            rem ^= 0x1F25 << i
    return (version << 12) | rem


def _build_matrix(version: int, codewords: List[int]) -> List[List[int]]:
    size = version * 4 + 17
    matrix: List[List[Optional[int]]] = [[None] * size for _ in range(size)]
    is_function: List[List[bool]] = [[False] * size for _ in range(size)]

    def set_func(r: int, c: int, val: int) -> None:
        if 0 <= r < size and 0 <= c < size:
            matrix[r][c] = val
            is_function[r][c] = True

    # 1. Finder patterns + separators
    def place_finder(top: int, left: int) -> None:
        for dr in range(-1, 8):
            for dc in range(-1, 8):
                r, c = top + dr, left + dc
                if 0 <= r < size and 0 <= c < size:
                    if 0 <= dr <= 6 and 0 <= dc <= 6:
                        if (
                            dr in (0, 6)
                            or dc in (0, 6)
                            or (2 <= dr <= 4 and 2 <= dc <= 4)
                        ):
                            set_func(r, c, 1)
                        else:
                            set_func(r, c, 0)
                    else:
                        set_func(r, c, 0)

    place_finder(0, 0)
    place_finder(0, size - 7)
    place_finder(size - 7, 0)

    # 2. Alignment patterns
    coords = _ALIGNMENT_COORDS[version]
    for ar in coords:
        for ac in coords:
            # Skip if overlapping any of the 3 finder patterns
            if (
                (ar <= 8 and ac <= 8)
                or (ar <= 8 and ac >= size - 8)
                or (ar >= size - 8 and ac <= 8)
            ):
                continue
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    dist = max(abs(dr), abs(dc))
                    set_func(ar + dr, ac + dc, 1 if dist in (0, 2) else 0)

    # 3. Timing patterns
    for i in range(8, size - 8):
        bit = 1 if i % 2 == 0 else 0
        if not is_function[6][i]:
            set_func(6, i, bit)
        if not is_function[i][6]:
            set_func(i, 6, bit)

    # 4. Dark module & reserved format areas
    set_func(4 * version + 9, 8, 1)
    for i in range(9):
        if not is_function[8][i]:
            set_func(8, i, 0)
        if not is_function[i][8]:
            set_func(i, 8, 0)
    for i in range(8):
        if not is_function[8][size - 1 - i]:
            set_func(8, size - 1 - i, 0)
        if not is_function[size - 1 - i][8]:
            set_func(size - 1 - i, 8, 0)

    # 5. Version info (for v >= 7)
    if version >= 7:
        vbits = _version_bits(version)
        for i in range(18):
            bit = (vbits >> i) & 1
            r = i // 3
            c = size - 11 + (i % 3)
            set_func(r, c, bit)
            set_func(c, r, bit)

    # 6. Data bits placement
    bit_stream: List[int] = []
    for cw in codewords:
        for i in range(7, -1, -1):
            bit_stream.append((cw >> i) & 1)
    for _ in range(_REMAINDER_BITS[version]):
        bit_stream.append(0)

    bit_idx = 0
    upward = True
    col = size - 1
    while col > 0:
        if col == 6:
            col -= 1
        row_iter = range(size - 1, -1, -1) if upward else range(size)
        for r in row_iter:
            for dc in (0, -1):
                c = col + dc
                if not is_function[r][c]:
                    bit = bit_stream[bit_idx] if bit_idx < len(bit_stream) else 0
                    matrix[r][c] = bit
                    bit_idx += 1
        upward = not upward
        col -= 2

    # 7. Apply Mask 0 ((r + c) % 2 == 0) and write format info
    mask_id = 0
    for r in range(size):
        for c in range(size):
            if not is_function[r][c]:
                if (r + c) % 2 == 0:
                    matrix[r][c] = (matrix[r][c] or 0) ^ 1

    fbits = _format_bits_ec_l(mask_id)
    # Place 15 format bits (bit 0 = LSB, bit 14 = MSB) around top-left finder
    tl_coords_lsb_to_msb = [
        (0, 8),
        (1, 8),
        (2, 8),
        (3, 8),
        (4, 8),
        (5, 8),
        (7, 8),
        (8, 8),
        (8, 7),
        (8, 5),
        (8, 4),
        (8, 3),
        (8, 2),
        (8, 1),
        (8, 0),
    ]
    for bit_idx, (r, c) in enumerate(tl_coords_lsb_to_msb):
        matrix[r][c] = (fbits >> bit_idx) & 1

    # Second copy of format bits:
    # Bits 0..7 along top-right horizontal (row 8, cols size-1 down to size-8)
    for bit_idx in range(8):
        matrix[8][size - 1 - bit_idx] = (fbits >> bit_idx) & 1
    # Bits 8..14 along bottom-left vertical (col 8, rows size-7 down to size-1)
    for bit_idx in range(8, 15):
        matrix[size - 15 + bit_idx][8] = (fbits >> bit_idx) & 1

    return [[int(cell or 0) for cell in row] for row in matrix]


def generate_qr_matrix(text: str) -> List[List[int]]:
    """Encode `text` into a 2D QR matrix of 0/1 ints."""
    payload = text.encode("utf-8")
    version = _select_version(len(payload))
    data_cw = _encode_data_codewords(payload, version)
    full_cw = _interleave_codewords(data_cw, version)
    return _build_matrix(version, full_cw)


def generate_qr_svg(text: str, border: int = 4) -> str:
    """Return a compact, crisp SVG string representing the QR code for `text`."""
    mat = generate_qr_matrix(text)
    n = len(mat)
    total = n + border * 2
    path_parts: List[str] = []
    for r, row in enumerate(mat):
        y = r + border
        c = 0
        while c < n:
            if row[c]:
                run = 1
                while c + run < n and row[c + run]:
                    run += 1
                x = c + border
                path_parts.append(f"M{x},{y}h{run}v1h-{run}z")
                c += run
            else:
                c += 1
    path_d = "".join(path_parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total} {total}" '
        f'shape-rendering="crispEdges">'
        f'<rect width="{total}" height="{total}" fill="#FFFFFF"/>'
        f'<path d="{path_d}" fill="#0F172A"/>'
        f"</svg>"
    )


def generate_qr_data_uri(text: str, border: int = 4) -> str:
    """Return a `data:image/svg+xml;base64,...` URI for embedding in `<img src=...>`."""
    svg = generate_qr_svg(text, border=border)
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"

