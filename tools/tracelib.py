"""Shared helpers for Accel-Sim text traces (.traceg / .traceg.xz), tracer versions 3-6.

Instruction line grammar (Accel-Sim tracer >= 3; see trace_parser.cc, inst_trace_t::parse_from_string):

    [line_num] PC mask dest_num [dests] opcode src_num [srcs] [gmma_flag] mem_width
               [addr_mode addresses...] imm [imm2 Val|NoVal [values...]]      (v6 adds imm2/Val)

Address modes (only present when mem_width > 0), n = popcount(mask) active lanes:
    0 list_all     n hex addresses, one per active lane
    1 base_stride  hex base + decimal stride (tracer uses it only for contiguous masks)
    2 base_delta   hex base + (n-1) decimal deltas, each relative to the previous active lane

The decoding is the same as accel_sim_hbf/experiments/23_qwen3_1p7b/prepare_probe.py; re-encoding keeps
mode 1/2 compression when every lane is shifted by the same amount (only the base changes) and falls back
to mode 0 otherwise. Trailing tokens (immediates, v6 register values) are kept unchanged.
TMA instructions (UTMA*/UBLK*, Hopper) use another grammar; parse_mem() returns kind 'tma' for them and
the callers leave them untouched (none of our sm_75/sm_86 traces contain them).
"""
import hashlib
import lzma
import re
from pathlib import Path

HEX = re.compile(r'[0-9a-fA-F]+$')
TMA_OPS = ('UTMALDG', 'UTMASTG', 'UTMAPF', 'UTMAREDG', 'UBLKCP', 'UBLKPF', 'UBLKRED')
# Opcodes whose addresses are not global memory (shared / local / constant windows).
NON_GLOBAL_PREFIX = ('LDS', 'STS', 'LDL', 'STL', 'ATOMS', 'LDC', 'LDSM')


def open_text(path, mode='rt'):
    """Open a text trace; .xz is (de)compressed on the fly (never written decompressed to disk)."""
    path = Path(path)
    if path.suffix == '.tracez':
        raise ValueError(f'{path}: binary .tracez, convert it first with tools/bin/traceDsm')
    if path.suffix == '.xz':
        return lzma.open(path, mode, preset=1) if 'w' in mode else lzma.open(path, mode)
    return open(path, mode)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def kernel_files(kernelslist):
    """Kernel trace file names of a kernelslist.g, in launch order (MemcpyHtoD lines skipped)."""
    return [x.strip() for x in Path(kernelslist).read_text().splitlines() if x.startswith('kernel')]


def is_store(op):
    return op.startswith(('STG', 'RED', 'ATOM')) or op == 'ST' or op.startswith('ST.')


def is_global(op):
    return not op.startswith(NON_GLOBAL_PREFIX)


class MemInst:
    """Decoded memory instruction: tokens t, opcode, width, addr mode, token span [idx, end), lane addrs."""
    __slots__ = ('t', 'op', 'pc', 'width', 'mode', 'wi', 'idx', 'end', 'addrs', 'kind')


def parse_mem(t, lineinfo=False):
    """t = line.split(). Returns a MemInst for an instruction with addresses, else None."""
    base = 1 if lineinfo else 0
    if len(t) < base + 6 or not HEX.match(t[base]) or not HEX.match(t[base + 1]):
        return None
    try:
        oi = base + 3 + int(t[base + 2])
        op = t[oi]
        wi = oi + 2 + int(t[oi + 1])
        if 'GMMA' in op:
            wi += 1
        width = int(t[wi])
    except (ValueError, IndexError):
        return None
    if width == 0:
        return None
    m = MemInst()
    m.t, m.op, m.pc, m.width, m.wi = t, op, int(t[base], 16), width, wi
    if op.startswith(TMA_OPS):
        m.kind, m.mode, m.idx, m.end, m.addrs = 'tma', None, None, None, []
        return m
    m.kind = 'mem'
    m.mode = int(t[wi + 1])
    m.idx = idx = wi + 2
    n = bin(int(t[base + 1], 16)).count('1')
    if n == 0:
        m.addrs, m.end = [], idx
    elif m.mode == 0:
        m.addrs = [int(x, 16) for x in t[idx:idx + n]]
        m.end = idx + n
    elif m.mode == 1:
        b, s = int(t[idx], 16), int(t[idx + 1])
        m.addrs = [b + i * s for i in range(n)]
        m.end = idx + 2
    elif m.mode == 2:
        a = [int(t[idx], 16)]
        for d in t[idx + 1:idx + n]:
            a.append(a[-1] + int(d))
        m.addrs = a
        m.end = idx + n
    else:
        raise ValueError(f'unknown address mode {m.mode}: {" ".join(t)}')
    return m


def encode(m, new):
    """Return the token list of m with its lane addresses replaced by `new` (same order/length)."""
    t = m.t
    shifts = {b - a for a, b in zip(m.addrs, new)}
    if m.mode in (1, 2) and len(shifts) == 1:
        t = list(t)
        t[m.idx] = hex(new[0])  # uniform shift: strides / deltas are unchanged
        return t
    return t[:m.wi + 1] + ['0'] + [hex(x) for x in new] + t[m.end:]


def header_value(line, key):
    """'-kernel id = 4659' -> '4659' for key 'kernel id'; None if the line is another header."""
    if line.startswith('-' + key + ' ='):
        return line.split('=', 1)[1].strip()
    return None
