# dummy_client.py

"""
Instruction constraints:
- Block length n must be even and ≤ 1,000,000.
- Transmitted energy abs(x)**2 must be ≤ 2000.
- Text messages are exactly 40 characters from the 64-symbol alphabet.
- Alphabet size = 64.
- Up to two transmission attempts allowed.
"""

import numpy as np
import requests
import galois
from scipy.linalg import hadamard

# ────────────── parameters ──────────────
SERVER_URL = 'http://localhost:8000/channel'
n          = 128                       # total chips per symbol
G, sigma2  = 10, 10
CHARSET    = list(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789 ."
)
max_energy = 2000.0                   # max energy per symbol

# verify n is power of two and derive half-size
m_full = int(np.log2(n))
assert 2**m_full == n, "n must be a power of 2"
m_half = m_full - 1   # for a 64×64 Hadamard when n=128

# build a balanced 64×128 codebook
def build_codebook(n: int) -> np.ndarray:
    H64 = hadamard(2**m_half)         # 2^(m_half)=64
    scale = np.sqrt((max_energy*0.9995) / (63 * n))       # meets ‖x‖²=2000 over 63 syms
    C = np.zeros((64, n))
    for i in range(64):
        row = H64[i] * scale                 # length-64 ±scale
        C[i, ::2] = row                      # even chips
        C[i, 1::2] = row                     # odd  chips
    return C

CODE64 = build_codebook(n)                  # 64×128 balanced codebook

# ────────────── Reed–Solomon setup ──────────────
GF  = galois.GF(2**6)
RS63 = galois.ReedSolomon(63, 40, field=GF)  # systematic (n=63, k=40)

def rs_encode(msg40: str):
    idx = [CHARSET.index(c) for c in msg40]
    return RS63.encode(GF(idx))

def rs_decode(decisions):
    cw  = GF(decisions)
    dec = RS63.decode(cw)                   # may raise if >11 errors
    return "".join(CHARSET[int(s)] for s in dec)

# ────────────── transmitter & receiver ──────────────
def tx(message40: str) -> np.ndarray:
    if len(message40) != 40:
        raise ValueError("Message must be exactly 40 chars.")
    cw = rs_encode(message40)               # GF(64) symbols, length 63
    blocks = [CODE64[int(sym)] for sym in cw]
    return np.concatenate(blocks)           # length 63·128 = 8064

def remote_channel(x: np.ndarray) -> np.ndarray:
    r = requests.post(SERVER_URL, json={'x': x.tolist()})
    r.raise_for_status()
    return np.array(r.json()['Y'], dtype=float)

def rx(Y: np.ndarray) -> str:
    # 1) detect which half is boosted (global test)
    if np.mean(Y[::2]**2) > np.mean(Y[1::2]**2):
        C = CODE64.copy()
        C[:, ::2] *= np.sqrt(G)
    else:
        C = CODE64.copy()
        C[:, 1::2] *= np.sqrt(G)

    # 2) matched‐filter each of the 63 blocks
    decisions = []
    for i in range(63):
        blk = Y[i*n:(i+1)*n]
        corr = C @ blk                      # (64,) correlations
        decisions.append(int(np.argmax(corr)))
    return rs_decode(decisions)

if __name__ == '__main__':
    message = "Hello World 123 ABC lets code in Python."
    assert len(message) == 40

    print("TX message:", message)
    x = tx(message)
    Y = remote_channel(x)
    out = rx(Y)
    print("RX message:", out)
    print("Match?    ", "✔" if out == message else "✘")
