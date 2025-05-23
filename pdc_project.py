import numpy as np
import random

# -----------------------------------
# Single-Transmission + 200-Trial Full-Message Accuracy Test
# -----------------------------------

# 1. Parameters and Constraints
R = 6                       # Matrix order → block length n = 2^(R+1) = 128
n = 2 ** (R + 1)
assert n % 2 == 0 and n <= 1_000_000
ALPHA = 2000.0 / (40 * n)   # Max α so that ‖x‖² ≤ 2000
G = 10                      # Channel boost factor
sigma2 = 10                 # Noise variance
CHARSET = list(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789 ."
)
assert len(CHARSET) == 64, "Charset must be 64 symbols."

# 2. Build Codebook
def hadamard_recursive(r):
    if r == 0:
        return np.array([[1]], int)
    H = hadamard_recursive(r - 1)
    return np.vstack((np.hstack((H, H)), np.hstack((H, -H))))

Mr = hadamard_recursive(R)
Br = np.vstack((Mr, -Mr))
CODEBOOK = np.sqrt(ALPHA) * np.hstack((Br, Br))[:64]  # shape (64, n)

# 3. Encoder
def encode_message(msg: str) -> np.ndarray:
    if len(msg) != 40:
        raise ValueError("Message must be exactly 40 characters.")
    x = np.concatenate([CODEBOOK[CHARSET.index(ch)] for ch in msg]).astype(float)
    assert np.sum(x*2) <= 2000 + 1e-6, f"Energy {np.sum(x*2):.2f} > 2000"
    return x

# 4. Specified Channel
def channel(x: np.ndarray) -> np.ndarray:
    s = random.choice([1, 2])
    Y = np.random.normal(0, np.sqrt(sigma2), x.size)
    if s == 1:
        Y[::2] += x[::2] * np.sqrt(G)
        Y[1::2] += x[1::2]
    else:
        Y[::2] += x[::2]
        Y[1::2] += x[1::2] * np.sqrt(G)
    return Y

# 5. Boost-Aware Decoder
def decode_message(y: np.ndarray) -> str:
    decoded = []
    for blk in y.reshape(40, n):
        C0 = CODEBOOK.copy(); C1 = CODEBOOK.copy()
        C0[:, ::2] *= np.sqrt(G)    # even positions boosted
        C1[:, 1::2] *= np.sqrt(G)   # odd positions boosted
        m0 = C0 @ blk; m1 = C1 @ blk
        idx = np.argmax(np.maximum(m0, m1))
        decoded.append(CHARSET[idx])
    return "".join(decoded)

# 6. Test on Single Transmission
message = "Hello World 123 ABC lets code in Python."
assert len(message) == 40

num_trials = 200
full_success = 0
total_correct_chars = 0

# * DO NOT reseed here! *

for _ in range(num_trials):
    x = encode_message(message)
    y = channel(x)
    decoded = decode_message(y)
    print(decoded)
    if decoded == message:
        full_success += 1
    total_correct_chars += sum(dc == oc for dc, oc in zip(decoded, message))

char_recovery_pct = total_correct_chars / (num_trials * len(message)) * 100
full_recovery_pct = full_success / num_trials * 100

print(f"Full-message recovery rate over {num_trials} trials: {full_recovery_pct:.2f}%")
print(f"Character recovery rate:                 {char_recovery_pct:.2f}%")