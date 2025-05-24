# test_client.py

import random
from dummy_client import tx, remote_channel, rx, CHARSET
from tqdm.auto import tqdm

def random_message() -> str:
    """Generate a random 40-char message from the allowed charset."""
    return ''.join(random.choice(CHARSET) for _ in range(40))

def main(trials: int = 200):
    success = 0
    for i in tqdm(range(1, trials+1)):
        msg = random_message()
        x   = tx(msg)
        Y   = remote_channel(x)
        out = rx(Y)
        if out == msg:
            success += 1
        # optional: print progress every 50 trials
        if i % 50 == 0:
            print(f"  → {i} trials done, success so far: {success}/{i} = {success/i:.2%}")
    print()
    print(f"Final result: {success}/{trials} messages passed "
          f"({success/trials*100:.2f}%)")

if __name__ == "__main__":
    main(trials=10000)
