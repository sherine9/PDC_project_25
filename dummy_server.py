# dummy_server.py

from flask import Flask, request, jsonify
import numpy as np
import random

app = Flask(__name__)

G = 10
sigma2 = 10

def channel(x: np.ndarray) -> np.ndarray:
    """Randomly choose odd/even boost, add AWGN."""
    s = random.choice([1, 2])
    n = x.size
    Y = np.random.normal(0, np.sqrt(sigma2), n)
    if s == 1:
        # even indices boosted
        Y[::2] += x[::2] * np.sqrt(G)
        Y[1::2] += x[1::2]
    else:
        # odd  indices boosted
        Y[::2] += x[::2]
        Y[1::2] += x[1::2] * np.sqrt(G)
    return Y

@app.route('/channel', methods=['POST'])
def channel_route():
    data = request.get_json()
    x = np.array(data['x'], dtype=float)
    Y = channel(x)
    return jsonify({'Y': Y.tolist()})

if __name__ == '__main__':
    # starts on http://localhost:5000
    app.run(host='0.0.0.0', port=8000)
