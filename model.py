"""
Tiny GPT From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_vocab
def build_vocab(text):
    """Return a sorted list of unique characters in text."""
    return sorted(set(text))

# Step 2 - build_stoi
def build_stoi(vocab):
    """Return a dict mapping each character in vocab to its index."""
    stoi = {}
    for idx, ch in enumerate(vocab):
        stoi[ch] = idx

    return stoi

# Step 3 - build_itos
def build_itos(vocab):
    """Return a dict mapping each index 0..len(vocab)-1 to its character."""
    itos = {}
    for idx, ch in enumerate(vocab):
        itos[idx] = ch
    
    return itos

# Step 4 - encode_char
def encode_char(ch, stoi):
    """Return the integer token id for a single character ch using stoi."""
    return stoi[ch]

# Step 5 - encode_string
def encode_string(text, stoi):
    """Encode a full string into a list of token ids using stoi."""
    return [encode_char(ch, stoi) for ch in text]

# Step 6 - decode_int
def decode_int(token_id, itos):
    """Return the single character mapped to token_id by itos."""
    return itos[token_id]

# Step 7 - decode_ids
def decode_ids(ids, itos):
    """Decode a list of token ids into a string using itos."""
    return ''.join([decode_int(token_id, itos) for token_id in ids])

# Step 8 - make_1d_array
import numpy as np

def make_1d_array(values):
    """Create a 1D NumPy array from a Python list of numbers."""
    return np.array(values)

# Step 9 - get_array_shape
import numpy as np

def get_array_shape(arr):
    """Return the shape tuple of a NumPy array."""
    return arr.shape

# Step 10 - get_array_dtype
import numpy as np

def get_array_dtype(arr):
    """Return the dtype of a NumPy array."""
    return arr.dtype

# Step 11 - make_2d_zeros
import numpy as np

def make_2d_zeros(rows, cols):
    """Return a 2D NumPy array of zeros with shape (rows, cols)."""
    return np.zeros((rows, cols))

# Step 12 - make_2d_random
import numpy as np

def make_2d_random(rows, cols, seed):
    """Return a (rows, cols) array of uniform floats in [0, 1) seeded by `seed`."""
    rng = np.random.default_rng(seed=seed)
    return rng.random(size=(rows, cols))

# Step 13 - index_element
def index_element(arr, i, j):
    """Return the scalar element at position (i, j) of a 2D array."""
    return arr[i, j]

# Step 14 - slice_row
import numpy as np

def slice_row(arr, i):
    """Return row i of a 2D array as a 1D view."""
    return arr[i, :]

# Step 15 - slice_column
import numpy as np

def slice_column(arr, j):
    """Return column j of a 2D array as a 1D array of length R."""
    return arr[:, j]

# Step 16 - slice_subblock
import numpy as np

def slice_subblock(arr, r0, r1, c0, c1):
    """Return the sub-block arr[r0:r1, c0:c1] of a 2D array."""
    return arr[r0:r1, c0:c1]

# Step 17 - elementwise_add
import numpy as np

def elementwise_add(a, b):
    """Return the elementwise sum of two same-shape arrays."""
    return a+b

# Step 18 - elementwise_multiply
import numpy as np

def elementwise_multiply(a, b):
    """Return the elementwise product of two same-shape arrays."""
    return a*b

# Step 19 - scalar_broadcast_add
import numpy as np

def scalar_broadcast_add(arr, scalar):
    """Return a new array equal to arr with scalar added to every element."""
    return arr + scalar

# Step 20 - vector_matrix_broadcast_add
import numpy as np

def vector_matrix_broadcast_add(matrix, vector):
    """Add a 1D vector to each row of a 2D matrix via broadcasting."""
    return matrix + vector.reshape(1, -1)

# Step 21 - array_exp
import numpy as np

def array_exp(arr):
    """Return the elementwise exponential of arr."""
    return np.exp(arr)

# Step 22 - array_log
import numpy as np

def array_log(arr):
    """Return the elementwise natural log of arr (assumes arr > 0)."""
    return np.log(arr)

# Step 23 - sum_all
import numpy as np

def sum_all(arr):
    """Return the sum of every element of arr as a scalar."""
    return np.sum(arr)

# Step 24 - sum_axis0
import numpy as np

def sum_axis0(arr):
    """Sum a 2D array along axis 0, collapsing rows into a 1D vector of column sums."""
    return np.sum(arr, axis=0)

# Step 25 - sum_axis1
import numpy as np

def sum_axis1(arr):
    """Sum a 2D array along axis 1, returning a 1D array of row sums."""
    return np.sum(arr, axis=1)

# Step 26 - max_along_axis
import numpy as np

def max_along_axis(arr, axis):
    """Return the maximum of arr along the given axis, with that axis removed."""
    return np.max(arr, axis=axis)

# Step 27 - matmul
import numpy as np

def matmul(a, b):
    """Return the matrix product a @ b for 2D arrays a (M,K) and b (K,N)."""
    return a @ b

# Step 28 - transpose_matrix
def transpose_matrix(arr):
    """Return the transpose of a 2D array."""
    return arr.T

# Step 29 - sum_keepdims
import numpy as np

def sum_keepdims(arr, axis):
    """Sum along `axis` while keeping that dimension as size 1."""
    return np.sum(arr, axis, keepdims=True)

# Step 30 - naive_softmax_1d
import numpy as np

def naive_softmax_1d(logits):
    """Compute softmax of a 1D logits vector via the direct exp/sum formula."""
    return array_exp(logits)/sum_all(array_exp(logits))

# Step 31 - softmax_overflow_demo
import math

def softmax_overflow_demo(large_value):
    """Show that naive exp overflows on a large logit.

    Return {'naive_exp': float, 'overflowed': bool}.
    """
    value = array_exp(large_value)
    return {"naive_exp":value, "overflowed":math.isinf(value)}

# Step 32 - stable_softmax_1d
import numpy as np

def stable_softmax_1d(logits):
    """Numerically stable softmax over a 1D logits vector."""
    max_logits = max_along_axis(logits, axis=0)
    logits_shifted = logits - max_logits
    logits_exp = array_exp(logits_shifted)
    return logits_exp/sum_all(logits_exp)

# Step 33 - stable_softmax_2d_rowwise
import numpy as np

def stable_softmax_2d_rowwise(logits):
    """Row-wise numerically stable softmax of a 2D logits array."""
    logits_shifted = logits - max_along_axis(logits, axis=1).reshape(-1, 1)
    logits_exp = array_exp(logits_shifted)
    return logits_exp/sum_keepdims(logits_exp, axis=1)

# Step 34 - read_text_file
def read_text_file(text_blob):
    """Return text_blob unchanged after validating it is a non-empty string."""
    if not isinstance(text_blob, str):
        raise TypeError("Text must be a non-empty string.")
    if not text_blob:
        raise ValueError("Text must be a non-empty string.")
    return text_blob

# Step 35 - encode_corpus_to_int_array
def encode_corpus_to_int_array(text, stoi):
    """Convert the corpus string into a 1D NumPy int64 array of token ids."""
    return np.array(encode_string(text, stoi), dtype=np.int64)

# Step 36 - pick_split_point
def pick_split_point(n, train_frac):
    """Return integer split index so data[:idx] is train and data[idx:] is val."""
    return int(np.floor(n*train_frac))

# Step 37 - slice_train_and_val
def slice_train_and_val(data, split_idx):
    """Split a 1D token-id array into (train, val) at split_idx."""
    return (data[:split_idx], data[split_idx:])

# Step 38 - pick_block_size
def pick_block_size(default_size):
    """Return the context length (block_size) for training windows."""
    return default_size if default_size > 1 else 1

# Step 39 - slice_x_at_offset
import numpy as np

def slice_x_at_offset(data, i, block_size):
    """Return the input window data[i : i + block_size]."""
    return data[i:i+block_size]

# Step 40 - slice_y_at_offset
import numpy as np

def slice_y_at_offset(data, i, block_size):
    """Return the target window of length block_size starting at i+1."""
    return data[i+1:i+1+block_size]

# Step 41 - sample_random_batch_offsets
def sample_random_batch_offsets(data_len, block_size, batch_size, rng):
    """Sample batch_size random valid starting offsets for (block_size+1)-windows."""
    return rng.integers(0, data_len-block_size, size=batch_size)

# Step 42 - stack_x_batch
import numpy as np

def stack_x_batch(data, offsets, block_size):
    """Stack per-offset X windows into a 2D batch matrix of shape (B, block_size)."""
    out = []
    for offset in offsets:
        out.append(slice_x_at_offset(data, offset, block_size))
    return np.array(out)

# Step 43 - stack_y_batch
import numpy as np

def stack_y_batch(data, offsets, block_size):
    """Stack per-offset Y windows into a 2D (B, block_size) target matrix."""
    out = []
    for offset in offsets:
        out.append(slice_y_at_offset(data, offset, block_size))
    return np.array(out)

# Step 44 - get_batch
def get_batch(data, block_size, batch_size, rng):
    offsets = sample_random_batch_offsets(len(data), block_size, batch_size, rng)
    X = stack_x_batch(data, offsets, block_size)
    Y = stack_y_batch(data, offsets, block_size)
    return X, Y

# Step 45 - allocate_count_matrix
import numpy as np

def allocate_count_matrix(vocab_size):
    """Allocate a (V, V) integer zero matrix for bigram counts."""
    return np.int64(make_2d_zeros(vocab_size, vocab_size))

# Step 46 - loop_fill_counts
import numpy as np

def loop_fill_counts(n_matrix, data):
    """Increment n_matrix[curr, next] for every consecutive pair in data."""
    for t in range(0, len(data)-1):
        n_matrix[data[t], data[t+1]] += 1
    return n_matrix

# Step 47 - vectorize_counts_add_at
import numpy as np

def vectorize_counts_add_at(vocab_size, data):
    """Build (V, V) bigram counts from a 1D id array using vectorized scatter-add."""
    n_matrix = allocate_count_matrix(vocab_size)
    np.add.at(n_matrix, (data[:-1], data[1:]), 1)
    return n_matrix

# Step 48 - add_one_smoothing
import numpy as np

def add_one_smoothing(n_matrix):
    """Return n_matrix with every entry incremented by 1 (Laplace smoothing)."""
    return n_matrix+1

# Step 49 - row_sums_of_counts
def row_sums_of_counts(n_matrix):
    """Return per-row sums of n_matrix with shape (V, 1)."""
    return sum_keepdims(n_matrix, axis=1)

# Step 50 - normalize_counts_to_probs
def normalize_counts_to_probs(n_matrix):
    """Normalize a (V, V) count matrix into a row-stochastic probability matrix."""
    return n_matrix / row_sums_of_counts(n_matrix)

# Step 51 - sample_next_token
def sample_next_token(p_matrix, current_id, rng):
    """Sample the next token id from P[current_id] using rng."""
    vocab_size = p_matrix.shape[1]
    probs = p_matrix[current_id]
    next_id = rng.choice(vocab_size, p=probs)
    return next_id

# Step 52 - generate_sequence
def generate_sequence(p_matrix, start_id, length, rng):
    """Autoregressively sample `length` token ids from a bigram matrix, starting with `start_id`."""
    sequence = [start_id]
    for l in range(length-1):
        next_id = sample_next_token(p_matrix, sequence[-1], rng)
        sequence.append(next_id)

    return np.array(sequence, dtype=np.int64)

# Step 53 - decode_generated_sequence
def decode_generated_sequence(ids, itos):
    """Decode a generated 1D array/list of token ids into a string via itos."""
    return "".join([itos[id_] for id_ in ids])

# Step 54 - log_prob_of_pair
def log_prob_of_pair(p_matrix, current_id, next_id):
    """Return the log probability of a single (current, next) bigram."""
    return array_log(index_element(p_matrix, current_id, next_id))

# Step 55 - sum_negative_log_probs
def sum_negative_log_probs(p_matrix, data):
    sum_val = 0
    for idx in range(len(data)-1):
        sum_val -= log_prob_of_pair(p_matrix, data[idx], data[idx+1])

    return sum_val

# Step 56 - average_nll
def average_nll(p_matrix, data):
    return sum_negative_log_probs(p_matrix, data)/(len(data)-1)

# Step 57 - initialize_w_random
import numpy as np

def initialize_w_random(vocab_size, rng):
    """Return a (vocab_size, vocab_size) float64 matrix of N(0,1) samples drawn from rng."""
    return rng.standard_normal((vocab_size, vocab_size))

# Step 58 - scale_w_small
import numpy as np

def scale_w_small(w_matrix, scale):
    """Return w_matrix scaled by the given small factor."""
    return elementwise_multiply(w_matrix, scale)

# Step 59 - one_hot_encode_batch
import numpy as np

def one_hot_encode_batch(ids, vocab_size):
    """Convert a 1D array of token ids into a (N, vocab_size) one-hot matrix."""
    N = len(ids)
    result = make_2d_zeros(N, vocab_size)
    result[np.arange(N), ids] = 1
    return result

# Step 60 - forward_logits_onehot
def forward_logits_onehot(onehot, w_matrix):
    return matmul(onehot, w_matrix)

# Step 61 - observe_lookup_equivalence
import numpy as np

def observe_lookup_equivalence(w, ids):
    """Show that one-hot @ W equals W[ids] for a small example.
    Returns a dict with keys 'onehot_result' and 'index_result'.
    """
    onehot = one_hot_encode_batch(ids, len(w))
    return {
            "onehot_result": forward_logits_onehot(onehot, w),
            "index_result": w[ids]
           }

# Step 62 - forward_logits_lookup
def forward_logits_lookup(w, ids):
    """Return logits (B, V) by gathering rows of w at positions ids."""
    return w[ids]

# Step 63 - logits_to_probs_rowwise
def logits_to_probs_rowwise(logits):
    return stable_softmax_2d_rowwise(logits)

# Step 64 - gather_correct_token_probs
def gather_correct_token_probs(probs, targets):
    """Return probs[i, targets[i]] for each i, shape (B,)."""
    token_probs = np.zeros_like(targets, dtype=np.float64)
    for i, target in enumerate(targets):
        token_probs[i] = probs[i, target]
    return token_probs

# Step 65 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(probs, targets):
    """Mean negative log-likelihood over a batch."""
    return -sum_all(array_log(gather_correct_token_probs(probs, targets)))/len(targets)

# Step 66 - derive_dlogits_on_paper
def derive_dlogits_on_paper():
    """Return a string summarizing the derivation of dL/dlogits for mean cross-entropy."""
    return """Derivation of the Cross-Entropy Loss Gradient w.r.t. Logits
===========================================================
1. Definitions:
   - Batch size: B, Vocabulary size: V
   - Logits for sample i: z_i in R^V
   - Softmax probabilities: p_i = softmax(z_i), where p_{i,j} = exp(z_{i,j}) / sum_k(exp(z_{i,k}))
   - Mean cross-entropy loss: L = -(1/B) * sum_{i=1}^B log(p_{i,y_i})

2. Single-sample gradient (dL/dz_{i,j}):
   - Case 1: j = y_i
     d/dz_{i,y_i} [-log(p_{i,y_i})] = -1/p_{i,y_i} * p_{i,y_i} * (1 - p_{i,y_i}) = p_{i,y_i} - 1
   - Case 2: j != y_i
     d/dz_{i,j} [-log(p_{i,y_i})] = -1/p_{i,y_i} * (-p_{i,y_i} * p_{i,j}) = p_{i,j}
   - Unified form: dL_i/dz_{i,j} = p_{i,j} - I[j = y_i]

3. Batch average and final formula:
   - Including the 1/B averaging factor across the entire batch:
     dL/dlogits = (probs - onehot(targets)) / B"""

# Step 67 - compute_dlogits
import numpy as np

def compute_dlogits(probs, targets):
    """Gradient of mean cross-entropy w.r.t. logits. probs: (B,V), targets: (B,)."""
    B = probs.shape[0]
    probs[np.arange(B), targets] -= 1
    return probs/B

# Step 68 - derive_dw_on_paper
def derive_dw_on_paper():
    """Return a short written derivation of dL/dW for the lookup-as-matmul forward."""
    return """Forward: logits = onehot(ids) @ W, equivalently logits[b] = W[ids[b]].
Shapes: ids (B,), onehot O (B, V), W (V, D), logits (B, D), dlogits (B, D).
Chain rule: dL/dW = O.T @ dlogits, shape (V, D).
Since O has a single 1 per row at column ids[b], O.T @ dlogits sums rows of dlogits into rows of dW.
Row v of dW equals the sum of dlogits[b] over all b with ids[b] == v.
Implementation: scatter-add dlogits rows into dW at indices ids."""

# Step 69 - compute_dw_scatter_add
import numpy as np

def compute_dw_scatter_add(ids, dlogits, vocab_size):
    """Scatter-add dlogits rows into dW at positions given by ids."""
    dW = np.zeros((vocab_size, dlogits.shape[1]))
    np.add.at(dW, ids, dlogits)
    return dW

# Step 70 - sgd_update_w
import numpy as np

def sgd_update_w(w, dw, learning_rate):
    """Apply one SGD step."""
    return w - learning_rate*dw

# Step 71 - run_one_training_step
def run_one_training_step(w, ids, targets, learning_rate):
    """Run forward, loss, backward, and SGD update once. Return {'w': new_w, 'loss': float}."""
    logits = forward_logits_lookup(w, ids)
    probs = logits_to_probs_rowwise(logits)
    loss = cross_entropy_loss(probs, targets)
    dlogits = compute_dlogits(probs, targets)
    dw = compute_dw_scatter_add(ids, dlogits, dlogits.shape[1])
    w = sgd_update_w(w, dw, learning_rate)
    return {"loss": loss, "w": w}

# Step 72 - train_neural_bigram_loop
import numpy as np

def train_neural_bigram_loop(w, data, block_size, batch_size, learning_rate, num_steps, log_every):
    """Run the neural bigram training loop and return {'w', 'loss_history'}."""
    (X, Y) = get_batch(data, block_size, batch_size, np.random.default_rng())
    loss_history = []
    for step in range(num_steps):
        batch_loss = 0
        for (x, y) in zip(X, Y):
            input_ids = x.reshape(-1)
            target_ids = y.reshape(-1)
            result = run_one_training_step(w, input_ids, target_ids, learning_rate)
            w = result["w"]
            batch_loss = result["loss"]
        if step % log_every == 0:
            loss_history.append(batch_loss)
    return {"w": w, "loss_history": loss_history}

# Step 73 - sample_from_neural_bigram
import numpy as np

def sample_from_neural_bigram(w, start_id, num_tokens, itos):
    """Generate a string by repeatedly sampling from softmax of W[id]."""
    current_ids = [start_id]
    for token_count in range(num_tokens):
        logits = forward_logits_lookup(w, current_ids)
        probs_last_token = logits_to_probs_rowwise(logits)[-1]
        rng = np.random.default_rng()
        next_token_id = rng.choice(np.arange(len(probs_last_token)), size=1, p=probs_last_token)[0]
        current_ids.append(next_token_id)
    
    return decode_ids(current_ids, itos)

# Step 74 - linear_forward
def linear_forward(x, w):
    y = matmul(x, w)
    cache = {"x": x, "w": w}
    return {"y": y, "cache": cache}

# Step 75 - derive_dx_on_paper
def derive_dx_on_paper():
    """Return notes deriving dL/dX = dY @ W.T for Y = X @ W."""
    return """Y = X @ W
dL/dX = dY @ W.T
shapes: X (B, In), W (In, Out), dY (B, Out) -> dL/dX (B, In)"""

# Step 76 - derive_linear_dw_on_paper
def derive_linear_dw_on_paper():
    """Return a string with the derivation of dL/dW for Y = X @ W."""
    return """Let X have shape (N, D_in), W have shape (D_in, D_out), and Y = X @ W have shape (N, D_out).
Suppose the upstream gradient dY = dL/dY has the same shape as Y.
dL/dW = X.T @ dY"""

# Step 77 - linear_backward_dx
def linear_backward_dx(dy, cache):
    return dy @ cache["w"].T

# Step 78 - linear_backward_dw
def linear_backward_dw(dy, cache):
    """Return dL/dW for a linear layer Y = X @ W."""
    return cache["x"].T @ dy

# Step 79 - bias_add_forward
def bias_add_forward(x, b):
    """Add bias vector b (D,) to every row of x (B, D).

    Returns {'y': ndarray (B, D), 'cache': {'b_shape': tuple}}.
    """
    return {"y": vector_matrix_broadcast_add(x, b), \
            "cache": {"b_shape": b.shape}}

# Step 80 - bias_add_backward_db
def bias_add_backward_db(dy, cache):
    """Compute db from upstream gradient dy for y = x + b."""
    return np.sum(dy, axis=0).reshape(-1)

# Step 81 - relu_forward
def relu_forward(x):
    """Apply elementwise ReLU and cache the input for backward.

    Returns a dict with keys 'y' (activated array) and 'cache' (dict with 'x').
    """
    return {"cache": {"x": x}, "y": np.maximum(0, x)}

# Step 82 - relu_backward
def relu_backward(dy, cache):
    """Backward pass for ReLU. cache['x'] holds the original input."""
    return dy * (cache["x"] > 0).astype(int)

# Step 83 - softmax_cross_entropy_backward (not yet solved)
# TODO: implement

# Step 84 - layernorm_forward_mean (not yet solved)
# TODO: implement

# Step 85 - layernorm_forward_variance (not yet solved)
# TODO: implement

# Step 86 - layernorm_forward_normalize (not yet solved)
# TODO: implement

# Step 87 - layernorm_forward_affine (not yet solved)
# TODO: implement

# Step 88 - layernorm_backward_subtract_mean (not yet solved)
# TODO: implement

# Step 89 - layernorm_backward_divide_std (not yet solved)
# TODO: implement

# Step 90 - layernorm_backward_full (not yet solved)
# TODO: implement

# Step 91 - layernorm_backward_implementation (not yet solved)
# TODO: implement

# Step 92 - create_token_embedding (not yet solved)
# TODO: implement

# Step 93 - token_embedding_forward (not yet solved)
# TODO: implement

# Step 94 - token_embedding_backward (not yet solved)
# TODO: implement

# Step 95 - create_positional_embedding (not yet solved)
# TODO: implement

# Step 96 - slice_positional_embedding (not yet solved)
# TODO: implement

# Step 97 - add_token_and_positional_embeddings (not yet solved)
# TODO: implement

# Step 98 - embedding_sum_backward (not yet solved)
# TODO: implement

# Step 99 - create_qkv_projections (not yet solved)
# TODO: implement

# Step 100 - compute_query (not yet solved)
# TODO: implement

# Step 101 - compute_key (not yet solved)
# TODO: implement

# Step 102 - compute_value (not yet solved)
# TODO: implement

# Step 103 - compute_attention_scores (not yet solved)
# TODO: implement

# Step 104 - scale_attention_scores (not yet solved)
# TODO: implement

# Step 105 - build_causal_mask (not yet solved)
# TODO: implement

# Step 106 - apply_causal_mask (not yet solved)
# TODO: implement

# Step 107 - softmax_attention_weights (not yet solved)
# TODO: implement

# Step 108 - attention_weighted_values (not yet solved)
# TODO: implement

# Step 109 - apply_output_projection (not yet solved)
# TODO: implement

# Step 110 - output_projection_backward (not yet solved)
# TODO: implement

# Step 111 - attention_value_backward (not yet solved)
# TODO: implement

# Step 112 - masked_softmax_backward (not yet solved)
# TODO: implement

# Step 113 - scale_scores_backward (not yet solved)
# TODO: implement

# Step 114 - qk_scores_backward (not yet solved)
# TODO: implement

# Step 115 - qkv_projection_backward (not yet solved)
# TODO: implement

# Step 116 - choose_attention_head_config (not yet solved)
# TODO: implement

# Step 117 - create_multihead_qkv_projections (not yet solved)
# TODO: implement

# Step 118 - create_multihead_output_projection (not yet solved)
# TODO: implement

# Step 119 - reshape_to_heads (not yet solved)
# TODO: implement

# Step 120 - transpose_heads_to_front (not yet solved)
# TODO: implement

# Step 121 - get_multihead_n_heads (not yet solved)
# TODO: implement

# Step 122 - get_multihead_sequence_length (not yet solved)
# TODO: implement

# Step 123 - compute_d_head (not yet solved)
# TODO: implement

# Step 124 - multihead_masked_softmax_scores (not yet solved)
# TODO: implement

# Step 125 - multihead_weighted_sum (not yet solved)
# TODO: implement

# Step 126 - transpose_heads_to_back (not yet solved)
# TODO: implement

# Step 127 - get_multihead_output_sequence_length (not yet solved)
# TODO: implement

# Step 128 - merge_heads_to_d_model (not yet solved)
# TODO: implement

# Step 129 - multihead_output_projection_forward (not yet solved)
# TODO: implement

# Step 130 - multihead_reshape_transpose_backward (not yet solved)
# TODO: implement

# Step 131 - ffn_linear_one_forward (not yet solved)
# TODO: implement

# Step 132 - ffn_activation_forward (not yet solved)
# TODO: implement

# Step 133 - ffn_linear_two_forward (not yet solved)
# TODO: implement

# Step 134 - ffn_backward (not yet solved)
# TODO: implement

# Step 135 - residual_forward (not yet solved)
# TODO: implement

# Step 136 - residual_backward (not yet solved)
# TODO: implement

# Step 137 - pre_layernorm_sublayer_forward (not yet solved)
# TODO: implement

# Step 138 - transformer_block_forward (not yet solved)
# TODO: implement

# Step 139 - transformer_block_backward (not yet solved)
# TODO: implement

# Step 140 - stack_transformer_blocks (not yet solved)
# TODO: implement

# Step 141 - forward_through_all_blocks (not yet solved)
# TODO: implement

# Step 142 - backward_through_all_blocks (not yet solved)
# TODO: implement

# Step 143 - final_layernorm_forward (not yet solved)
# TODO: implement

# Step 144 - lm_head_linear_forward (not yet solved)
# TODO: implement

# Step 145 - full_model_forward (not yet solved)
# TODO: implement

# Step 146 - full_model_backward (not yet solved)
# TODO: implement

# Step 147 - initialize_adam_moments (not yet solved)
# TODO: implement

# Step 148 - initialize_adam_step_counter (not yet solved)
# TODO: implement

# Step 149 - adam_increment_step (not yet solved)
# TODO: implement

# Step 150 - adam_update_first_moment (not yet solved)
# TODO: implement

# Step 151 - adam_update_second_moment (not yet solved)
# TODO: implement

# Step 152 - adam_bias_correction (not yet solved)
# TODO: implement

# Step 153 - adam_parameter_update (not yet solved)
# TODO: implement

# Step 154 - wire_full_training_loop (not yet solved)
# TODO: implement

# Step 155 - logging_and_validation_loss (not yet solved)
# TODO: implement

# Step 156 - encode_prompt (not yet solved)
# TODO: implement

# Step 157 - crop_context_to_block_size (not yet solved)
# TODO: implement

# Step 158 - forward_to_get_logits (not yet solved)
# TODO: implement

# Step 159 - take_last_position_logits (not yet solved)
# TODO: implement

# Step 160 - apply_temperature (not yet solved)
# TODO: implement

# Step 161 - top_k_filter (not yet solved)
# TODO: implement

# Step 162 - softmax_to_probs (not yet solved)
# TODO: implement

# Step 163 - sample_one_token (not yet solved)
# TODO: implement

# Step 164 - append_token_to_sequence (not yet solved)
# TODO: implement

# Step 165 - generation_loop_for_n_steps (not yet solved)
# TODO: implement

# Step 166 - decode_final_sequence (not yet solved)
# TODO: implement

