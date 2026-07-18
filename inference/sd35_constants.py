"""Fixed dimensions of the SD3.5 T5 + CLIP text embedding layout.

The full text-embedding vector produced by the SD3.5 pipeline is the T5
sequence flattened (T5_SEQ_LEN x T5_HIDDEN_DIM) concatenated with the pooled
CLIP vector (CLIP_POOLED_DIM). These are architectural constants of the
model; see get_embeddings_large_turbo_many_h5.py for the extraction path.
"""

T5_SEQ_LEN = 333
T5_HIDDEN_DIM = 4096
CLIP_POOLED_DIM = 2048
