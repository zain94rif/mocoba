"""Mojo AI - pustaka jaringan neural kecil yang ditulis murni dengan Mojo.

Modul yang tersedia:
    `nn`         - MLP (forward, backpropagation, mini-batch SGD)
    `datasets`   - dataset sintesis (XOR, spiral, blobs)
    `text`       - tokenisasi, stopword, vektorisasi TF-IDF
    `intents`    - data intent untuk asisten chat
    `calculator` - parser ekspresi aritmetika
    `chatbot`    - asisten chat berbasis klasifikasi intent
    `line_reader`- pembaca baris dari stdin
    `rng`        - PRNG xorshift64 (deterministik)
"""

from mojo_ai.nn import (
    MLP,
    ACT_RELU,
    ACT_TANH,
    ACT_SIGMOID,
    ACT_LINEAR,
    ACT_SOFTMAX,
)
from mojo_ai.datasets import Dataset, xor_dataset, spiral_dataset, blobs_dataset
from mojo_ai.text import (
    tokenize,
    tokenize_filtered,
    stem,
    is_stopword,
    Vectorizer,
)
from mojo_ai.intents import Intent, default_intents
from mojo_ai.calculator import evaluate, format_number
from mojo_ai.chatbot import ChatBot, Reply
from mojo_ai.line_reader import LineReader
from mojo_ai.rng import Rng
