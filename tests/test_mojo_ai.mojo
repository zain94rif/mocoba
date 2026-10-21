"""Unit test untuk paket `mojo_ai`.

Jalankan dengan:
    pixi run test
"""

from std.io.file import open
from std.os import remove
from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_true,
    TestSuite,
)
from mojo_ai.rng import Rng
from mojo_ai.text import (
    is_stopword,
    stem,
    tokenize,
    tokenize_filtered,
    Vectorizer,
)
from mojo_ai.calculator import evaluate, format_number
from mojo_ai.datasets import blobs_dataset, spiral_dataset, xor_dataset
from mojo_ai.nn import MLP, ACT_RELU, ACT_SOFTMAX
from mojo_ai.chatbot import ChatBot
from mojo_ai.line_reader import LineReader

comptime TMP_FILE = "/tmp/mojo_ai_line_reader_test.txt"


# ----------------------------------------------------------------------
# RNG
# ----------------------------------------------------------------------
def test_rng_is_deterministic() raises:
    var a = Rng(1234)
    var b = Rng(1234)
    for _ in range(16):
        assert_equal(a.next_u64(), b.next_u64())


def test_rng_uniform_stays_in_range() raises:
    var rng = Rng(7)
    for _ in range(200):
        var v = rng.uniform(-2.0, 5.0)
        assert_true(v >= -2.0 and v < 5.0)


def test_rng_randint_covers_bounds() raises:
    var rng = Rng(99)
    var seen_low = False
    var seen_high = False
    for _ in range(500):
        var v = rng.randint(0, 3)
        assert_true(v >= 0 and v <= 3)
        if v == 0:
            seen_low = True
        if v == 3:
            seen_high = True
    assert_true(seen_low)
    assert_true(seen_high)


def test_rng_shuffle_keeps_all_indices() raises:
    var rng = Rng(5)
    var idx = List[Int]()
    for i in range(10):
        idx.append(i)
    rng.shuffle(idx)
    var total = 0
    for v in idx:
        total += v
    assert_equal(total, 45)


# ----------------------------------------------------------------------
# Teks
# ----------------------------------------------------------------------
def test_tokenize_splits_and_lowercases() raises:
    var tokens = tokenize("Halo, Dunia! Kamu APA?")
    assert_equal(len(tokens), 4)
    assert_equal(tokens[0], "halo")
    assert_equal(tokens[1], "dunia")
    assert_equal(tokens[3], "apa")


def test_tokenize_drops_empty_tokens() raises:
    var tokens = tokenize("  ,,, halo ...  ")
    assert_equal(len(tokens), 1)
    assert_equal(tokens[0], "halo")


def test_stem_removes_common_suffixes() raises:
    assert_equal(stem("bukunya"), "buku")
    assert_equal(stem("mematikan"), "memati")
    assert_equal(stem("makan"), "makan")


def test_stopword_detection() raises:
    assert_true(is_stopword("yang"))
    assert_true(is_stopword("adalah"))
    assert_false(is_stopword("buku"))
    assert_false(is_stopword("mesin"))


def test_tokenize_filtered_removes_stopwords() raises:
    var tokens = tokenize_filtered("apa itu machine learning")
    assert_true("yang" not in tokens)
    assert_true("machine" in tokens)
    assert_true("learning" in tokens)


def test_vectorizer_produces_normalized_features() raises:
    var texts = List[String]()
    texts.append("kucing duduk di atas tikar")
    texts.append("anjing berlari di taman")
    texts.append("kucing tidur di sofa")
    var vec = Vectorizer()
    vec.fit(texts)
    assert_true(vec.n_features > 0)
    var features = vec.transform("kucing di tikar")
    assert_equal(len(features), vec.n_features)
    var norm: Float32 = 0.0
    for f in features:
        norm += f * f
    assert_almost_equal(Float64(norm), 1.0, atol=1e-5)


def test_vectorizer_vocabulary_is_deterministic() raises:
    var texts = List[String]()
    texts.append("satu dua")
    texts.append("dua tiga")
    var a = Vectorizer()
    a.fit(texts)
    var b = Vectorizer()
    b.fit(texts)
    assert_equal(a.n_features, b.n_features)
    var fa = a.transform("satu tiga")
    var fb = b.transform("satu tiga")
    for i in range(len(fa)):
        assert_equal(fa[i], fb[i])


# ----------------------------------------------------------------------
# Kalkulator
# ----------------------------------------------------------------------
def test_calculator_basic_operations() raises:
    assert_true(evaluate("12 * 7").ok)
    assert_almost_equal(Float64(evaluate("12 * 7").value), 84.0, atol=1e-4)
    assert_almost_equal(Float64(evaluate("10 + 5").value), 15.0, atol=1e-4)
    assert_almost_equal(Float64(evaluate("10 - 5").value), 5.0, atol=1e-4)
    assert_almost_equal(Float64(evaluate("9 / 3").value), 3.0, atol=1e-4)


def test_calculator_respects_precedence() raises:
    assert_almost_equal(Float64(evaluate("2 + 3 * 4").value), 14.0, atol=1e-4)
    assert_almost_equal(Float64(evaluate("(2 + 3) * 4").value), 20.0, atol=1e-4)
    assert_almost_equal(Float64(evaluate("-3 + 5").value), 2.0, atol=1e-4)


def test_calculator_rejects_garbage() raises:
    assert_false(evaluate("halo dunia").ok)
    assert_false(evaluate("10 / 0").ok)
    assert_false(evaluate("apa kabar").ok)


def test_format_number() raises:
    assert_equal(format_number(84.0), "84")
    assert_equal(format_number(0.0), "0")
    assert_equal(format_number(1.5), "1.5000")


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
def test_xor_dataset_shape() raises:
    var ds = xor_dataset()
    assert_equal(ds.n, 4)
    assert_equal(ds.n_features, 2)
    assert_equal(ds.n_classes, 2)
    assert_equal(ds.label_of(0), 0)
    assert_equal(ds.label_of(1), 1)
    assert_equal(ds.label_of(2), 1)
    assert_equal(ds.label_of(3), 0)


def test_spiral_dataset_shape() raises:
    var ds = spiral_dataset(n_per_class=25, noise=0.05, seed=3)
    assert_equal(ds.n, 50)
    assert_equal(ds.n_features, 2)
    assert_equal(ds.n_classes, 2)


def test_blobs_dataset_shape() raises:
    var ds = blobs_dataset(n_per_class=10, n_classes=3, seed=3)
    assert_equal(ds.n, 30)
    assert_equal(ds.n_classes, 3)


def test_dataset_slice_and_standardize() raises:
    var ds = spiral_dataset(n_per_class=20, noise=0.1, seed=8)
    var part = ds.slice(0, 10)
    assert_equal(part.n, 10)
    assert_equal(part.n_features, ds.n_features)
    ds.standardize()
    var mean: Float32 = 0.0
    for i in range(ds.n):
        mean += ds.x[i * ds.n_features]
    mean /= Float32(ds.n)
    assert_almost_equal(Float64(mean), 0.0, atol=1e-3)


# ----------------------------------------------------------------------
# Jaringan neural
# ----------------------------------------------------------------------
def test_mlp_output_is_a_distribution() raises:
    var sizes = List[Int]()
    sizes.append(2)
    sizes.append(8)
    sizes.append(3)
    var net = MLP(sizes, ACT_RELU, ACT_SOFTMAX, seed=1)
    var x = List[Float32]()
    x.append(0.5)
    x.append(-0.2)
    var probs = net.predict_proba(x)
    assert_equal(len(probs), 3)
    var total: Float32 = 0.0
    for p in probs:
        assert_true(p >= 0.0)
        total += p
    assert_almost_equal(Float64(total), 1.0, atol=1e-5)


def test_mlp_learns_xor() raises:
    var ds = xor_dataset()
    var sizes = List[Int]()
    sizes.append(2)
    sizes.append(8)
    sizes.append(2)
    var net = MLP(sizes, ACT_RELU, ACT_SOFTMAX, seed=5, lr=0.2, momentum=0.9)
    var history = net.fit(
        ds.x, ds.y, ds.n, epochs=400, batch_size=4, seed=11, lr_decay=0.01
    )
    assert_equal(len(history), 400)
    assert_true(net.accuracy(ds.x, ds.y, ds.n) > 0.99)


def test_mlp_loss_decreases() raises:
    var ds = blobs_dataset(n_per_class=15, n_classes=3, seed=17)
    var sizes = List[Int]()
    sizes.append(ds.n_features)
    sizes.append(12)
    sizes.append(ds.n_classes)
    var net = MLP(sizes, ACT_RELU, ACT_SOFTMAX, seed=2, lr=0.1)
    var history = net.fit(
        ds.x, ds.y, ds.n, epochs=25, batch_size=16, seed=3, lr_decay=0.02
    )
    assert_true(history[len(history) - 1] < history[0])


# ----------------------------------------------------------------------
# Chatbot
# ----------------------------------------------------------------------
def test_chatbot_recognizes_intents() raises:
    var bot = ChatBot(seed=11, epochs=120)
    var acc = bot.train()
    assert_true(acc > 0.95)

    assert_equal(bot.classify("halo").intent, "sapaan")
    assert_equal(bot.classify("siapa kamu?").intent, "nama")
    assert_equal(bot.classify("apa itu machine learning").intent, "tentang_ai")
    assert_equal(bot.classify("terima kasih banyak").intent, "terima_kasih")


def test_chatbot_answers_arithmetic() raises:
    var bot = ChatBot(seed=11, epochs=120)
    _ = bot.train()
    var reply = bot.classify("berapa 12 * 7")
    assert_equal(reply.intent, "kalkulator")
    assert_true(reply.text.find("84") != -1)
    # Bahasa yang tidak dikenal harus jatuh ke jawaban cadangan.
    assert_equal(bot.classify("zzz qqq xyzzy").intent, "tidak_diketahui")


# ----------------------------------------------------------------------
# Pembaca baris
# ----------------------------------------------------------------------
def test_line_reader_splits_lines() raises:
    var f = open(TMP_FILE, "w")
    f.write_string("satu\ndua\ntiga\n")
    f.close()

    var g = open(TMP_FILE, "r")
    var reader = LineReader(g.handle)
    var (first, ok1) = reader.read_line()
    assert_true(ok1)
    assert_equal(first, "satu")
    var (second, ok2) = reader.read_line()
    assert_true(ok2)
    assert_equal(second, "dua")
    var (third, ok3) = reader.read_line()
    assert_true(ok3)
    assert_equal(third, "tiga")
    var (_, ok4) = reader.read_line()
    assert_false(ok4)
    g.close()
    _ = remove(TMP_FILE)


def test_line_reader_handles_missing_trailing_newline() raises:
    var f = open(TMP_FILE, "w")
    f.write_string("alpha\nbeta")
    f.close()

    var g = open(TMP_FILE, "r")
    var reader = LineReader(g.handle)
    var lines = reader.read_all()
    assert_equal(len(lines), 2)
    assert_equal(lines[0], "alpha")
    assert_equal(lines[1], "beta")
    g.close()
    _ = remove(TMP_FILE)


def test_line_reader_strips_carriage_return() raises:
    var f = open(TMP_FILE, "w")
    f.write_string("windows\r\nline\r\n")
    f.close()

    var g = open(TMP_FILE, "r")
    var reader = LineReader(g.handle)
    var lines = reader.read_all()
    assert_equal(len(lines), 2)
    assert_equal(lines[0], "windows")
    assert_equal(lines[1], "line")
    g.close()
    _ = remove(TMP_FILE)


def test_line_reader_replaces_invalid_utf8() raises:
    # Byte 0xFF/0xFE tidak membentuk UTF-8 yang sah dan harus diganti.
    var f = open(TMP_FILE, "w")
    var raw = List[UInt8]()
    raw.append(UInt8(97))
    raw.append(UInt8(255))
    raw.append(UInt8(254))
    raw.append(UInt8(98))
    raw.append(UInt8(10))
    f.write_bytes(Span[UInt8](raw))
    f.close()

    var g = open(TMP_FILE, "r")
    var reader = LineReader(g.handle)
    var lines = reader.read_all()
    assert_equal(len(lines), 1)
    assert_equal(lines[0], "a??b")
    g.close()
    _ = remove(TMP_FILE)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
