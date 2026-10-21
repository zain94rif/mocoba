"""Demo utama: melatih jaringan neural dari nol pada beberapa dataset.

Jalankan dengan:
    pixi run demo
"""

from std.math import exp
from std.time import perf_counter
from mojo_ai.nn import MLP, ACT_RELU, ACT_TANH, ACT_SOFTMAX
from mojo_ai.datasets import xor_dataset, spiral_dataset, blobs_dataset


def _sizes(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    return s^


def _pad(text: String, width: Int) -> String:
    var out = text
    while out.byte_length() < width:
        out = " " + out
    return out


def _bar(value: Float32, width: Int = 24) -> String:
    var filled = Int(value * Float32(width))
    if filled < 0:
        filled = 0
    if filled > width:
        filled = width
    var out = "["
    for _ in range(filled):
        out += "#"
    for _ in range(width - filled):
        out += "."
    out += "]"
    return out


def demo_xor():
    print("=" * 62)
    print("1) XOR - dua input, satu hidden layer 8 unit")
    print("=" * 62)
    var ds = xor_dataset()
    var net = MLP(_sizes(2, 8, 2), ACT_RELU, ACT_SOFTMAX, seed=1, lr=0.5)
    var start = perf_counter()
    var history = net.fit(ds.x, ds.y, ds.n, epochs=400, batch_size=4, seed=5)
    var elapsed = perf_counter() - start
    print("loss  awal : ", history[0])
    print("loss  akhir : ", history[len(history) - 1])
    print("akurasi     : ", net.accuracy(ds.x, ds.y, ds.n) * 100.0, "%")
    print("waktu       : ", elapsed, " detik")
    print("-" * 62)
    for i in range(ds.n):
        var feats = List[Float32]()
        feats.append(ds.x[i * ds.n_features])
        feats.append(ds.x[i * ds.n_features + 1])
        var probs = net.predict_proba(feats)
        print(
            "  (",
            feats[0],
            ", ",
            feats[1],
            ") -> kelas ",
            String(net.predict(feats)),
            "  (p=",
            probs[0],
            " / ",
            probs[1],
            ")",
        )


def demo_spiral():
    print()
    print("=" * 62)
    print("2) SPIRAL - boundary decision non-linier, 2 hidden layer")
    print("=" * 62)
    var ds = spiral_dataset(80, 0.15, 7)
    var s = List[Int]()
    s.append(2)
    s.append(24)
    s.append(12)
    s.append(2)
    var net = MLP(s, ACT_TANH, ACT_SOFTMAX, seed=3, lr=0.08)
    var start = perf_counter()
    var history = net.fit(
        ds.x, ds.y, ds.n, epochs=150, batch_size=32, seed=11, lr_decay=0.01
    )
    var elapsed = perf_counter() - start
    print("samples     : ", ds.n)
    print("loss  awal : ", history[0])
    print("loss  akhir : ", history[len(history) - 1])
    print("akurasi     : ", net.accuracy(ds.x, ds.y, ds.n) * 100.0, "%")
    print("waktu       : ", elapsed, " detik")
    # Kurva loss beberapa epoch (bar dihitung dari skala eksponensial)
    print("-" * 62)
    var step = 10
    for i in range(0, len(history), step):
        var quality = Float32(exp(-Float64(history[i]) * 4.0))
        print(
            "  epoch ",
            _pad(String(i), 4),
            "  loss ",
            _pad(String(history[i]), 14),
            _bar(quality),
        )


def demo_blobs():
    print()
    print("=" * 62)
    print("3) BLOBS - 3 kelas gaussian, softmax + cross entropy")
    print("=" * 62)
    var ds = blobs_dataset(50, 3, 21)
    var s = List[Int]()
    s.append(2)
    s.append(16)
    s.append(3)
    var net = MLP(s, ACT_RELU, ACT_SOFTMAX, seed=9, lr=0.08)
    var start = perf_counter()
    var history = net.fit(
        ds.x, ds.y, ds.n, epochs=120, batch_size=30, seed=13, lr_decay=0.01
    )
    var elapsed = perf_counter() - start
    print("samples     : ", ds.n)
    print("loss  awal : ", history[0])
    print("loss  akhir : ", history[len(history) - 1])
    print("akurasi     : ", net.accuracy(ds.x, ds.y, ds.n) * 100.0, "%")
    print("waktu       : ", elapsed, " detik")
    print("-" * 62)
    var points = List[List[Float32]]()
    var p0 = List[Float32]()
    p0.append(2.5)
    p0.append(0.0)
    points.append(p0^)
    var p1 = List[Float32]()
    p1.append(-1.25)
    p1.append(2.16)
    points.append(p1^)
    var p2 = List[Float32]()
    p2.append(-1.25)
    p2.append(-2.16)
    points.append(p2^)
    for i in range(len(points)):
        var probs = net.predict_proba(points[i])
        print(
            "  titik (",
            points[i][0],
            ", ",
            points[i][1],
            ") -> kelas ",
            String(net.predict(points[i])),
            "  probs: ",
            probs,
        )


def main() raises:
    print("MOC - Mini AI, 100% Mojo tanpa Python")
    print(
        "Jaringan neural feed-forward ditulis dari nol (forward + backprop +"
        " SGD)"
    )
    demo_xor()
    demo_spiral()
    demo_blobs()
    print()
    print("Selesai. Lihat `pixi run chatbot` untuk asisten chat interaktif.")
