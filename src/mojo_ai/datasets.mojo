"""Kumpulan dataset sintesis untuk melatih dan menguji jaringan."""

from std.math import cos, sin, sqrt
from mojo_ai.rng import Rng

comptime PI = 3.14159265358979323846


struct Dataset(Copyable):
    """Dataset padat (flat): x berisi `n * n_features`, y berisi one-hot."""

    var x: List[Float32]
    var y: List[Float32]
    var n: Int
    var n_features: Int
    var n_classes: Int
    var labels: List[String]

    def __init__(out self, n: Int = 0, n_features: Int = 0, n_classes: Int = 0):
        self.x = List[Float32]()
        self.y = List[Float32]()
        self.n = n
        self.n_features = n_features
        self.n_classes = n_classes
        self.labels = List[String]()

    def push(mut self, features: List[Float32], label: Int):
        for i in range(len(features)):
            self.x.append(features[i])
        for k in range(self.n_classes):
            if k == label:
                self.y.append(1.0)
            else:
                self.y.append(0.0)
        self.n += 1

    def label_of(self, index: Int) -> Int:
        for k in range(self.n_classes):
            if self.y[index * self.n_classes + k] > 0.5:
                return k
        return 0

    def slice(self, start: Int, count: Int) -> Dataset:
        var result = Dataset(
            n=0, n_features=self.n_features, n_classes=self.n_classes
        )
        for k in range(self.n_classes):
            result.labels.append(self.labels[k])
        for i in range(start, start + count):
            var feats = List[Float32]()
            for f in range(self.n_features):
                feats.append(self.x[i * self.n_features + f])
            result.push(feats, self.label_of(i))
        return result^

    def standardize(mut self):
        """Normalisasi mean=0 std=1 per fitur."""
        for f in range(self.n_features):
            var mean: Float32 = 0.0
            for i in range(self.n):
                mean += self.x[i * self.n_features + f]
            mean /= Float32(self.n)
            var var_sum: Float32 = 0.0
            for i in range(self.n):
                var d = self.x[i * self.n_features + f] - mean
                var_sum += d * d
            var std = Float32(sqrt(var_sum / Float32(self.n)))
            if std < 1e-6:
                std = 1.0
            for i in range(self.n):
                self.x[i * self.n_features + f] = (
                    self.x[i * self.n_features + f] - mean
                ) / std


def xor_dataset() -> Dataset:
    """Klasifikasi XOR - masalah klasik yang butuh hidden layer."""
    var ds = Dataset(n=0, n_features=2, n_classes=2)
    ds.labels.append("kelas 0")
    ds.labels.append("kelas 1")
    var points = List[List[Float32]]()
    var f0 = List[Float32]()
    f0.append(0.0)
    f0.append(0.0)
    points.append(f0^)
    var f1 = List[Float32]()
    f1.append(0.0)
    f1.append(1.0)
    points.append(f1^)
    var f2 = List[Float32]()
    f2.append(1.0)
    f2.append(0.0)
    points.append(f2^)
    var f3 = List[Float32]()
    f3.append(1.0)
    f3.append(1.0)
    points.append(f3^)
    var labels = List[Int]()
    labels.append(0)
    labels.append(1)
    labels.append(1)
    labels.append(0)
    for i in range(len(points)):
        ds.push(points[i].copy(), labels[i])
    return ds^


def spiral_dataset(
    n_per_class: Int = 100, noise: Float32 = 0.12, seed: UInt64 = 99
) -> Dataset:
    """Dua lengan spiral - butuh decision boundary yang non-linier."""
    var ds = Dataset(n=0, n_features=2, n_classes=2)
    ds.labels.append("lengan A")
    ds.labels.append("lengan B")
    var rng = Rng(seed)
    var total = n_per_class * 2
    for i in range(total):
        var t = 1.4 * PI * Float32(i) / Float32(total)
        var arm = i % 2
        var radius = t
        if arm == 1:
            radius = t + PI
        var feats = List[Float32]()
        feats.append(
            Float32(radius * cos(t) * 0.6) + noise * rng.normal(0.0, 1.0)
        )
        feats.append(
            Float32(radius * sin(t) * 0.6) + noise * rng.normal(0.0, 1.0)
        )
        ds.push(feats, arm)
    return ds^


def blobs_dataset(
    n_per_class: Int = 60, n_classes: Int = 3, seed: UInt64 = 1234
) -> Dataset:
    """Sekumpulan gumpalan gaussian yang bisa dipisahkan."""
    var ds = Dataset(n=0, n_features=2, n_classes=n_classes)
    var rng = Rng(seed)
    for c in range(n_classes):
        ds.labels.append("gumpalan " + String(c + 1))
        var angle = 2.0 * PI * Float32(c) / Float32(n_classes)
        var cx = 2.5 * cos(angle)
        var cy = 2.5 * sin(angle)
        for _ in range(n_per_class):
            var feats = List[Float32]()
            feats.append(cx + 0.55 * rng.normal(0.0, 1.0))
            feats.append(cy + 0.55 * rng.normal(0.0, 1.0))
            ds.push(feats, c)
    return ds^
