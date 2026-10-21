"""Deterministic pseudo random number generator (xorshift64*).

Mojo 1.1 tidak menyediakan modul `random` yang stabil, jadi kita
implementasikan PRNG kecil sendiri supaya seluruh pipeline bisa direproduksi.
"""

from std.math import cos, log, sqrt


struct Rng:
    var state: UInt64

    def __init__(out self, seed: UInt64 = 88172645463325252):
        self.state = seed if seed != 0 else 88172645463325252

    def next_u64(mut self) -> UInt64:
        var x = self.state
        x ^= x << 13
        x ^= x >> 7
        x ^= x << 17
        self.state = x
        return x

    def next_float(mut self) -> Float32:
        # 24 bit mantissa -> [0, 1)
        return Float32(self.next_u64() >> 40) / Float32(1 << 24)

    def uniform(mut self, low: Float32, high: Float32) -> Float32:
        return low + (high - low) * self.next_float()

    def normal(mut self, mean: Float32 = 0.0, stddev: Float32 = 1.0) -> Float32:
        # Box-Muller transform.
        var u1 = self.next_float()
        if u1 < 1e-7:
            u1 = 1e-7
        var u2 = self.next_float()
        var r = sqrt(-2.0 * log(u1))
        return mean + stddev * r * Float32(cos(2.0 * 3.14159265358979 * u2))

    def randint(mut self, low: Int, high: Int) -> Int:
        if high <= low:
            return low
        return low + Int(self.next_u64() % UInt64(high - low + 1))

    def shuffle(mut self, mut indices: List[Int]):
        var i = len(indices) - 1
        while i > 0:
            var j = self.randint(0, i)
            var tmp = indices[i]
            indices[i] = indices[j]
            indices[j] = tmp
            i -= 1
