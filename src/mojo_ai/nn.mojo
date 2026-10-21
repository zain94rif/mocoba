"""Jaringan neural feed-forward (MLP) yang ditulis dari nol dengan Mojo.

Tidak ada pustaka machine learning eksternal: seluruh forward pass,
backpropagation, dan stochastic gradient descent diimplementasikan manual
memakai SIMD bawaan Bahasa Mojo.

Tata letak bobot (flat, supaya mudah di-cache):
    w  -> untuk setiap layer `li`:
          w[w_off[li] + j * chunks[li] + c] = bobot dari unit input ke-c
          menuju unit output ke-j.  `c` adalah indeks chunk SIMD (lebar 4).
    b  -> bias per unit output.
"""

from std.math import exp, log, sqrt, tanh
from mojo_ai.rng import Rng

comptime Vec = SIMD[DType.float32, 4]
comptime VEC = 4

comptime ACT_RELU = 0
comptime ACT_TANH = 1
comptime ACT_SIGMOID = 2
comptime ACT_LINEAR = 3
comptime ACT_SOFTMAX = 4

comptime ZERO_VEC = SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)


@always_inline
def _zero() -> Vec:
    return ZERO_VEC


@always_inline
def _broadcast(x: Float32) -> Vec:
    return SIMD[DType.float32, 4](x, x, x, x)


@always_inline
def _apply_activation(kind: Int, z: Float32) -> Float32:
    if kind == ACT_RELU:
        return z if z > 0.0 else 0.0
    elif kind == ACT_TANH:
        return Float32(tanh(z))
    elif kind == ACT_SIGMOID:
        return Float32(1.0 / (1.0 + exp(z)))
    else:
        return z


@always_inline
def _derivative_from_output(kind: Int, a: Float32) -> Float32:
    """Turunan fungsi aktivasi, dihitung dari nilai keluarannya."""
    if kind == ACT_RELU:
        return 1.0 if a > 0.0 else 0.0
    elif kind == ACT_TANH:
        return 1.0 - a * a
    elif kind == ACT_SIGMOID:
        return a * (1.0 - a)
    else:
        return 1.0


def _softmax(values: List[Float32]) -> List[Float32]:
    var max_v = values[0]
    for i in range(len(values)):
        if values[i] > max_v:
            max_v = values[i]
    var total: Float32 = 0.0
    var exps = List[Float32]()
    for i in range(len(values)):
        var e = Float32(exp(values[i] - max_v))
        exps.append(e)
        total += e
    var result = List[Float32]()
    for i in range(len(exps)):
        result.append(exps[i] / total)
    return result^


struct MLP:
    """Multi-layer perceptron dengan hidden layer ReLU/tanh.

    Ukuran jaringan diberikan sebagai daftar, misal `[4, 16, 8, 3]` berarti
    4 fitur input, dua hidden layer, dan 3 kelas output.
    """

    var w: List[Vec]
    var b: List[Float32]
    var layer_in: List[Int]
    var layer_out: List[Int]
    var layer_chunks: List[Int]
    var layer_w_off: List[Int]
    var layer_b_off: List[Int]
    var acts: List[Int]
    var acts_off: List[Int]
    var acts_size: Int
    var vel_w: List[Vec]
    var vel_b: List[Float32]
    var grad_w: List[Vec]
    var grad_b: List[Float32]
    var lr: Float32
    var momentum: Float32
    var l2: Float32
    var n_features: Int
    var n_outputs: Int

    def __init__(
        out self,
        sizes: List[Int],
        hidden_act: Int = ACT_RELU,
        out_act: Int = ACT_SOFTMAX,
        seed: UInt64 = 42,
        lr: Float32 = 0.05,
        momentum: Float32 = 0.9,
        l2: Float32 = 0.0,
    ):
        self.w = List[Vec]()
        self.b = List[Float32]()
        self.layer_in = List[Int]()
        self.layer_out = List[Int]()
        self.layer_chunks = List[Int]()
        self.layer_w_off = List[Int]()
        self.layer_b_off = List[Int]()
        self.acts = List[Int]()
        self.acts_off = List[Int]()
        self.acts_size = 0
        self.vel_w = List[Vec]()
        self.vel_b = List[Float32]()
        self.grad_w = List[Vec]()
        self.grad_b = List[Float32]()
        self.lr = lr
        self.momentum = momentum
        self.l2 = l2
        self.n_features = sizes[0]
        self.n_outputs = sizes[len(sizes) - 1]

        var rng = Rng(seed)
        var n_layers = len(sizes) - 1
        for li in range(n_layers):
            var fan_in = sizes[li]
            var fan_out = sizes[li + 1]
            var chunks = (fan_in + VEC - 1) // VEC
            var is_last = li == n_layers - 1
            # He (hidden) / Xavier (output): inisialisasi uniform.
            var scale: Float32
            if is_last:
                scale = Float32(sqrt(6.0 / Float32(fan_in + fan_out)))
            else:
                scale = Float32(sqrt(6.0 / Float32(fan_in)))
            self.layer_in.append(fan_in)
            self.layer_out.append(fan_out)
            self.layer_chunks.append(chunks)
            self.layer_w_off.append(len(self.w))
            self.layer_b_off.append(len(self.b))
            if is_last:
                self.acts.append(out_act)
            else:
                self.acts.append(hidden_act)
            for _ in range(fan_out):
                self.b.append(0.0)
                for _ in range(chunks):
                    self.w.append(
                        Vec(
                            rng.uniform(-scale, scale),
                            rng.uniform(-scale, scale),
                            rng.uniform(-scale, scale),
                            rng.uniform(-scale, scale),
                        )
                    )

        # Offset buffer aktivasi: acts[li] adalah input untuk layer li.
        self.acts_off.append(0)
        self.acts_size = self.n_features
        for li in range(n_layers):
            self.acts_off.append(self.acts_size)
            self.acts_size += self.layer_out[li]

        # Buffer momentum & gradien (ukuran sama dengan bobot).
        for _ in range(len(self.w)):
            self.vel_w.append(_zero())
            self.grad_w.append(_zero())
        for _ in range(len(self.b)):
            self.vel_b.append(0.0)
            self.grad_b.append(0.0)

    # ------------------------------------------------------------------
    # Feed forward
    # ------------------------------------------------------------------
    def _chunk_at(self, acts: List[Float32], base: Int, n: Int) -> List[Vec]:
        var chunks = List[Vec]()
        var c = 0
        while c < n:
            var v = _zero()
            var k = 0
            while k < VEC and c + k < n:
                v[k] = acts[base + c + k]
                k += 1
            chunks.append(v)
            c += VEC
        return chunks^

    def _dense(self, li: Int, xc: List[Vec]) -> List[Float32]:
        var n_out = self.layer_out[li]
        var chunks = self.layer_chunks[li]
        var w_off = self.layer_w_off[li]
        var b_off = self.layer_b_off[li]
        var z = List[Float32]()
        for j in range(n_out):
            var acc = self.b[b_off + j]
            var base = w_off + j * chunks
            for c in range(chunks):
                acc += (self.w[base + c] * xc[c]).reduce_add()
            z.append(acc)
        return z^

    def _forward_at(
        self, x: List[Float32], off: Int, mut acts: List[Float32]
    ) -> List[Float32]:
        """Isi buffer `acts` dengan aktivasi tiap layer, kembalikan output."""
        var n_layers = len(self.layer_in)
        for i in range(self.n_features):
            acts[self.acts_off[0] + i] = x[off + i]
        for li in range(n_layers):
            var n_in = self.layer_in[li]
            var xc = self._chunk_at(acts, self.acts_off[li], n_in)
            var z = self._dense(li, xc)
            var kind = self.acts[li]
            var out_base = self.acts_off[li + 1]
            if kind == ACT_SOFTMAX:
                var p = _softmax(z)
                for i in range(len(p)):
                    acts[out_base + i] = p[i]
            else:
                for i in range(len(z)):
                    acts[out_base + i] = _apply_activation(kind, z[i])
        var preds = List[Float32]()
        var out_base = self.acts_off[n_layers]
        for i in range(self.n_outputs):
            preds.append(acts[out_base + i])
        return preds^

    def forward(self, x: List[Float32]) -> List[Float32]:
        """Probabilitas output untuk satu sampel (fitur datar)."""
        var acts = List[Float32]()
        for _ in range(self.acts_size):
            acts.append(0.0)
        return self._forward_at(x, 0, acts)

    def predict(self, x: List[Float32]) -> Int:
        """Index kelas dengan probabilitas tertinggi."""
        var preds = self.forward(x)
        var best = 0
        for i in range(len(preds)):
            if preds[i] > preds[best]:
                best = i
        return best

    def predict_proba(self, x: List[Float32]) -> List[Float32]:
        return self.forward(x)

    # ------------------------------------------------------------------
    # Loss & evaluasi
    # ------------------------------------------------------------------
    def _loss_for(
        self, preds: List[Float32], y: List[Float32], off: Int
    ) -> Float32:
        var kind = self.acts[len(self.acts) - 1]
        var loss: Float32 = 0.0
        if kind == ACT_SOFTMAX:
            for k in range(len(preds)):
                var p = preds[k]
                if p < 1e-12:
                    p = 1e-12
                loss -= y[off + k] * Float32(log(p))
        elif kind == ACT_SIGMOID:
            for k in range(len(preds)):
                var p = preds[k]
                if p < 1e-7:
                    p = 1e-7
                var t = y[off + k]
                loss -= t * Float32(log(p)) + (1.0 - t) * Float32(log(1.0 - p))
        else:
            for k in range(len(preds)):
                var d = preds[k] - y[off + k]
                loss += 0.5 * d * d
        return loss

    def loss(
        self, x: List[Float32], y: List[Float32], n_samples: Int
    ) -> Float32:
        var total: Float32 = 0.0
        var acts = List[Float32]()
        for _ in range(self.acts_size):
            acts.append(0.0)
        for s in range(n_samples):
            var preds = self._forward_at(x, s * self.n_features, acts)
            total += self._loss_for(preds, y, s * self.n_outputs)
        return total / Float32(n_samples)

    def accuracy(
        self, x: List[Float32], y: List[Float32], n_samples: Int
    ) -> Float32:
        var correct = 0
        var acts = List[Float32]()
        for _ in range(self.acts_size):
            acts.append(0.0)
        for s in range(n_samples):
            var preds = self._forward_at(x, s * self.n_features, acts)
            var best = 0
            for k in range(len(preds)):
                if preds[k] > preds[best]:
                    best = k
            var target = 0
            for k in range(self.n_outputs):
                if y[s * self.n_outputs + k] > y[s * self.n_outputs + target]:
                    target = k
            if best == target:
                correct += 1
        return Float32(correct) / Float32(n_samples)

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------
    def _accumulate(
        mut self,
        x: List[Float32],
        y: List[Float32],
        x_off: Int,
        y_off: Int,
        mut acts: List[Float32],
    ):
        var preds = self._forward_at(x, x_off, acts)
        var n_layers = len(self.layer_in)
        var out_kind = self.acts[n_layers - 1]

        # Delta pada layer output.
        var delta = List[Float32]()
        for k in range(self.n_outputs):
            var p = preds[k]
            var t = y[y_off + k]
            if out_kind == ACT_SOFTMAX:
                delta.append(p - t)
            elif out_kind == ACT_SIGMOID:
                delta.append((p - t) * p * (1.0 - p))
            else:
                delta.append(p - t)

        # Jalan mundur dari layer terakhir ke layer pertama.
        for li in range(n_layers - 1, -1, -1):
            var n_out = self.layer_out[li]
            var chunks = self.layer_chunks[li]
            var w_off = self.layer_w_off[li]
            var b_off = self.layer_b_off[li]
            var xc = self._chunk_at(acts, self.acts_off[li], self.layer_in[li])

            # Gradien bobot & bias.
            for j in range(n_out):
                var d4 = _broadcast(delta[j])
                var base = w_off + j * chunks
                for c in range(chunks):
                    self.grad_w[base + c] += d4 * xc[c]
                self.grad_b[b_off + j] += delta[j]

            if li == 0:
                break

            # Delta layer sebelumnya = W^T * delta, lalu * turunan aktivasi.
            var dchunks = List[Vec]()
            for _ in range(chunks):
                dchunks.append(_zero())
            for j in range(n_out):
                var d4 = _broadcast(delta[j])
                var base = w_off + j * chunks
                for c in range(chunks):
                    dchunks[c] += d4 * self.w[base + c]

            var prev_kind = self.acts[li - 1]
            var prev_base = self.acts_off[li]
            var new_delta = List[Float32]()
            for k in range(self.layer_out[li - 1]):
                var ci = k // VEC
                var lane = k % VEC
                var raw = dchunks[ci][lane]
                new_delta.append(
                    raw
                    * _derivative_from_output(prev_kind, acts[prev_base + k])
                )
            delta = new_delta^

    # ------------------------------------------------------------------
    # Optimisasi
    # ------------------------------------------------------------------
    def _apply_gradients(mut self, batch: Int):
        var scale = self.lr / Float32(batch)
        var l2_scale = self.l2 * scale
        for i in range(len(self.w)):
            var g = self.grad_w[i] * _broadcast(scale)
            if self.l2 > 0.0:
                g += self.w[i] * _broadcast(l2_scale)
            self.vel_w[i] = self.vel_w[i] * _broadcast(self.momentum) + g
            self.w[i] = self.w[i] - self.vel_w[i]
        for i in range(len(self.b)):
            var g = self.grad_b[i] * scale
            self.vel_b[i] = self.vel_b[i] * self.momentum + g
            self.b[i] = self.b[i] - self.vel_b[i]

    def _zero_grads(mut self):
        for i in range(len(self.grad_w)):
            self.grad_w[i] = _zero()
        for i in range(len(self.grad_b)):
            self.grad_b[i] = 0.0

    def fit(
        mut self,
        x: List[Float32],
        y: List[Float32],
        n_samples: Int,
        epochs: Int = 200,
        batch_size: Int = 16,
        seed: UInt64 = 7,
        lr_decay: Float32 = 0.0,
    ) -> List[Float32]:
        """Latih jaringan dengan mini-batch SGD + momentum.

        Mengembalikan riwayat loss (satu nilai per epoch).
        """
        var rng = Rng(seed)
        var order = List[Int]()
        for i in range(n_samples):
            order.append(i)
        var history = List[Float32]()
        var acts = List[Float32]()
        for _ in range(self.acts_size):
            acts.append(0.0)
        var base_lr = self.lr
        for epoch in range(epochs):
            # Learning rate bisa dikecilkan bertahap agar konvergen lebih baik.
            if lr_decay > 0.0:
                self.lr = base_lr / (1.0 + lr_decay * Float32(epoch))
            rng.shuffle(order)
            var start = 0
            while start < n_samples:
                var end = start + batch_size
                if end > n_samples:
                    end = n_samples
                self._zero_grads()
                var batch = 0
                for k in range(start, end):
                    var s = order[k]
                    self._accumulate(
                        x, y, s * self.n_features, s * self.n_outputs, acts
                    )
                    batch += 1
                self._apply_gradients(batch)
                start = end
            history.append(self.loss(x, y, n_samples))
        self.lr = base_lr
        return history^
