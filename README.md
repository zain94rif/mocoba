# Mocoba — AI kecil written in Mojo (no Python)

Asisten AI sederhana yang **100% ditulis dengan Bahasa Mojo**: jaringan neural
feed-forward (forward pass + backpropagation + SGD) dibuat dari nol, tanpa
pustaka machine learning eksternal. Runtime dikelola dengan
[Pixi](https://pixi.sh).

```bash
pixi run demo      # latih MLP pada dataset XOR, spiral, dan blobs
pixi run chatbot   # asisten chat interaktif (klasifikasi intent + kalkulator)
pixi run test      # unit test (28 test)
```

---

## Isi paket

| Modul | Isi |
| --- | --- |
| `mojo_ai/nn.mojo` | `MLP`: multi-layer perceptron, aktivasi ReLU/tanh/sigmoid/softmax, cross-entropy, mini-batch SGD + momentum + learning-rate decay. Perkalian matriks memakai SIMD `SIMD[DType.float32, 4]`. |
| `mojo_ai/datasets.mojo` | Dataset sintesis: `xor_dataset`, `spiral_dataset`, `blobs_dataset`, plus `standardize` dan `slice`. |
| `mojo_ai/text.mojo` | `tokenize`, `tokenize_filtered`, `stem`, `is_stopword`, dan `Vectorizer` (bag-of-words dengan bobot TF-IDF). |
| `mojo_ai/intents.mojo` | 10 intent siap pakai (sapaan, perpisahan, terima kasih, bantuan, nama, kalkulator, tentang AI, lelucon, cuaca, pujian). |
| `mojo_ai/calculator.mojo` | Parser aritmetika rekursif-descendant (`+ - * /` dan kurung) plus `format_number`. |
| `mojo_ai/chatbot.mojo` | `ChatBot`: vektorisasi TF-IDF → MLP softmax → intent + jawaban; shortcut kalkulator dan jawaban cadangan untuk input tak dikenal. |
| `mojo_ai/line_reader.mojo` | `LineReader`: pembaca baris dari stdin (dipakai CLI agar bisa dijalankan lewat pipe). |
| `mojo_ai/rng.mojo` | PRNG xorshift64 sendiri supaya seluruh pipeline deterministik dan bisa direproduksi. |

Aplikasi:

| File | Keterangan |
| --- | --- |
| `apps/train_mlp.mojo` | Demo pelatihan: XOR (2→8→2), spiral (2→24→12→2, tanh), blobs (2→16→3). |
| `apps/chatbot.mojo` | REPL asisten chat, bisa interaktif maupun_non-interaktif lewat pipe. |

---

## Persiapan

Butuh [Pixi](https://pixi.sh/latest/) saja. Mojo 1.1 sudah dijamin lewat
`mojoproject.toml`:

```bash
pixi install          # membuat environment di .pixi/
pixi shell            # masuk ke environment (opsional)
```

Tanpa `pixi shell`, semua perintah cukup lewat `pixi run <task>`.

## Menjalankan

### Demo pelatihan jaringan neural

```bash
pixi run demo
```

Contoh keluaran:

```
1) XOR - dua input, satu hidden layer 8 unit
loss  awal :  0.693147
loss  akhir :  0.000112
akurasi     :  100.0 %
```

### Asisten chat

```bash
pixi run chatbot
```

```
Ketik pesanmu. Perintah: 'quit' untuk keluar, 'help' untuk contoh.

Kamu > halo
MOC >  Halo! Senang bertemu denganmu.    [intent:  sapaan , keyakinan:  99 %]
Kamu > berapa 12 * 7
MOC >  Hasil perhitungannya: 84    [intent:  kalkulator , keyakinan:  99 %]
Kamu > quit
MOC > Sampai jumpa! Senang ngobrol denganmu.
```

Aplikasi ini juga bisa dipakai sebagai filter di dalam pipeline shell:

```bash
printf 'halo\nberapa 25 * 4\n' | pixi run chatbot
```

### Unit test

```bash
pixi run test
```

```
Running 28 tests for tests/test_mojo_ai.mojo
    PASS [ 0.001 ] test_rng_is_deterministic
    ...
--------
Summary [ 1253.945 ] 28 tests run: 28 passed , 0 failed , 0 skipped
```

### Format dan build binary

```bash
pixi run fmt          # mojo format src apps tests
pixi run build        # menghasilkan target/moc-demo dan target/moc-chat
./target/moc-chat     # jalan tanpa kompilasi ulang
```

---

## Cara kerja

### 1. Jaringan neural (`MLP`)

Bobot disimpan datar (flat) per layer supaya mudah di-cache:

```
w[w_off[li] + j * chunks[li] + c]   bobot dari unit input ke-c menuju unit output ke-j
b[b_off[li] + j]                    bias unit output ke-j
```

`chunks[li] = ceil(fan_in / 4)` karena perkalian dilakukan per chunk SIMD
lebar 4 (`SIMD[DType.float32, 4]`).

Satu epoch pada `fit()`:

1. Acak indeks sampel dengan PRNG deterministik.
2. Bagi menjadi mini-batch.
3. Untuk tiap sampel: forward pass → hitung delta → akumulasikan gradien bobot
   dan bias.
4. Perbarui bobot: `v = momentum * v + grad; w = w - lr * v`, plus regularisasi
   L2 opsional dan `lr_decay`.

Aktivasi yang tersedia: `ACT_RELU`, `ACT_TANH`, `ACT_SIGMOID`, `ACT_LINEAR`,
`ACT_SOFTMAX` (output). Loss: cross-entropy (softmax), binary cross-entropy
(sigmoid), atau MSE (linear).

Inisialisasi memakai He untuk hidden layer dan Xavier untuk layer output, dengan
skala `sqrt(6 / fan_in)` dan `sqrt(6 / (fan_in + fan_out))`.

### 2. Klasifikasi intent

```
teks  →  tokenisasi + stopword + stemming
      →  bag-of-words TF-IDF (normalisasi L2)
      →  MLP (softmax)
      →  intent + jawaban acak dari kumpulan respons
```

Kalau keyakinan di bawah `0.45` (dan bukan ekspresi aritmetika yang valid),
asisten menjawab dengan pesan cadangan alih-alih daripada menebak.

### 3. Pembaca baris stdin

`input()` bawaan Mojo membaca seluruh buffer stdin sekaligus, sehingga input
berupa pipe hanya menghasilkan satu baris. `LineReader` membaca langsung dari
file descriptor, memecah baris sendiri pada karakter newline, dan menangani
baris tanpa newline di akhir serta input gaya Windows (`\r\n`).

---

## Struktur proyek

```
.
├── mojoproject.toml        # manifest Pixi (channel, dependency, tasks)
├── pixi.lock               # lockfile dependency
├── apps/
│   ├── train_mlp.mojo      # demo pelatihan
│   └── chatbot.mojo        # asisten chat interaktif
├── src/mojo_ai/            # pustaka (dijalankan dengan -I src)
│   ├── nn.mojo
│   ├── datasets.mojo
│   ├── text.mojo
│   ├── intents.mojo
│   ├── calculator.mojo
│   ├── chatbot.mojo
│   ├── line_reader.mojo
│   └── rng.mojo
└── tests/
    └── test_mojo_ai.mojo   # unit test (TestSuite)
```

## Catatan

- Semua numerik literal di Mojo default ke `Float64`, jadi kode ini selalu
  menulis tipe eksplisit (`var x: Float32 = ...`).
- `List`/`Dict` bukan `ImplicitlyCopyable`; nilai ownership dipindahkan dengan
  `^` (mis. `return items^`).
- Mojo 1.1 tidak menyediakan modul `random` yang stabil, jadi PRNG
  xorshift64 ditulis sendiri di `rng.mojo`.

## Troubleshooting

### `error: unable to locate module 'std'`

Muncul kalau folder proyek dipindah/di-rename setelah `pixi install`. Environment
di `.pixi/` menyimpan path absolut (prefix) saat dibuat, sehingga compiler tidak
lagi menemukan `std.mojoc`. Perbaiki dengan memasang ulang environment:

```bash
pixi reinstall --offline   # atau: rm -rf .pixi && pixi install
```

## Lisensi

MIT
