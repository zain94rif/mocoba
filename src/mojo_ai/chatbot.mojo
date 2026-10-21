"""Asisten chat sederhana: klasifikasi intent memakai jaringan neural.

Alur:
    teks -> tokenisasi -> bag-of-words TF-IDF -> MLP (softmax) -> intent
"""

from mojo_ai.nn import MLP, ACT_RELU, ACT_SOFTMAX
from mojo_ai.text import Vectorizer
from mojo_ai.intents import Intent, default_intents
from mojo_ai.calculator import evaluate, format_number
from mojo_ai.rng import Rng

comptime CONFIDENCE_FLOOR = 0.45


struct Reply(Copyable):
    var text: String
    var intent: String
    var confidence: Float32

    def __init__(out self, text: String, intent: String, confidence: Float32):
        self.text = text
        self.intent = intent
        self.confidence = confidence


struct ChatBot:
    var intents: List[Intent]
    var vec: Vectorizer
    var net: MLP
    var rng: Rng
    var epochs: Int
    var last_accuracy: Float32

    def __init__(out self, seed: UInt64 = 11, epochs: Int = 80):
        self.intents = default_intents()
        self.vec = Vectorizer()
        self.rng = Rng(seed)
        self.epochs = epochs
        self.last_accuracy = 0.0
        var sizes = List[Int]()
        sizes.append(2)
        sizes.append(2)
        self.net = MLP(sizes, ACT_RELU, ACT_SOFTMAX, seed=seed, lr=0.05)

    def intent_index(self, name: String) -> Int:
        for i in range(len(self.intents)):
            if self.intents[i].name == name:
                return i
        return -1

    def train(mut self) raises -> Float32:
        """Bangun dataset dari contoh intent, lalu latih jaringan."""
        var texts = List[String]()
        var labels = List[Int]()
        for i in range(len(self.intents)):
            var examples = self.intents[i].examples()
            for e in examples:
                texts.append(e)
                labels.append(i)

        self.vec.fit(texts)

        var x = List[Float32]()
        var y = List[Float32]()
        for i in range(len(texts)):
            var features = self.vec.transform(texts[i])
            for v in features:
                x.append(v)
            for k in range(len(self.intents)):
                if k == labels[i]:
                    y.append(1.0)
                else:
                    y.append(0.0)

        var sizes = List[Int]()
        sizes.append(self.vec.n_features)
        sizes.append(48)
        sizes.append(len(self.intents))
        self.net = MLP(
            sizes,
            ACT_RELU,
            ACT_SOFTMAX,
            seed=21,
            lr=0.12,
            momentum=0.9,
        )
        _ = self.net.fit(
            x,
            y,
            len(texts),
            epochs=self.epochs,
            batch_size=8,
            seed=33,
            lr_decay=0.02,
        )
        self.last_accuracy = self.net.accuracy(x, y, len(texts))
        return self.last_accuracy

    def classify(mut self, text: String) raises -> Reply:
        """Klasifikasikan teks jadi intent, plus jawaban yang sesuai."""
        var features = self.vec.transform(text)
        # Kalau tidak ada satu pun kata yang dikenal, jangan speculate.
        var known: Float32 = 0.0
        for f in features:
            known += f
        if known < 1e-6:
            return Reply(
                "Maaf, aku belum paham maksudmu. Coba tanya soal sapaan, "
                + "perhitungan, atau hal-hal dasar ya.",
                "tidak_diketahui",
                0.0,
            )
        var probs = self.net.predict_proba(features)
        var best = 0
        for i in range(len(probs)):
            if probs[i] > probs[best]:
                best = i
        var confidence = probs[best]
        var intent = self.intents[best].copy()

        # Perhitungan arithmetic didahulukan bila ekspresinya valid.
        var calc = evaluate(text)
        var calc_index = self.intent_index("kalkulator")
        if calc.ok and (best == calc_index or confidence < 0.7):
            return Reply(
                "Hasil perhitungannya: " + format_number(calc.value),
                "kalkulator",
                0.99,
            )

        if confidence < CONFIDENCE_FLOOR:
            return Reply(
                "Maaf, aku belum paham maksudmu. Coba tanya soal sapaan, "
                + "perhitungan, atau hal-hal dasar ya.",
                "tidak_diketahui",
                confidence,
            )

        var responses = intent.responses()
        var idx = self.rng.randint(0, len(responses) - 1)
        return Reply(responses[idx], intent.name, confidence)
