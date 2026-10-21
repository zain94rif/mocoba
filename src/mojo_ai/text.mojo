"""Pemrosesan teks: tokenisasi, stopword, dan vektorisasi bag-of-words."""

from std.collections import Set
from std.math import log, sqrt

comptime WORD_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789"

# Kata umum Indonesia/Inggris yang diabaikan agar vektor lebih fokus.
comptime STOPWORDS = (
    " ada adalah adanya agar akan atau apa bagi bahwa banyak beberapa bisa buat"
    " dan dalam dari dengan di dia hanya hari ini itu jadi jangan jika juga"
    " kali kami kamu karena ke kita lagi lain maka masih mau nya oleh pada"
    " paling para pernah saja sama sangat saya sebagai sebuah sedang sehingga"
    " sejak sekarang selain semua satu seperti serta setelah setiap sini sudah"
    " supaya tak tanpa tapi telah tentang tersebut tetapi tidak untuk yaitu"
    " yang dong sih deh aja kok nih tuh banget yuk lho yang "
)


def _is_word_char(ch: StringSpan) raises -> Bool:
    return ch in WORD_CHARS


def is_stopword(word: String) raises -> Bool:
    """Cek stopword dengan pola ' spasi + kata + spasi ' agar tidak salah pilih.
    """
    var needle = " " + word + " "
    return needle in (" " + STOPWORDS + " ")


def tokenize(text: String) raises -> List[String]:
    """Lowercase, buang tanda baca, dan pecah menjadi kata."""
    var tokens = List[String]()
    var current = ""
    for ch in text.lower():
        if _is_word_char(ch):
            current += ch
        else:
            if current.byte_length() > 0:
                tokens.append(current^)
                current = ""
    if current.byte_length() > 0:
        tokens.append(current^)
    return tokens^


def stem(word: String) -> String:
    """Akar kata sederhana: buang sufiks -nya/-kan/-pun/-mu."""
    var n = word.byte_length()
    var cut = n
    if n > 5:
        var tail = word[byte = n - 3 : n]
        if tail == "nya" or tail == "kan" or tail == "pun":
            cut = n - 3
        elif word[byte = n - 2 : n] == "mu":
            cut = n - 2
    if cut == n:
        return word
    var out = ""
    var i = 0
    for ch in word:
        if i >= cut:
            break
        out += ch
        i += 1
    return out


def tokenize_filtered(text: String) raises -> List[String]:
    """Tokenisasi sekaligus membuang stopword dan token pendek."""
    var raw = tokenize(text)
    var tokens = List[String]()
    for tok in raw:
        if tok.byte_length() >= 2 and not is_stopword(tok):
            tokens.append(stem(tok))
    return tokens^


struct Vectorizer:
    """Bag-of-words dengan bobot TF-IDF.

    Vocabulary dibangun dari data latih; urutan indeks mengikuti urutan
    kemunculan pertama token sehingga hasilnya deterministik.
    """

    var vocab: Dict[String, Int]
    var idf: List[Float32]
    var n_features: Int

    def __init__(out self):
        self.vocab = Dict[String, Int]()
        self.idf = List[Float32]()
        self.n_features = 0

    def fit(mut self, texts: List[String]) raises:
        var df = Dict[String, Int]()
        var order = List[String]()
        var seen_all = Set[String]()
        var total = len(texts)
        for t in texts:
            var toks = tokenize_filtered(t)
            var seen = Set[String]()
            for tok in toks:
                if tok not in seen:
                    seen.add(tok)
                    if tok not in seen_all:
                        seen_all.add(tok)
                        order.append(tok.copy())
                    if tok in df:
                        df[tok] = df[tok] + 1
                    else:
                        df[tok] = 1
        for i in range(len(order)):
            var tok = order[i]
            self.vocab[tok] = i
            var d = df[tok]
            self.idf.append(
                Float32(log(Float64(total) / Float64(1 + d))) + Float32(1.0)
            )
        self.n_features = len(order)

    def transform(self, text: String) raises -> List[Float32]:
        """Ubah teks menjadi vektor fitur siap makan jaringan."""
        var vec = List[Float32]()
        for _ in range(self.n_features):
            vec.append(0.0)
        var toks = tokenize_filtered(text)
        for tok in toks:
            if tok in self.vocab:
                var idx = self.vocab[tok]
                vec[idx] += self.idf[idx]
        var norm: Float32 = 0.0
        for v in vec:
            norm += v * v
        norm = Float32(sqrt(norm))
        if norm > 1e-9:
            for i in range(len(vec)):
                vec[i] = vec[i] / norm
        return vec^

    def token_count(self) -> Int:
        return self.n_features
