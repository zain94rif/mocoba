"""Dataset intent untuk asisten chat.

Setiap intent menyimpan contoh kalimat (dipisah dengan "|") dan beberapa
respons yang bisa dipilih acak.
"""


def split_pipe(text: String) -> List[String]:
    """Pecah teks berdasarkan pemisah "|" (menghasilkan String, bukan StringSpan).
    """
    var parts = List[String]()
    var current = ""
    for ch in text:
        if ch == "|":
            parts.append(current^)
            current = ""
        else:
            current += ch
    parts.append(current^)
    return parts^


struct Intent(Copyable):
    var name: String
    var samples: String
    var replies: String

    def __init__(out self, name: String, samples: String, replies: String):
        self.name = name
        self.samples = samples
        self.replies = replies

    def examples(self) -> List[String]:
        return split_pipe(self.samples)

    def responses(self) -> List[String]:
        return split_pipe(self.replies)


def default_intents() -> List[Intent]:
    """10 intent dasar untuk asisten chat berbahasa Indonesia."""
    var intents = List[Intent]()

    intents.append(
        Intent(
            "sapaan",
            (
                "halo|hallo|hai|hei|selamat pagi|selamat siang|selamat"
                " sore|selamat malam|apa kabar|kabar kamu|permisi|hai"
                " bot|hello|hi there|selamat datang|salam|met pagi|good"
                " morning|how are you|halo semua|halo teman"
            ),
            (
                "Halo! Senang bertemu denganmu.|Hai! Ada yang bisa"
                " kubantu?|Halo! Selamat datang di obrolan kita."
            ),
        )
    )

    intents.append(
        Intent(
            "perpisahan",
            (
                "dadah|dadah ya|sampai jumpa|bye|sampai nanti|selamat"
                " tinggal|bye bye|sudah ya|sampai bertemu|see you|good"
                " night|pamit|sampai ketemu|aku duluan ya"
            ),
            (
                "Sampai jumpa! Semoga harimu menyenangkan.|Dah! Sampai ketemu"
                " lagi ya.|Oke, bye! Jangan lupa untuk kembali."
            ),
        )
    )

    intents.append(
        Intent(
            "terima_kasih",
            (
                "terima kasih|makasih|thanks|terima kasih banyak|tgks|makasih"
                " ya|thanks a lot|nuhun|terima kasih sekali|banyak terima"
                " kasih|jazakumillah|makasih banyak sekali"
            ),
            (
                "Sama-sama! Senang bisa membantu.|Sama-sama, tugas selesai! Ada"
                " lagi yang bisa kubantu?|Sama-sama, dengan senang hati."
            ),
        )
    )

    intents.append(
        Intent(
            "bantuan",
            (
                "bantuan|tolong|bantu|tolong"
                " bantu|help|kemampuan|kemampuanmu|fungsi|fitur|menu|bantuan"
                " dong|tolong bantu ya|saya butuh bantuan|panduan|cara"
                " pakai|mohon bantuan"
            ),
            (
                "Aku bisa menyapa, menghitung, menjelaskan AI, dan mengobrol."
                " Silakan tanya apa saja!|Tentu! Kamu bisa tanya apa saja"
                " tentang AI atau meminta perhitungan."
            ),
        )
    )

    intents.append(
        Intent(
            "nama",
            (
                "siapa kamu|nama kamu apa|siapa namamu|kamu siapa|who are"
                " you|namamu siapa|perkenalkan diri|sebutkan nama|nama"
                " lengkapmu|kamu siapa sih|nama panggilanmu|namamu siapa"
            ),
            (
                "Namaku MOC, asisten AI kecil yang ditulis dengan Mojo.|Aku MOC"
                " - Mojo Online Companion. Senang bertemu!"
            ),
        )
    )

    intents.append(
        Intent(
            "kalkulator",
            (
                "hitung|kalkulator|calculate|perkalian|pembagian|penjumlahan|pengurangan|berapa"
                " hasil|berapa jumlah|berapa total|hasil"
                " hitungan|perhitungan|hitung dong"
            ),
            "Boleh, aku hitung ya.|Oke, aku kerjakan perhitungannya.",
        )
    )

    intents.append(
        Intent(
            "tentang_ai",
            (
                "apa itu ai|apa itu machine learning|apa itu mojo|apa itu"
                " neural network|machine learning|artificial"
                " intelligence|jaringan neural|deep learning|bahasa"
                " mojo|belajar machine learning|apa itu jaringan"
                " syaraf|mengenal ai|pintasan machine learning"
            ),
            (
                "AI adalah program yang belajar dari data. Contohnya: jaringan"
                " syaraf tiruan yang menebak dari contoh-contoh, persis seperti"
                " yang kamu pakai sekarang!|Machine learning = pola dari data."
                " Mojo = bahasa baru dari Modular untuk AI yang cepat."
            ),
        )
    )

    intents.append(
        Intent(
            "lelucon",
            (
                "lelucon|joke|lucu|cerita lucu|humor|tertawa|menghibur|lelucon"
                " lucu|ceritakan lelucon|ada lelucon"
            ),
            (
                "Kenapa komputer tidak lapar? Karena sudah messing with"
                " bytes.|Apa bedanya bug dan fitur? Tergantung dari"
                " perspektifmu."
            ),
        )
    )

    intents.append(
        Intent(
            "cuaca",
            (
                "cuaca|hujan|panas|mendung|berawan|how is the weather|cuaca"
                " hari ini|sunny|rainy|angin|cerah|deras|hujan deras"
            ),
            (
                "Aku tidak bisa melihat langit, sepertinya hari ini mendung."
                " Bawa payung saja biar aman!|Sayang, aku tidak punya sensor"
                " cuaca. Cek aplikasi cuaca ya."
            ),
        )
    )

    intents.append(
        Intent(
            "pujian",
            (
                "keren|hebat|mantap|bagus|wah|kamu pintar|like"
                " you|awesome|cool|hebat sekali|keren banget|luar biasa|jempol"
            ),
            (
                "Makasih! Aku senang bisa membantu.|Terima kasih! Aku belajar"
                " terus dari setiap percakapan."
            ),
        )
    )

    return intents^
