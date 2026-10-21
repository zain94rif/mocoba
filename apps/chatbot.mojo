"""Asisten chat interaktif berbasis jaringan neural (100% Mojo).

Jalankan dengan:
    pixi run chatbot

Model dilatih ulang setiap kali aplikasi dibuka (hanya beberapa milidetik).
Input bisa datang dari keyboard (interaktif) maupun dari pipe, misalnya:
    printf 'halo\\nsiapa kamu\\n' | pixi run chatbot
"""

from mojo_ai.chatbot import ChatBot
from mojo_ai.line_reader import LineReader
import std.os


def main() raises:
    print("=" * 62)
    print(" MOC - Mojo Online Companion")
    print("=" * 62)
    print("Asisten AI kecil: klasifikasi intent + kalkulator aritmetika.")
    print("Semua ditulis dengan Mojo, tanpa Python dan tanpa pustaka ML.")
    print("-" * 62)

    var bot = ChatBot(seed=11, epochs=150)
    print("Melatih model...")
    var acc = bot.train()
    print("Selesai! Akurasi pada data latih: ", acc * 100.0, "%")
    print("-" * 62)
    print("Ketik pesanmu. Perintah: 'quit' untuk keluar, 'help' untuk contoh.")
    print()

    var interactive = std.os.isatty(0)
    var reader = LineReader(0)

    while True:
        if interactive:
            print("Kamu > ", end="", flush=True)
        var (line, ok) = reader.read_line()
        if not ok:
            if interactive:
                print()
            break

        var text = line.strip().lower()
        if text == "quit" or text == "exit" or text == "keluar":
            print("MOC > Sampai jumpa! Senang ngobrol denganmu.")
            break
        if text == "help" or text == "bantuan":
            print(
                "MOC > Contoh: 'halo', 'siapa kamu', 'berapa 12 * 7', "
                + "'apa itu machine learning', 'tolong bantu'"
            )
            continue
        if text.byte_length() == 0:
            continue

        var reply = bot.classify(text)
        print(
            "MOC > ",
            reply.text,
            "   [intent: ",
            reply.intent,
            ", keyakinan: ",
            String(Int(reply.confidence * 100.0)),
            "%]",
        )
