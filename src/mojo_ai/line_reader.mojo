"""Pembaca baris dari stdin yang tidak bergantung pada `input()`.

`input()` bawaan Mojo membaca sekaligus seluruh isi buffer stdin, sehingga
piped input seperti `printf 'satu\\ndua\\n' | app` hanya menghasilkan satu
baris lalu langsung EOF.

`LineReader` membaca langsung dari file descriptor, memecah buffer sendiri
pada karakter newline, dan mengembalikan satu baris per panggilan. moduli ini
membuat program Mojo bisa dipakai sebagai filter/CLI yang interaktif maupun
dipakai dalam pipeline shell.
"""

from std.io.file_descriptor import FileDescriptor

comptime DEFAULT_CAPACITY = 4096
comptime BYTE_NEWLINE = UInt8(10)
comptime BYTE_CARRIAGE_RETURN = UInt8(13)
comptime BYTE_QUESTION = UInt8(63)


struct LineReader:
    """Pembaca baris streaming di atas file descriptor."""

    var fd: FileDescriptor
    var buf: List[UInt8]
    var start: Int
    var end: Int
    var eof: Bool

    def __init__(out self, fd: Int = 0, capacity: Int = DEFAULT_CAPACITY):
        self.fd = FileDescriptor(fd)
        self.buf = List[UInt8]()
        self.buf.resize(capacity, UInt8(0))
        self.start = 0
        self.end = 0
        self.eof = False

    def _compact(mut self):
        """Geser byte yang belum diproses ke depan buffer."""
        if self.start == 0:
            return
        var remaining = self.end - self.start
        var i = 0
        while i < remaining:
            self.buf[i] = self.buf[self.start + i]
            i += 1
        self.start = 0
        self.end = remaining

    def _grow(mut self):
        """Perbesar buffer dua kali lipat (baris sangat panjang)."""
        var new_capacity = len(self.buf) * 2
        if new_capacity < 64:
            new_capacity = 64
        var grown = List[UInt8]()
        grown.resize(new_capacity, UInt8(0))
        var i = 0
        while i < self.end:
            grown[i] = self.buf[i]
            i += 1
        self.buf = grown^
        self.start = 0

    def _fill(mut self) raises -> Bool:
        """Baca byte baru dari file descriptor. False berarti sudah EOF."""
        if self.eof:
            return False
        self._compact()
        if self.end >= len(self.buf):
            self._grow()
        var span = Span[UInt8](self.buf)
        var n = self.fd.read_bytes(span[self.end : len(self.buf)])
        if n <= 0:
            self.eof = True
            return False
        self.end += n
        return True

    def _is_continuation(self, byte: UInt8) -> Bool:
        """True untuk byte continuation UTF-8 (10xxxxxx)."""
        return byte >= 0x80 and byte < 0xC0

    def _sanitize(self, a: Int, b: Int) -> List[UInt8]:
        """Salin rentang byte menjadi UTF-8 yang valid.

        Byte yang tidak membentuk urutan UTF-8 yang sah diganti dengan '?'
        supaya program tidak gagal saat input berisi data biner.
        """
        var out = List[UInt8]()
        var i = a
        while i < b:
            var c = self.buf[i]
            if c < 0x80:
                out.append(c)
                i += 1
                continue
            var size = 0
            var low2: UInt8 = 0x80
            var high2: UInt8 = 0xBF
            if c >= 0xC2 and c <= 0xDF:
                size = 2
            elif c == 0xE0:
                size = 3
                low2 = 0xA0
            elif c >= 0xE1 and c <= 0xEC:
                size = 3
            elif c == 0xED:
                size = 3
                high2 = 0x9F
            elif c == 0xEE or c == 0xEF:
                size = 3
            elif c == 0xF0:
                size = 4
                low2 = 0x90
            elif c >= 0xF1 and c <= 0xF3:
                size = 4
            elif c == 0xF4:
                size = 4
                high2 = 0x8F
            if size == 0 or i + size > b:
                out.append(BYTE_QUESTION)
                i += 1
                continue
            var ok = self.buf[i + 1] >= low2 and self.buf[i + 1] <= high2
            var k = 2
            while ok and k < size:
                ok = self._is_continuation(self.buf[i + k])
                k += 1
            if not ok:
                out.append(BYTE_QUESTION)
                i += 1
                continue
            var j = 0
            while j < size:
                out.append(self.buf[i + j])
                j += 1
            i += size
        return out^

    def _text(self, a: Int, b: Int) raises -> String:
        """Ubah rentang byte menjadi String (tanpa newline)."""
        var i = a
        var ascii_only = True
        while i < b:
            if self.buf[i] >= 0x80:
                ascii_only = False
                break
            i += 1
        if ascii_only:
            var span = Span[UInt8](self.buf)
            return String(StringSpan(from_utf8=span[a:b]))
        var clean = self._sanitize(a, b)
        var span2 = Span[UInt8](clean)
        return String(StringSpan(from_utf8=span2))

    def read_line(mut self) raises -> Tuple[String, Bool]:
        """Baca satu baris.

        Mengembalikan `(teks, True)` bila ada baris yang terbaca, dan
        `(String(""), False)` saat stdin sudah habis (EOF).
        """
        while True:
            var i = self.start
            while i < self.end:
                if self.buf[i] == BYTE_NEWLINE:
                    # Buang carriage return untuk input bergaya Windows.
                    var stop = i
                    if (
                        stop > self.start
                        and self.buf[stop - 1] == BYTE_CARRIAGE_RETURN
                    ):
                        stop -= 1
                    var line = self._text(self.start, stop)
                    self.start = i + 1
                    return (line, True)
                i += 1
            if not self._fill():
                if self.start < self.end:
                    var line = self._text(self.start, self.end)
                    self.start = self.end
                    return (line, True)
                return (String(""), False)

    def read_all(mut self) raises -> List[String]:
        """Baca seluruh stdin lalu pecah menjadi daftar baris."""
        var lines = List[String]()
        while True:
            var (line, ok) = self.read_line()
            if not ok:
                return lines^
            lines.append(line)
