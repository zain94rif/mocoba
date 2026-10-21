"""Kalkulator ekspresi aritmetika sederhana.

Parser rekursif descendant untuk operasi +, -, *, /, dan tanda kurung.
Dipakai oleh asisten chat untuk menjawab pertanyaan seperti "12 * 7 = ?".
"""


def _is_digit(ch: StringSpan) -> Bool:
    return ch in "0123456789"


def _is_number(tok: String) -> Bool:
    var digits = 0
    var dots = 0
    for ch in tok:
        if _is_digit(ch):
            digits += 1
        elif ch == ".":
            dots += 1
        else:
            return False
    return digits > 0 and dots <= 1


def _parse_number(tok: String) -> Float32:
    var value: Float32 = 0.0
    var frac: Float32 = 0.1
    var after_dot = False
    for ch in tok:
        if ch == ".":
            after_dot = True
        else:
            var d = Float32(ord(ch) - ord("0"))
            if after_dot:
                value += d * frac
                frac /= 10.0
            else:
                value = value * 10.0 + d
    return value


def tokenize_expression(text: String) -> List[String]:
    """Ambil token angka/operator dari teks bebas."""
    var tokens = List[String]()
    var current = ""
    for ch in text:
        if _is_digit(ch) or ch == ".":
            current += ch
        elif (
            ch == "+"
            or ch == "-"
            or ch == "*"
            or ch == "/"
            or ch == "("
            or ch == ")"
        ):
            if current.byte_length() > 0:
                tokens.append(current^)
                current = ""
            if ch == "+":
                tokens.append("+")
            elif ch == "-":
                tokens.append("-")
            elif ch == "*":
                tokens.append("*")
            elif ch == "/":
                tokens.append("/")
            elif ch == "(":
                tokens.append("(")
            else:
                tokens.append(")")
        else:
            if current.byte_length() > 0:
                tokens.append(current^)
                current = ""
    if current.byte_length() > 0:
        tokens.append(current^)
    return tokens^


struct CalcResult(Copyable):
    var value: Float32
    var ok: Bool

    def __init__(out self, value: Float32 = 0.0, ok: Bool = False):
        self.value = value
        self.ok = ok


struct Parser:
    var toks: List[String]
    var pos: Int
    var ok: Bool

    def __init__(out self, toks: List[String]):
        self.toks = toks.copy()
        self.pos = 0
        self.ok = True

    def peek(self) -> String:
        if self.pos < len(self.toks):
            return self.toks[self.pos]
        return ""

    def parse_expr(mut self) -> Float32:
        var value = self.parse_term()
        while True:
            var op = self.peek()
            if op == "+":
                self.pos += 1
                value += self.parse_term()
            elif op == "-":
                self.pos += 1
                value -= self.parse_term()
            else:
                break
        return value

    def parse_term(mut self) -> Float32:
        var value = self.parse_factor()
        while True:
            var op = self.peek()
            if op == "*":
                self.pos += 1
                value *= self.parse_factor()
            elif op == "/":
                self.pos += 1
                var d = self.parse_factor()
                if d == 0.0:
                    self.ok = False
                    return 0.0
                value /= d
            else:
                break
        return value

    def parse_factor(mut self) -> Float32:
        var tok = self.peek()
        if tok == "-":
            self.pos += 1
            return -self.parse_factor()
        elif tok == "+":
            self.pos += 1
            return self.parse_factor()
        elif tok == "(":
            self.pos += 1
            var value = self.parse_expr()
            if self.peek() == ")":
                self.pos += 1
            else:
                self.ok = False
            return value
        elif _is_number(tok):
            self.pos += 1
            return _parse_number(tok)
        else:
            self.ok = False
            return 0.0


def evaluate(text: String) -> CalcResult:
    """Hitung ekspresi aritmetika yang ada di dalam teks.

    Mengembalikan `ok = False` bila teks tidak mengandung ekspresi yang
    valid (minimal satu angka dan satu operator).
    """
    var toks = tokenize_expression(text)
    var has_number = False
    var has_operator = False
    for t in toks:
        if _is_number(t):
            has_number = True
        elif t == "+" or t == "-" or t == "*" or t == "/":
            has_operator = True
    if not has_number or not has_operator:
        return CalcResult(0.0, False)
    var parser = Parser(toks)
    var value = parser.parse_expr()
    if not parser.ok or parser.pos != len(toks):
        return CalcResult(0.0, False)
    return CalcResult(value, True)


def format_number(value: Float32) -> String:
    """Format angka: bulat tanpa desimal, pecahan dibatasi 4 digit."""
    var rounded = value
    if rounded == Float32(Int(rounded)) and abs(rounded) < 1.0e9:
        return String(Int(rounded))
    var scaled = Int(rounded * 10000.0)
    var whole = scaled // 10000
    var frac = scaled % 10000
    if frac < 0:
        frac = -frac
    var frac_str = String(frac)
    while frac_str.byte_length() < 4:
        frac_str = "0" + frac_str
    return String(whole) + "." + frac_str
