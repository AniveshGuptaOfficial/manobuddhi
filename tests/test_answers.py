from engine.data import answers as A


def test_gsm8k_gold():
    assert A.gsm8k_gold("5 + 5 = 10\n#### 1,250") == "1250"


def test_norm_number():
    assert A.norm_number("$0.50") == "0.5"
    assert A.norm_number("$20.0") == "20"
    assert A.norm_number("-3") == "-3"
    assert A.norm_number("None") == ""


def test_last_boxed_nested():
    s = r"first \boxed{1} then \boxed{\left( 3, \frac{\pi}{2} \right)}."
    assert A.last_boxed(s) == r"\left( 3, \frac{\pi}{2} \right)"
    assert A.last_boxed("no answer") is None
