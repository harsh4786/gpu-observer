"""Check submission/form.md against lablab's field limits."""
import pathlib, re, sys

text = pathlib.Path(__file__).with_name("form.md").read_text()

def section(name):
    m = re.search(rf"^## {re.escape(name)}\n\n(.*?)(?=\n## |\Z)", text, re.S | re.M)
    return m.group(1).strip() if m else ""

title, short, long_ = section("Title"), section("Short description"), section("Long description")
checks = [
    ("title", len(title), "chars", len(title) <= 50, "<= 50"),
    ("short description", len(short), "chars", len(short) <= 255, "<= 255"),
    ("long description", len(long_.split()), "words", len(long_.split()) >= 100, ">= 100"),
]
ok = True
for name, value, unit, passed, rule in checks:
    ok &= passed
    print(f"{'PASS' if passed else 'FAIL'}  {name}: {value} {unit} ({rule})")
sys.exit(0 if ok else 1)
