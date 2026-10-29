from pypdf import PdfReader
deck = PdfReader(r"arizona-super-highways/pitch-deck/Arizona-Super-Highways-Pitch-Deck.pdf")
print("--- DECK SLIDE 1 ---")
print(deck.pages[0].extract_text()[:300])
print("--- DECK SLIDE 3 ---")
print(deck.pages[2].extract_text()[:500])
print("--- brace check ---")
bad = [i+1 for i,p in enumerate(deck.pages) if "{" in p.extract_text() or "map((" in p.extract_text()]
print("pages with stray code:", bad)
guide = PdfReader(r"arizona-super-highways/prototype/pocket-guide/Arizona-Super-Highways-Pocket-Passport.pdf")
print("--- GUIDE P1 ---")
print(guide.pages[0].extract_text()[:250])
print("--- GUIDE P3 ---")
print(guide.pages[2].extract_text()[:400])
