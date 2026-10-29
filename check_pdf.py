from pypdf import PdfReader
for f in [r"arizona-super-highways/prototype/pocket-guide/Arizona-Super-Highways-Pocket-Passport.pdf",
          r"arizona-super-highways/pitch-deck/Arizona-Super-Highways-Pitch-Deck.pdf"]:
    r = PdfReader(f)
    p = r.pages[0]
    w = float(p.mediabox.width)/72; h = float(p.mediabox.height)/72
    print(f, "pages:", len(r.pages), "size:", round(w,2), "x", round(h,2), "in")
