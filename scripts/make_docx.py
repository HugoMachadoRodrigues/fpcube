"""Word version of the ESSD manuscript, for editing before submission.

Reads manuscript/fpcube_essd.tex (with numbers.tex, tables/*.tex and figures/*.pdf as written by paper_numbers.py and
figures.py) and writes manuscript/fpcube_essd.docx on the Copernicus Word template, with Copernicus author-year
citations from references.bib. Copernicus-specific LaTeX commands become plain sections; figure, table and section
references are resolved to numbers here, because pandoc does not number them; figures are converted to 300 dpi PNG.

Once the .docx is edited by hand it becomes the master text: numbers regenerated later do not flow into it.

python scripts/make_docx.py
"""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

MS = Path(__file__).resolve().parents[1] / "manuscript"
BUILD = Path(tempfile.mkdtemp(prefix="fpcube_docx_"))   # intermediates stay out of OneDrive
OUT = MS / "fpcube_essd.docx"
TEMPLATE = MS / "word_template" / "Copernicus_Word_template.docx"
CSL = MS / "word_template" / "copernicus-publications.csl"
REFS_MARK = "REFERENCESGOHERE"


def braced(s, i):
    """Content of the balanced {...} group that opens at s[i]; returns (content, index after the group)."""
    assert s[i] == "{", s[i:i + 30]
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{" and s[j - 1] != "\\":
            depth += 1
        elif s[j] == "}" and s[j - 1] != "\\":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    raise ValueError("unbalanced braces")


def pop(s, name, nargs=1):
    """Remove every \\name[opt]{a}{b}... from s; return (s, [(opt, [args...]), ...])."""
    found = []
    pat = re.compile(r"\\" + name + r"(?![A-Za-z])")
    while (m := pat.search(s)):
        i, opt = m.end(), None
        if i < len(s) and s[i] == "[":
            k = s.index("]", i)
            opt, i = s[i + 1:k], k + 1
        args = []
        for _ in range(nargs):
            while s[i] in " \n":
                i += 1
            a, i = braced(s, i)
            args.append(a)
        found.append((opt, args))
        s = s[:m.start()] + s[i:]
    return s, found


def replace_cmd_with_section(s, name, heading):
    """\\name{text} -> \\section*{heading} text, in place."""
    pat = re.compile(r"\\" + name + r"(?![A-Za-z])")
    while (m := pat.search(s)):
        i = m.end()
        while s[i] in " \n":
            i += 1
        body, j = braced(s, i)
        s = s[:m.start()] + f"\\section*{{{heading}}}\n{body}\n" + s[j:]
    return s


def number_refs(s):
    """Number sections, figures and tables in order of appearance; resolve \\ref and prefix captions."""
    labels, sec, sub, fig, tab = {}, 0, 0, 0, 0
    env = None
    out, pos = [], 0
    token = re.compile(r"\\(section|subsection)(\*?)\{|\\begin\{(figure\*?|table\*?)\}|\\end\{(figure\*?|table\*?)\}"
                       r"|\\label\{([^}]+)\}|\\caption\{")
    last = None
    for m in token.finditer(s):
        if m.group(1) == "section" and not m.group(2):
            sec, sub = sec + 1, 0
            last = f"{sec}"
        elif m.group(1) == "subsection" and not m.group(2):
            sub += 1
            last = f"{sec}.{sub}"
        elif m.group(3):
            env = "Figure" if m.group(3).startswith("figure") else "Table"
            if env == "Figure":
                fig += 1
                last = str(fig)
            else:
                tab += 1
                last = str(tab)
        elif m.group(4):
            env = None
        elif m.group(5):
            labels[m.group(5)] = last
    # captions: prefix with "Figure n." / "Table n."
    fig = tab = 0
    env = None
    res, i = [], 0
    for m in re.finditer(r"\\begin\{(figure|table)\*?\}|\\caption\{", s):
        if m.group(1):
            env = m.group(1)
            if env == "figure":
                fig += 1
            else:
                tab += 1
            continue
        n = fig if env == "figure" else tab
        res.append(s[i:m.end()])
        res.append(f"\\textbf{{{'Figure' if env == 'figure' else 'Table'} {n}.}} ")
        i = m.end()
    res.append(s[i:])
    s = "".join(res)
    missing = sorted(set(re.findall(r"\\ref\{([^}]+)\}", s)) - set(labels))
    if missing:
        raise RuntimeError(f"unresolved references: {missing}")
    s = re.sub(r"\\ref\{([^}]+)\}", lambda m: labels[m.group(1)], s)
    return re.sub(r"\\label\{[^}]+\}", "", s)


def figures_png(s):
    (BUILD / "png").mkdir(parents=True, exist_ok=True)

    def conv(m):
        opts, name = m.group(1) or "", m.group(2)
        src = MS / (name if name.endswith(".pdf") else name + ".pdf")
        png = BUILD / "png" / (Path(name).stem + ".png")
        if src.exists():
            subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", str(src), str(png.with_suffix(""))],
                           check=True)
        elif (MS / (name + ".png")).exists():
            shutil.copy(MS / (name + ".png"), png)
        else:
            raise FileNotFoundError(src)
        return f"\\includegraphics{opts}{{{png}}}"
    return re.sub(r"\\includegraphics(\[[^\]]*\])?\{([^}]+)\}", conv, s)


def inline_tables(s):
    """Inline \\input{tables/...} and turn Copernicus table macros into plain LaTeX that pandoc reads."""
    def inp(m):
        t = (MS / (m.group(1) if m.group(1).endswith(".tex") else m.group(1) + ".tex")).read_text()
        t = re.sub(r"\\(tophline|middlehline|bottomhline)", r"\\hline", t)
        while (k := t.find("\\parbox")) >= 0:            # \parbox[t]{w}{\raggedright text} -> text
            i = k + len("\\parbox")
            if t[i] == "[":
                i = t.index("]", i) + 1
            _, i = braced(t, i)
            body, j = braced(t, i)
            t = t[:k] + body.replace("\\raggedright", "").strip() + t[j:]
        return t
    return re.sub(r"\\input\{(tables/[^}]+)\}", inp, s)


def reference_doc():
    """The Copernicus template plus the paragraph styles pandoc writes (Body Text, First Paragraph, Bibliography,
    captions...) taken from pandoc's default reference; without them LibreOffice draws stray marks after links."""
    import zipfile
    ref = BUILD / "pandoc_default.docx"
    subprocess.run(["pandoc", "-o", str(ref), "--print-default-data-file", "reference.docx"], check=True)
    tpl, pd_ = zipfile.ZipFile(TEMPLATE), zipfile.ZipFile(ref)
    ts, rs = tpl.read("word/styles.xml").decode(), pd_.read("word/styles.xml").decode()
    have = set(re.findall(r'w:styleId="([^"]+)"', ts))
    add = "".join(m.group(0) for m in re.finditer(r'<w:style [^>]*w:styleId="([^"]+)".*?</w:style>', rs, re.S)
                  if m.group(1) not in have)
    out = BUILD / "reference.docx"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for it in tpl.infolist():
            data = tpl.read(it.filename)
            z.writestr(it, ts.replace("</w:styles>", add + "</w:styles>").encode() if it.filename == "word/styles.xml"
                       else data)
    return out


def front_matter(title, authors, affils, corr):
    sup = lambda n: f"^{n}^" if len(affils) > 1 else "^1^"
    names = ", ".join(f"{a[0]} {a[1]}{sup(o)}" for o, a in authors)
    aff = "\n\n".join(f"^{o}^{a[0]}" for o, a in affils)
    return (f"::: {{custom-style=\"MS title\"}}\n{title}\n:::\n\n"
            f"::: {{custom-style=\"Authors\"}}\n{names}\n:::\n\n"
            f"::: {{custom-style=\"Affiliation\"}}\n{aff}\n:::\n\n"
            f"::: {{custom-style=\"Correspondence\"}}\nCorrespondence to: {corr}\n:::\n\n")


def main():
    BUILD.mkdir(exist_ok=True)
    s = (MS / "fpcube_essd.tex").read_text()
    s, title = pop(s, "title")
    s, authors = pop(s, "Author", 2)
    s, affils = pop(s, "affil")
    s, corr = pop(s, "correspondence")
    for c in ("runningtitle", "runningauthor", "received", "pubdiscuss", "revised", "accepted", "published",
              "firstpage", "bibliographystyle"):
        s, _ = pop(s, c)
    s = s.replace("\\maketitle", "")
    s = re.sub(r"\\documentclass(\[[^\]]*\])?\{copernicus\}", r"\\documentclass{article}", s)
    s = s.replace("\\begin{abstract}", "\\textbf{Abstract.} ").replace("\\end{abstract}", "")
    s = s.replace("\\introduction", "\\section{Introduction}").replace("\\conclusions", "\\section{Conclusions}")
    s = replace_cmd_with_section(s, "codedataavailability", "Code and data availability")
    s = replace_cmd_with_section(s, "authorcontribution", "Author contributions")
    s = replace_cmd_with_section(s, "competinginterests", "Competing interests")
    s = s.replace("\\begin{acknowledgements}", "\\section*{Acknowledgements}").replace("\\end{acknowledgements}", "")
    s = re.sub(r"\\bibliography\{[^}]+\}", REFS_MARK, s)
    # generated numbers inline, so that \\ref inside them is resolved with the rest
    s = s.replace("\\input{numbers}", (MS / "numbers.tex").read_text())
    s = inline_tables(s)
    # one-symbol inline math as plain characters: pandoc writes it as Word equations, which some readers draw
    # with an empty box (km$^2$ -> km[]2) or the wrong glyph ($>$)
    for tex_, ch in (("$>$", ">"), ("$<$", "<"), ("$^2$", "\u00b2"), ("$^3$", "\u00b3"), ("$\\geq$", "\u2265"),
                     ("$\\leq$", "\u2264"), ("$\\times$", "\u00d7"), ("$\\pm$", "\u00b1")):
        s = s.replace(tex_, ch)
    # degrees written as math ($-2\\,^\\circ$C, 3$^\\circ$) -> plain degree sign, which Word fonts all have
    s = re.sub(r"\$(-?)([0-9.]*)\s*(?:\\,)?\s*\^\{?\\circ\}?\$",
               lambda m: ("\u2212" if m.group(1) else "") + m.group(2) + ("\u2009" if m.group(2) else "") + "\u00b0", s)
    # wide (starred) floats and placement options make pandoc emit raw HTML figures/tables, dropped in Word
    s = re.sub(r"\\begin\{(figure|table)\*?\}(\[[^\]]*\])?", r"\\begin{\1}", s)
    s = re.sub(r"\\end\{(figure|table)\*\}", r"\\end{\1}", s)
    # \citep[prefix][]{key}: pandoc's LaTeX reader renders it as [prefix; Author (year)]; a placeholder is put back
    # as a markdown citation [prefix @key] after conversion, which gives (prefix Author, year)
    prefixed = []

    def keep(m):
        prefixed.append(f"[{m.group(1).replace(';', chr(92) + ';')} " + "; ".join("@" + k.strip() for k in m.group(2).split(",")) + "]")
        return f"PREFIXEDCITE{len(prefixed) - 1}X"
    s = re.sub(r"\\citep\[([^\]]*)\]\[\]\{([^}]+)\}", keep, s)
    s = number_refs(s)
    s = figures_png(s)
    tex = BUILD / "manuscript_word.tex"
    tex.write_text(s)
    md = subprocess.run(["pandoc", str(tex), "-f", "latex", "-t", "markdown", "--wrap=none"],
                        cwd=MS, check=True, capture_output=True, text=True).stdout
    if REFS_MARK not in md:
        raise RuntimeError("reference marker lost in conversion")
    md = re.sub(r"PREFIXEDCITE(\d+)X", lambda m: prefixed[int(m.group(1))], md)
    md = md.replace(REFS_MARK, "# References {.unnumbered}\n\n::: {#refs}\n:::\n")
    md = front_matter(title[0][1][0], [(int(o or 1), a) for o, a in authors],
                      [(int(o or 1), a) for o, a in affils], corr[0][1][0] if corr else "") + md
    (BUILD / "manuscript_word.md").write_text(md)
    subprocess.run(["pandoc", "manuscript_word.md", "-f", "markdown", "-o", str(OUT), "--citeproc",
                    f"--bibliography={MS / 'references.bib'}", f"--csl={CSL}", f"--reference-doc={reference_doc()}",
                    "--number-sections", f"--resource-path={MS}:{BUILD}"], cwd=BUILD, check=True)
    subprocess.run(["zip", "-q", "-d", str(OUT), "*/"], check=False, capture_output=True)
    shutil.rmtree(BUILD, ignore_errors=True)
    print(OUT)


if __name__ == "__main__":
    main()
