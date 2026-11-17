"""Assemble the revised manuscript, compile clean and tracked-changes PDFs."""
import os, re, shutil, subprocess, sys
os.makedirs(OUT, exist_ok=True)
m = open(f"{REV}/methods_part.tex").read(); a = open(f"{REV}/results_a.tex").read(); b = open(f"{REV}/results_b.tex").read()
abstract = open(f"{REV}/abstract.tex").read().strip()
tex = m.replace("%%ABSTRACT%%", abstract) + "\n" + a + "\n" + b
# title: risk aversion was inactive -> scope the title to stochastic scheduling
tex = tex.replace("Probabilistic Time-Series Foundation Models for Risk-Aware Microgrid Optimization", "Probabilistic Time-Series Foundation Models for Stochastic Microgrid Scheduling")
open(f"{OUT}/manuscript_R1_clean.tex", "w").write(tex)
for fn in os.listdir(f"{W}/figures"):
    shutil.copy(f"{W}/figures/{fn}", OUT)
for fn in ("schematic5.png", "proposalnsf.cls"):
    shutil.copy(f"{ORIG}/{fn}", OUT)
shutil.copy(f"{W}/references.bib", OUT)
# original for latexdiff
shutil.copy(f"{ORIG}/manuscript.tex", f"{OUT}/manuscript_original.tex")


def build(name):
    for cmd in (["pdflatex", "-interaction=nonstopmode", name], ["bibtex", name], ["pdflatex", "-interaction=nonstopmode", name], ["pdflatex", "-interaction=nonstopmode", name]):
        subprocess.run(cmd, cwd=OUT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log = open(f"{OUT}/{name}.log", errors="ignore").read()
    errs = [l for l in log.splitlines() if l.startswith("!")]
    undef = sorted(set(re.findall(r"(?:Reference|Citation) `([^']+)' .*undefined", log)))
    pages = subprocess.run(["pdfinfo", f"{OUT}/{name}.pdf"], capture_output=True, text=True).stdout
    pages = re.search(r"Pages:\s+(\d+)", pages); print(name, "pages", pages.group(1) if pages else "?", "errors", errs[:5], "undefined", undef)


build("manuscript_R1_clean")
if "--diff" in sys.argv:
    with open(f"{OUT}/manuscript_R1_tracked.tex", "w") as fh:
        subprocess.run(["latexdiff", "--type=CFONT", "--math-markup=coarse", "--disable-citation-markup", "--graphics-markup=none", "--exclude-textcmd=section,subsection,subsubsection,paragraph", "--config=PICTUREENV=(?:picture|DIFnomarkup|figure|table)[\\w\\d*@]*",
                        f"{OUT}/manuscript_original.tex", f"{OUT}/manuscript_R1_clean.tex"], cwd=OUT, stdout=fh, stderr=subprocess.DEVNULL)
    import re as _re
    O = {m[0]: m[1] for m in _re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", open(f"{ORIG}/manuscript.aux").read())}
    Rv = set(_re.findall(r"\\label\{([^}]*)\}", open(f"{OUT}/manuscript_R1_clean.tex").read()))
    t = open(f"{OUT}/manuscript_R1_tracked.tex").read()
    for lab, num in O.items():
        if lab not in Rv: t = t.replace("\\ref{" + lab + "}", num).replace("\\eqref{" + lab + "}", "(" + num + ")")
    open(f"{OUT}/manuscript_R1_tracked.tex", "w").write(t)
    build("manuscript_R1_tracked")
