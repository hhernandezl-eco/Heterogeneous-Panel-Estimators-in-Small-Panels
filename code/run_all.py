"""Run the complete pipeline: data -> panel -> descriptives -> diagnostics -> tests -> estimation -> robustness
-> Monte-Carlo summaries -> CCE anatomy -> paper.

Usage (from the project root):
    python code/run_all.py              # everything using the cached Monte-Carlo draws in output/mc/ (~3 min)
    python code/run_all.py --download   # also refresh the raw data from the World Bank / UNDP
    python code/run_all.py --montecarlo # also re-run the Monte-Carlo study (2,000 replications per cell, ~1-1.5 h on 12 cores)
                                        # and the size/power study of the cointegration tests (~10 min)
"""
import subprocess, sys, shutil, time, os
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
steps = []                                       # (script path, extra args)
if "--download" in sys.argv:
    steps.append((CODE / "01_download_data.py", []))
for f in ["02_build_panel.py", "03_descriptives.py", "04_diagnostics.py", "05_cointegration.py",
          "06_estimation.py", "07_causality.py", "08_robustness.py", "09_accounting.py"]:
    steps.append((CODE / f, []))
if "--montecarlo" in sys.argv:
    steps.append((CODE / "10_montecarlo.py", ["run", "2000"]))
steps.append((CODE / "10_montecarlo.py", ["summarize"]))
steps.append((CODE / "11_cce_anatomy.py", []))
steps.append((ROOT / "tests" / "validate_code.py", []))
steps.append((ROOT / "tests" / "export_validation.py", []))
steps.append((ROOT / "validation" / "compare_external.py", []))   # needs validation/external_results_r.csv (run validation/check_r.R once)
if "--montecarlo" in sys.argv:
    steps.append((ROOT / "tests" / "power_cointegration.py", ["500", "99"]))

env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONWARNINGS": "ignore"}
for s, args in steps:
    t0 = time.time()
    tag = s.stem + ("_" + args[0] if args else "")
    print(f"\n=== {s.name} {' '.join(args)} ===", flush=True)
    log = ROOT / "output" / "logs" / (tag + ".log")
    with open(log, "w", encoding="utf-8") as fh:
        r = subprocess.run([sys.executable, str(s), *args], cwd=str(CODE), env=env, stdout=fh, stderr=subprocess.STDOUT)
    print(f"   exit {r.returncode}  ({time.time() - t0:.0f}s)  log: {log.relative_to(ROOT)}", flush=True)
    if r.returncode != 0:
        print(log.read_text(encoding="utf-8")[-1500:])
        sys.exit(r.returncode)

paper = ROOT / "paper"
if shutil.which("latexmk"):
    print("\n=== compiling paper/main.tex ===")
    subprocess.run(["latexmk", "-g", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=str(paper))
else:
    print("latexmk not found: compile paper/main.tex manually (pdflatex + bibtex)")
