import subprocess
import os

env = os.environ.copy()
env["PYTHONUTF8"] = "1"

# subprocess.run([
#     "panel", "convert", "LM_mod_app_clean.py", "--to", "pyodide-worker", "--out", "docs"
# ], env=env)

subprocess.run([
    "panel", "convert", "LM_mod_app.py", "--to", "pyodide-worker", "--out", "docs"
], env=env)
