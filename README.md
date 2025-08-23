#FunctionFormer2

FunctionFormer2 is a self-healing autocoder that turns a high-level goal into a working Python script.
It plans an outline, generates the code section-by-section, runs pre-flight fixes, executes the program in a sandboxed subprocess, and iteratively repairs issues using your feedback.


This repo ships two launchers:

ff2_mistral7b.py — local, offline(ish) build targeting Mistral 7B Instruct v0.3

ff2_chatgpt.py — API build targeting OpenAI ChatGPT models

⚠️ You’ll also need two helper modules placed next to the ff2_*.py file you run:

Overload.py (a lightweight, layer-wise runner / fallback engine)

LLM_Reasoning_Engine.py (small “planning/critique” helper)

These are pulled from their own repos on my Github page. Drop the files in the same folder as ff2_mistral7b.py / ff2_chatgpt.py.



✨ What it does

Interview → Outline → Code: asks a few clarifying questions, produces a structured outline, then generates imports, globals, functions, and main in small passes.

Strict reviewers: every section is critiqued across multiple aspects (spec match, structure, dependencies, readiness, style) and rewritten if needed.

Pre-run QA sweep: repairs sections against the outline before any execution.

Run & auto-fix loop: executes your script in a separate Python process, analyzes tracebacks, picks the smallest fix target, and patches just that part.

Human-in-the-loop: after each run, you can say “ship it”, or give feedback like “rename X”, “add logging”, “call Y first”—and it will apply one minimal change.

Snapshots: saves versions under ./script_snapshots/ so you can diff the evolution.



🧰 Requirements

Python: 3.10–3.12 recommended

OS: Windows, Linux, or macOS (Windows works; see bitsandbytes note below)

Core deps (install these):

pip install --upgrade torch transformers bitsandbytes psutil


tkinter ships with most Python installers (on Linux: sudo apt install python3-tk if needed).

Model (for the Mistral build):

Mistral 7B Instruct v0.3 weights, accessible locally on disk.

bitsandbytes on Windows: 8-bit quantization support can be finicky. If the HF load fails, FunctionFormer2 automatically falls back to Overload.py (its layer-wise runner). You can still use GPU memory when available; otherwise it will stream layers via CPU.

⚙️ Setup

Clone this repo.

Put Overload.py and LLM_Reasoning_Engine.py in the same folder as the launcher you plan to run.



(Mistral build) Download Mistral 7B Instruct v0.3 to a local folder, e.g.:

D:/models/mistral-7b-instruct-v0.3


Open ff2_mistral7b.py and set:

BASE_MODEL_PATH = "D:/models/mistral-7b-instruct-v0.3"


That path should contain config.json, tokenizer files, and the safetensors.



OpenAI / ChatGPT build

For ff2_chatgpt.py:

In ff2_chatgpt.py, pick the model name you want to use (e.g., gpt-4o), and then provide your API key.




The app opens a Tkinter GUI with three columns:

Outline (editable) — live outline that seeds each generation pass

Script (editable) — the code being built & revised

Goal / Status — your current goal, plus live logs

Flow:

Enter your overall goal (e.g., “A Tkinter CSV deduper with preview & export”).

Answer the short interview (Ctrl+Enter to submit).

Review & confirm the outline (a popup will show it).

Watch it generate imports → globals → functions → main.

Pre-run QA sweep runs; then it asks permission to execute your script.

After each run, give feedback or say “ship it”. It will apply one minimal edit per round.

Shortcuts: Ctrl+Enter submits text in popups.



🧩 Troubleshooting

Model won’t fit / bitsandbytes error
The app should print “Using Overload fallback”. Keep Overload.py next to the launcher and ensure BASE_MODEL_PATH has the model shards & tokenizer.

No GUI / Tkinter error
Install tk: sudo apt install python3-tk (Linux). On macOS/Windows, use the official Python.org installer.

“NameError: X is not defined” during run
The autocoder will attempt to synthesize a missing function based on traceback. If it can’t, add a brief description as feedback (“Create X to do …”) and run again.

Feedback is ignored
It applies one minimal change per run. Keep feedback short and specific (e.g., “rename foo to bar in main”, “call init_ui() before load_config()”).

🔒 Safety & execution

This tool executes generated Python in a subprocess you control. Only run it on projects and machines you trust. Review code in the Script pane before authorizing the first run.




⭐ Support

If you find this useful, star the repo and open issues/PRs with concrete repros or improvements.
You can also browse my other projects here: github.com/garagesteve1155.
