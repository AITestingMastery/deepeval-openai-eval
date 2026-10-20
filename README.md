# LLM Evaluation with DeepEval + OpenAI

A demo project that **evaluates the quality of LLM answers**. You ask a question, an OpenAI model answers, and [DeepEval](https://github.com/confident-ai/deepeval) scores the answer on up to 13 metrics: relevancy, correctness, faithfulness, clarity and more.

It covers two kinds of AI apps:

| Mode | What it does | What gets evaluated |
|---|---|---|
| 💬 **LLM Chat** | Question → OpenAI → answer | Is the answer good? (10 answer metrics) |
| 📄 **RAG (chat with documents)** | Question → search your documents → OpenAI answers from them | Did it find the right document parts? Is the answer faithful to them? (3 retrieval + 10 answer metrics) |

There are two ways to use it:

- **Streamlit app** (`app.py`): a chat window that shows the scores under every answer. Best for demos.
- **Command-line scripts**: evaluate a whole test set at once, compare settings and generate test questions automatically.

---

## Table of contents

1. [Prerequisites](#1-prerequisites)
2. [Clone the repository](#2-clone-the-repository)
3. [Create a virtual environment](#3-create-a-virtual-environment)
4. [Install the dependencies](#4-install-the-dependencies)
5. [Add your OpenAI API key](#5-add-your-openai-api-key)
6. [Run the Streamlit app](#6-run-the-streamlit-app)
7. [Run the command-line scripts](#7-run-the-command-line-scripts)
8. [Metrics explained](#8-metrics-explained)
9. [Project structure](#9-project-structure)
10. [Configuration](#10-configuration)
11. [Optional: Confident AI dashboard](#11-optional-confident-ai-dashboard)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.10 or newer** | Check with `python --version`. Download from https://www.python.org/downloads/. On Windows, tick **"Add Python to PATH"** during install. |
| **Git** | Check with `git --version`. Download from https://git-scm.com/downloads |
| **OpenAI API key** | Create one at https://platform.openai.com/api-keys. The account needs billing enabled, because both the answers and the scoring use the OpenAI API. |

> **Cost note:** each answer is scored by an LLM judge, so every scored question makes several OpenAI calls. With the default `gpt-4o-mini` this is cheap, usually a fraction of a cent per question.

---

## 2. Clone the repository

```bash
git clone <your-repo-url>
cd deepeval-openai-eval
```

All the commands below are run **from inside this project folder**.

---

## 3. Create a virtual environment

A virtual environment keeps this project's packages separate from the rest of your system.

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt)**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

When it's active, `(.venv)` appears at the start of your terminal prompt. **Activate it again every time you open a new terminal.**

> On Windows, if PowerShell says *"running scripts is disabled on this system"*, run this once and then activate again:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

---

## 4. Install the dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs DeepEval, OpenAI, Streamlit, pandas, and the PDF and Word readers used by RAG mode. It takes a minute or two the first time.

Check that it worked:
```bash
deepeval --version
streamlit --version
```

---

## 5. Add your OpenAI API key

Copy the example settings file to `.env`:

**Windows**
```powershell
copy .env.example .env
```

**macOS / Linux**
```bash
cp .env.example .env
```

Open `.env` in any text editor and paste your key:

```dotenv
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxx
```

That is the only required setting. The other values have sensible defaults (see [Configuration](#10-configuration)).

> `.env` is listed in `.gitignore`, so your key is never committed to Git. Never share this file.

---

## 6. Run the Streamlit app

```bash
streamlit run app.py
```

Your browser opens at **http://localhost:8501**. If it doesn't, open that address yourself. To stop the app, press `Ctrl + C` in the terminal.

### 6.1 💬 LLM Chat mode

1. In the sidebar, keep **Mode → 💬 Chat**.
2. Type a question in the chat box at the bottom, for example *"What are some healthy breakfast ideas?"*
3. The answer appears, followed by **10 metric scores**, each marked ✅ Pass or ❌ Fail.
4. Click **"Why these scores?"** to see the judge's reason for each score.

**Optional sidebar fields** (fill them in *before* asking):

| Field | What it does |
|---|---|
| **Expected answer** | Correctness and Coverage compare the answer against this |
| **Source context** | Text the answer must be based on. Faithfulness and Traceability check against it |
| **System prompt** | Instructions to the model. Prompt Adherence checks whether they were followed |
| **Pass threshold** | Minimum score for a Pass (default 0.7) |
| **Metrics to run** | Turn metrics off to make replies faster and cheaper |

**Try these to see a mix of passes and fails:**

| Question | What to expect |
|---|---|
| `What are some healthy breakfast ideas?` | Almost everything passes |
| `Is it good?` | Relevancy, Helpfulness and Coverage fail (the question is vague) |
| `Tell me 5 tips for good sleep, but answer in only 3 words.` | Prompt Adherence and Coverage fail |
| `Explain why drinking water is important, and repeat every point three times.` | Redundancy fails |

### 6.2 📄 RAG mode (chat with documents)

1. In the sidebar, choose **Mode → 📄 Chat with documents (RAG)**.
2. Under **Documents**, keep **Sample documents** (a fictional company's leave, travel and work-from-home policies), or choose **Upload my own** to add PDF, Word, TXT or Markdown files.
3. Wait for **"Indexed N chunks from M file(s)"**.
4. Ask a question about the documents.

Every answer shows:
- **📄 Retrieved chunks**: the parts of the documents that were used, with the file name and similarity score
- **🔎 Retrieval quality**: Contextual Relevancy, Precision and Recall
- **✍️ Answer quality**: the 10 answer metrics, now checked against the retrieved chunks
- **⏱️ Timing**: retrieval, generation and scoring time

**Try these with the sample documents:**

| Question | Expected answer (optional, sidebar) | What it shows |
|---|---|---|
| `How many casual leaves do I get?` | `12 casual leaves per year` | Direct answer from one document |
| `What is the hotel limit in Hyderabad?` | `5,000 rupees per night` | Picks the right document (travel policy) |
| `Can I work from Goa for a month?` | `No, only up to 2 weeks per year` | Has to combine rules |
| `What is the company's dress code?` | — | Not in the documents, so it should say it couldn't find it (no made-up answer) |

> Contextual Precision and Recall need an **Expected answer**. Without one, they show as *Skipped*.

Use **top-k** and **chunk size** in the sidebar to see how retrieval settings change the scores. At the bottom of the page, **Score history** collects every question and its scores, and **Download as CSV** saves them.

---

## 7. Run the command-line scripts

These evaluate many questions at once, with no UI. Reports are saved to the `reports/` folder as CSV files you can open in Excel.

### 7.1 LLM evaluation: `run_eval.py`

Asks every question in `questions.json` and scores the answers (Answer Relevancy, plus Correctness where an expected answer is given).

```bash
python run_eval.py
```

To use your own questions, edit `questions.json`:
```json
[
  { "input": "Your question", "expected_output": "The ideal answer (optional)" }
]
```

The same checks also run as a test suite, which is useful in CI:
```bash
deepeval test run tests/test_llm_quality.py
```

### 7.2 RAG evaluation: generate → evaluate → compare

Run these in order. Together they show the complete RAG evaluation workflow:

**Step 1: generate test questions from the documents automatically**
```bash
python generate_test_questions.py
```
DeepEval reads each document chunk and writes realistic questions with correct answers, saved to `generated_questions.json`.

**Step 2: evaluate the whole test set**
```bash
python rag_eval.py                                       # uses the hand-written rag_questions.json
python rag_eval.py --questions generated_questions.json  # uses the generated questions
```
Prints the average score and pass rate per metric, plus the 3 weakest questions. Saves `reports/rag_eval_report.csv`.

**Step 3: compare settings to find the best one**
```bash
python compare_rag_settings.py --limit 4
```
Tries every combination of chunk size (400, 800, 1200) and top-k (2, 4), ranks them in a leaderboard and explains the winner in plain English. Saves `reports/rag_settings_comparison.csv`.

Useful options:
```bash
python rag_eval.py --top-k 4 --chunk-size 600
python compare_rag_settings.py --chunk-sizes 400 800 --top-ks 2 3 5 --limit 4
python generate_test_questions.py --per-chunk 3 --output my_questions.json
```

> `compare_rag_settings.py` runs the test set once per setting, so it makes the most OpenAI calls. Keep `--limit` small during demos.

---

## 8. Metrics explained

All scores range from **0 to 1**; higher is better. A metric **passes** when its score is at or above the pass threshold (default 0.7). Every score is decided by an LLM judge (the *scoring model*), which also explains its reasoning.

### Answer metrics (both modes)

| Metric | Question it answers |
|---|---|
| **Answer Relevancy** | Does the answer address the question? |
| **Correctness** | Is it factually correct? (compared to the expected answer, if given) |
| **Coverage** | Does it cover every part of the question, or every key point of the expected answer? |
| **Faithfulness** | Is it free of made-up claims? (checked against the context, if given) |
| **Structure Quality** | Is it well organised and formatted? |
| **Prompt Adherence** | Does it follow the system prompt and the user's instructions? |
| **Traceability** | Can each claim be traced to the context, or verified? |
| **Redundancy** | Is it free of repetition and filler? (higher = less redundant) |
| **Helpfulness** | Is it useful and actionable? |
| **Clarity** | Is it easy to understand? |

### Retrieval metrics (RAG mode only)

| Metric | Question it answers |
|---|---|
| **Contextual Relevancy** | Are the retrieved chunks relevant to the question? |
| **Contextual Precision** | Are the most useful chunks ranked at the top? *(needs expected answer)* |
| **Contextual Recall** | Was everything needed for the answer actually retrieved? *(needs expected answer)* |

> Because the judge is an LLM, a score can vary slightly between runs of the same question.

---

## 9. Project structure

```
deepeval-openai-eval/
├── app.py                       # Streamlit app: LLM Chat + RAG modes with live scores
├── rag.py                       # RAG pipeline: load → chunk → embed → retrieve → answer
├── sample_docs/                 # Sample company documents used by RAG mode
│   ├── leave_policy.md
│   ├── travel_policy.md
│   └── work_from_home_faq.md
│
├── run_eval.py                  # LLM batch evaluation of questions.json
├── llm_client.py                # Sends a question to OpenAI, returns the answer
├── eval_core.py                 # Shared metric definitions for run_eval.py and the tests
├── questions.json               # LLM test questions
├── tests/test_llm_quality.py    # Same checks as a pytest / deepeval test suite
│
├── generate_test_questions.py   # RAG: auto-generate test questions from documents
├── rag_eval.py                  # RAG: batch-evaluate a test set, save a report
├── compare_rag_settings.py      # RAG: compare chunk size / top-k, leaderboard
├── rag_questions.json           # RAG: hand-written test questions
│
├── requirements.txt             # Python dependencies
├── .env.example                 # Settings template (copy to .env)
└── reports/                     # CSV reports (created when scripts run)
```

### How RAG works in this project

```
Documents ─► split into chunks ─► OpenAI embeddings ─► in-memory vector index
                                                             │
Question  ─► embedding ─► top-k most similar chunks ◄────────┘
                               │
               OpenAI answers using ONLY those chunks
                               │
       DeepEval: retrieval metrics + answer metrics
```

The vector index is kept in memory, which is enough for demos and small document sets. For production, replace `VectorIndex` in `rag.py` with a vector database such as FAISS, Chroma, pgvector or Pinecone.

---

## 10. Configuration

All settings live in `.env`. Only `OPENAI_API_KEY` is required.

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** Your OpenAI key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model that answers the questions |
| `DEEPEVAL_JUDGE_MODEL` | `gpt-4o-mini` | Model that scores the answers |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model that embeds document chunks (RAG) |
| `PASS_THRESHOLD` | `0.7` | Minimum score for a Pass |
| `SYSTEM_PROMPT` | *"You are a helpful assistant..."* | Default system prompt in Chat mode |
| `CONFIDENT_API_KEY` | — | Optional. Only for the Confident AI dashboard |

In the Streamlit app, the chat model, scoring model, threshold and system prompt can also be changed from the sidebar without editing `.env`.

> **Tip:** use a stronger scoring model (for example `gpt-4o`) for stricter, more consistent scores. It costs more per question.

---

## 11. Optional: Confident AI dashboard

Everything above works **locally without any extra account**. If you also want the command-line runs (`run_eval.py`, `deepeval test run`) stored in a hosted dashboard:

1. Sign up at https://app.confident-ai.com and create a project.
2. Copy the project's **API key** from its settings. This is not the project ID.
3. Log in from the project folder:
   ```bash
   deepeval login --api-key YOUR_CONFIDENT_API_KEY
   ```
4. Run `python run_eval.py`. The results appear under **Test Runs**.

To switch accounts later, run `deepeval logout` and then log in again.

---

## 12. Troubleshooting

| Problem | Fix |
|---|---|
| `python` is not recognised | Reinstall Python with **"Add Python to PATH"** ticked, or use `py` instead of `python` on Windows |
| *"running scripts is disabled on this system"* when activating | Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, then activate again |
| `ModuleNotFoundError: No module named ...` | The virtual environment isn't active. Activate it (step 3) and run `pip install -r requirements.txt` |
| App shows *"OPENAI_API_KEY is missing"* | Check that `.env` exists in the project folder (not only `.env.example`) and contains your key, then restart the app |
| `openai.AuthenticationError` / 401 | The API key is wrong or revoked. Create a new one |
| `openai.RateLimitError` / 429 / *insufficient_quota* | Add billing or credit to your OpenAI account, or wait and retry. You can also turn off some metrics in the sidebar |
| Windows: *"Shared locks on Windows require the win32 extra (pywin32)"* or `'NoneType' object has no attribute 'test_cases_lookup_map'` | Run `pip install "portalocker[win32]"` |
| A metric shows `—` with *Error:* in "Why these scores?" | That one metric failed (often a temporary API error). Ask again; the other metrics are not affected |
| Contextual Precision / Recall show *Skipped* | Fill **Expected answer** in the sidebar before asking |
| `deepeval login`: *"No such option: --confident-api-key"* | Use `--api-key` instead |
| Confident AI shows *"Access Denied"* | You're signed in to a different account than the one the API key belongs to. Use the key from **your** project |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` |

---

## Quick start (summary)

```bash
git clone <your-repo-url>
cd deepeval-openai-eval
python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env              # macOS/Linux: cp .env.example .env
# put your OPENAI_API_KEY in .env
streamlit run app.py
```