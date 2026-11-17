# HR & IT FAQ Chatbot (RAG with OpenAI)

A chatbot that answers HR and IT questions **only from your own documents**, cites its sources, and says so when the documents don't contain the answer.

It uses **Retrieval-Augmented Generation (RAG)**:

1. **Retrieval:** your question is turned into a vector (an *embedding*), and the most similar passages are looked up in a local ChromaDB database.
2. **Augmented generation:** those passages are sent to an OpenAI chat model with instructions to answer only from them and to cite them as `[1]`, `[2]`, ...

```
INDEXING (automatic when the app opens, or: python scripts/ingest.py)

  documents/{hr,it}/*.{md,txt,pdf,docx,html}
      -> Load -> Clean -> Chunk (structure-aware, <= 350 tokens) -> Embed (OpenAI) -> ChromaDB (index/)

QUESTION

  Question -> Embed (OpenAI) -> ChromaDB similarity search -> top 6 chunks
           -> OpenAI chat model (answer only from these chunks, cite [n])
           -> streamed answer + "Sources" panel
```

The full design, its reasoning and its open questions are in [DESIGN.md](DESIGN.md). This build follows the **2-day scope** in section 14, with **OpenAI** as the LLM (open question Q1).

---

## Contents

1. [Get an OpenAI API key](#1-get-an-openai-api-key)
2. [Install and run](#2-install-and-run)
3. [Using the chatbot](#3-using-the-chatbot)
4. [Configuration](#4-configuration)
5. [Evaluation](#5-evaluation)
6. [Costs](#6-costs)
7. [Project structure and code tour](#7-project-structure-and-code-tour)
8. [Troubleshooting](#8-troubleshooting)
9. [Security notes](#9-security-notes)
10. [Differences from DESIGN.md](#10-differences-from-designmd)

---

## 1. Get an OpenAI API key

> **Important:** A ChatGPT subscription (Plus, Pro, Team) does **not** include API access. The API is billed separately, through the OpenAI Platform, with prepaid credits.

### 1.1 Create an OpenAI Platform account

1. Go to **https://platform.openai.com/signup**.
2. Sign up with an email address, or continue with Google, Microsoft or Apple. If you already use ChatGPT, you can sign in with the same login.
3. Verify your email address with the link OpenAI sends you.
4. Enter your name and, if asked, your organization name. Use your own name for personal or student projects.
5. If asked, verify a phone number. OpenAI sends a code by SMS.

### 1.2 Add credit (billing)

API calls fail with error `429 insufficient_quota` until the account has credit.

1. Open **https://platform.openai.com/settings/organization/billing**.
2. Click **Add payment details** and add a card.
3. Buy prepaid credits. The minimum purchase is usually **$5**, which is far more than this project needs (see [Costs](#6-costs)).
4. Leave **auto-recharge off** for a demo, so spending can't exceed what you prepaid.
5. Optional: under **Settings > Limits** (https://platform.openai.com/settings/organization/limits), set a monthly budget and an email alert.

### 1.3 Create the API key

1. Open **https://platform.openai.com/api-keys**. You can also find it under **Settings > API keys** in the dashboard.
2. Click **Create new secret key**.
3. Fill in the form:
   - **Owned by:** *You*.
   - **Name:** for example `rag-chatbot-laptop`.
   - **Project:** *Default project* is fine. Projects let you separate budgets and keys.
   - **Permissions:** *All* is simplest. For least privilege, choose *Restricted* and allow **Model capabilities** (chat completions and embeddings) as *Write*.
4. Click **Create secret key** and **copy the key right away**. It starts with `sk-proj-...` and **is shown only once**. If you lose it, delete it and create a new one.

### 1.4 Put the key in the project

```powershell
cd D:\Projects\projects\rag_chatbot
copy .env.example .env
notepad .env
```

Replace the placeholder so that the line reads:

```
OPENAI_API_KEY=sk-proj-...your real key...
```

`.env` is listed in `.gitignore`, so it is never committed. Never paste the key into code, chats or screenshots. If a key leaks, delete it on the API keys page and create a new one.

### 1.5 Good to know

| Topic | Details |
|---|---|
| Models used | `text-embedding-3-small` for embeddings, and `gpt-4.1-mini` to write answers. Both can be changed in `config.yaml`. |
| Model list | https://platform.openai.com/docs/models shows which models exist and which your account can use. |
| Prices | https://openai.com/api/pricing. Always check here, because prices change. |
| Usage | https://platform.openai.com/usage shows the tokens and money spent per day and model. |
| Rate limits and tiers | New accounts start at a low usage tier with lower request limits. Tiers rise automatically as you spend. That is enough for this demo. See https://platform.openai.com/settings/organization/limits. |
| Organization verification | Some of the newest models, or features such as streaming on them, need *Verified Organization* status (Settings > General). The default model in this project does not. |
| Data use | By default, OpenAI does not use API inputs and outputs to train models. Only send documents that may leave your machine (DESIGN.md section 12). |

---

## 2. Install and run

### 2.1 Requirements

- Windows, macOS or Linux
- **Python 3.11 or newer** (tested with 3.14) from https://www.python.org/downloads/. On Windows, tick **"Add python.exe to PATH"** in the installer. Windows also provides the `py` launcher.
- An OpenAI API key with credit (section 1)

### 2.2 Set up (once)

```powershell
cd D:\Projects\projects\rag_chatbot

# Create an isolated Python environment in .venv
py -m venv .venv

# Install the libraries into it
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Add your API key (see 1.4)
copy .env.example .env
notepad .env
```

On macOS or Linux, use `python3 -m venv .venv` and then `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`.

> Optional: activate the environment with `.\.venv\Scripts\Activate.ps1` and then type just `python`. If PowerShell blocks the script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or keep using the full `.\.venv\Scripts\python.exe` path.

### 2.3 Build the index (optional)

The app indexes automatically when it opens. You can also index from the terminal and see the progress:

```powershell
.\.venv\Scripts\python.exe scripts\ingest.py
```

Expected output: `Done: 6 added, 0 updated, 0 removed, 0 unchanged, 0 skipped`.

### 2.4 Start the chat app

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Your browser opens **http://localhost:8501**. Stop the app with `Ctrl+C` in the terminal.

> Use `python -m streamlit`, not `streamlit.exe`. On this laptop, Windows Smart App Control blocks the `streamlit.exe` launcher that pip generates (section 8).

---

## 3. Using the chatbot

- **Ask a question** in the box at the bottom, for example *"How many days of annual leave do I get?"* or *"The VPN won't connect, what should I try?"*.
- The answer streams in, with citations like `[1]`.
- Open **Sources** under an answer to see the retrieved passages, their file and section, the similarity score, and which ones the answer cited (✅).
- If the documents don't cover the question, the bot says *"I couldn't find this in the documents"* and suggests contacting HR or IT.
- **Each question is answered on its own.** The bot doesn't remember earlier questions, so include all the details every time.

### Adding or changing documents

1. Put files into `documents/hr/` or `documents/it/`. Supported types are `.md`, `.txt`, `.pdf`, `.docx` and `.html`.
2. Refresh the browser page, or click **Re-index documents** in the sidebar.

Only new, changed and deleted files are processed, so the index is never re-embedded unnecessarily. Deleting a file removes its content from the bot at the next sync. Scanned PDFs (images without text) are skipped with a warning, because OCR is out of scope.

The 6 files in `documents/` are **fictional sample documents** for "Example Co.", so the demo works out of the box. Replace them with your real public or sample documents (DESIGN.md Q4). Then update `eval/golden.jsonl` to match.

The sidebar shows the number of indexed documents and chunks, and when the index was last updated.

---

## 4. Configuration

All settings are in [config.yaml](config.yaml):

| Setting | Default | Meaning |
|---|---|---|
| `documents_dir` | `documents` | Input folder |
| `index_dir` | `index` | Where ChromaDB and the manifest are stored |
| `embedding_model` | `text-embedding-3-small` | OpenAI embedding model. Changing it requires `--rebuild`. |
| `chat_model` | `gpt-4.1-mini` | OpenAI model that writes the answers |
| `temperature` | `0.1` | Low = consistent answers. Set it to `null` for reasoning models (such as the gpt-5 family), which reject this parameter. |
| `max_tokens` | `1024` | Maximum answer length |
| `chunk_size` / `chunk_overlap` | `350` / `50` | Maximum chunk size, and the trailing sentences repeated in the next chunk, in tokens |
| `min_chunk_tokens` | `100` | Smaller sections are merged with neighbours on the same topic |

Changing any of the three chunking settings requires `--rebuild`.
| `top_k` | `6` | Passages sent to the model per question |

### How documents are chunked

Chunk quality has the biggest effect on answer accuracy. A chunk that cuts a procedure in half, or a 15-token fragment with no context, is retrieved badly and misleads the model. [rag/chunking.py](rag/chunking.py) therefore follows the document's structure instead of cutting every N tokens:

| Rule | Why it helps accuracy |
|---|---|
| Split at headings, and start every chunk with its heading path (`Leave Policy > Sick Leave`) | Each chunk covers one topic, and the embedding and the model both know what it is about |
| Merge tiny sections on the same topic (below `min_chunk_tokens`) | *Parental Leave: Eligibility*, *Duration and Pay* and *How to Apply* become one chunk that answers the whole question |
| Never merge different topics | *Bereavement Leave* is never mixed into a parental leave chunk |
| Keep lists, tables and code blocks whole, with their lead-in line (`To reset your password:`) | Numbered steps are never split across chunks |
| Split oversized blocks at sentences (prose), items (lists) or rows (tables, repeating the header row) | No chunk ends mid-sentence, and table columns stay labelled |
| Overlap of up to 50 tokens of whole sentences between consecutive chunks of a long section | A fact on a chunk boundary can be found from either side |
| Ignore `#` lines inside code blocks | Code comments don't create false sections |

To see exactly how your documents are chunked (offline and free), run:

```powershell
.\.venv\Scripts\python.exe scripts\inspect_chunks.py                     # summary of all files
.\.venv\Scripts\python.exe scripts\inspect_chunks.py hr/leave-policy.md  # full chunk text of one file
```

For the sample documents this gives 26 chunks of 37–203 tokens, each covering one topic.

After changing the embedding model or the chunk settings, rebuild the index:

```powershell
.\.venv\Scripts\python.exe scripts\ingest.py --rebuild
```

The app refuses to answer if the index was built with different settings, because mixing the two would give wrong results.

---

## 5. Evaluation

`eval/golden.jsonl` contains 20 hand-written test questions for the sample documents:

| Category | Count | What it tests |
|---|---|---|
| Direct lookup | 8 | One section holds the answer |
| Paraphrased | 5 | Everyday wording that differs from the policy text |
| Multi-section | 2 | The answer needs two passages |
| Unanswerable | 5 | Questions where declining is the correct answer |

Run the evaluation:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py           # retrieval + answers
.\.venv\Scripts\python.exe scripts\evaluate.py --no-llm  # retrieval only (cheaper)
```

This writes **`eval/report.md`**, with:

- **Hit rate@6:** the share of questions where an expected section was retrieved.
- **MRR:** how high the first correct passage ranks. 1.0 means it is always first.
- **Declining:** whether unanswerable questions were declined and answerable ones were not.
- One entry per question showing the answer, the expected answer and the retrieved passages, with an empty **Grade** field. Grade each answer by hand as `correct`, `partial` or `incorrect`, then fill in the correctness row.

Raw results are also saved to `eval/results/run-<timestamp>.jsonl`, so runs can be compared.

### Automated tests

The tests run offline (no API key, no cost). They check the chunking rules above, and index syncing with fake embeddings:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

---

## 6. Costs

These are rough estimates at list prices. Check https://openai.com/api/pricing for current rates.

| Item | Assumption | Approx. cost |
|---|---|---|
| Indexing the 6 sample documents | ~3,000 tokens with `text-embedding-3-small` (~$0.02 per million tokens) | < $0.001 |
| One question with `gpt-4.1-mini` | ~2,500 input + ~300 output tokens | ~ $0.0015 |
| One full evaluation run | 20 questions | ~ $0.03 |

The minimum $5 credit covers thousands of questions. Unchanged documents are never re-embedded.

---

## 7. Project structure and code tour

```
rag_chatbot/
├── README.md                  # this file
├── DESIGN.md                  # design document and decisions
├── requirements.txt           # Python libraries
├── .env.example               # template for your API key (copy to .env)
├── .gitignore                 # keeps .env, the index and the documents out of git
├── config.yaml                # all tunable settings
├── app.py                     # Streamlit chat UI
├── rag/                       # the RAG pipeline (used by the app and the scripts)
│   ├── config.py              # reads config.yaml and the API key
│   ├── loaders.py             # .md/.txt/.pdf/.docx/.html -> text with "#" headings
│   ├── chunking.py            # text -> ~350-token chunks with a heading prefix
│   ├── embeddings.py          # OpenAI client + text -> vector
│   ├── index.py               # syncs documents/ into ChromaDB (skips unchanged files)
│   ├── retrieval.py           # question -> most similar chunks
│   ├── generation.py          # prompt + OpenAI chat call + citation helpers
│   └── pipeline.py            # retrieve -> generate, in one call
├── scripts/
│   ├── ingest.py              # index from the command line (--rebuild)
│   ├── inspect_chunks.py      # shows how documents are chunked (offline)
│   └── evaluate.py            # runs the test questions, writes eval/report.md
├── tests/
│   ├── test_chunking.py       # chunking rules
│   └── test_index.py          # sync + retrieval with fake embeddings
├── documents/                 # INPUT: your documents (hr/, it/)
├── eval/
│   ├── golden.jsonl           # test questions with expected answers and sources
│   └── report.md              # generated evaluation report
└── index/                     # GENERATED: ChromaDB database + manifest.json
```

Every source file has comments on its important lines explaining what they do and why. Read the files in this order:

1. **`rag/loaders.py`:** Each format is converted to plain text. Word headings and HTML `<h1>`–`<h6>` tags become Markdown `#` lines, so the document structure survives. Page numbers and extra blank lines are removed.
2. **`rag/chunking.py`:** The structure-aware chunker described in [How documents are chunked](#how-documents-are-chunked). It splits the text into sections, merges tiny sections on the same topic, and breaks each section into blocks (paragraph, list, table, code). Blocks are packed into chunks of at most 350 tokens, splitting only at sentence, item or row boundaries. Tokens are counted with `tiktoken`, using the same tokenizer as the embedding model.
3. **`rag/embeddings.py`:** `client.embeddings.create(...)` turns text into a list of 1,536 numbers. Texts with similar meanings get vectors that point in similar directions. Requests are batched (100 texts per call) to save time.
4. **`rag/index.py`:** `sync_index()` compares a SHA-256 hash of each file with `index/manifest.json`. New and changed files are chunked, embedded and stored. Changed and deleted files have their old chunks removed first, so outdated policy text can't be retrieved. The manifest also records the embedding model and chunk settings, so a mismatch is detected.
5. **`rag/retrieval.py`:** The question is embedded with the same model, and ChromaDB returns the `top_k` chunks with the smallest **cosine distance**. The code converts that to a similarity score, where 1.0 means identical meaning.
6. **`rag/generation.py`:** The prompt numbers each chunk `[1] ... [6]`. The **system prompt** tells the model to answer only from them, to cite them, and to start with a fixed phrase when the answer isn't there. The answer is **streamed** (`stream=True`) so text appears as it's generated. API errors are turned into readable messages.
7. **`app.py`:** Streamlit re-runs the whole script on every click. `st.session_state` keeps the chat history and a "synced" flag, so indexing runs once per browser session rather than once per message.

---

## 8. Troubleshooting

| Problem | Cause and fix |
|---|---|
| `No OpenAI API key found` | `.env` is missing or still holds the placeholder. See section 1.4, then restart the app. |
| `OpenAI rejected the API key` (401) | The key was copied incompletely, or it was deleted. Create a new key. |
| `rate limit or quota reached` (429) | No credit left (`insufficient_quota`), or too many requests too quickly. Add credit on the Billing page, or wait a minute. |
| `model ... was not found` (404) | Your account can't use the model in `config.yaml`. Pick one listed on https://platform.openai.com/docs/models. |
| `Unsupported parameter: 'temperature'` | The chosen model is a reasoning model. Set `temperature: null` in `config.yaml`. |
| `The embedding model or chunk settings ... changed` | Run `scripts\ingest.py --rebuild`. |
| `No documents found` | Put files in `documents\hr` or `documents\it`. |
| `DLL load failed ... An Application Control policy has blocked this file` | **Windows Smart App Control** blocks some recently released compiled packages. `requirements.txt` pins versions that load on this laptop (`pydantic 2.12`, `pandas 2.3`). If a new error names another package, install an earlier release of it, for example `pip install "package==x.y.*"`. Don't turn Smart App Control off just for this. |
| `streamlit.exe` is blocked | Start with `.\.venv\Scripts\python.exe -m streamlit run app.py`. |
| `py` / `python` not found | Install Python from python.org and tick "Add python.exe to PATH". |
| Answers seem to ignore a new file | Refresh the page or click **Re-index documents**, and check the **Last sync** box in the sidebar for skipped files. |

---

## 9. Security notes

- Questions and the retrieved passages are sent to the OpenAI API. Document **embeddings are also created by OpenAI**, so the full text of every document is sent once while indexing. Index only documents that may leave the machine.
- There is no access control, so index only material that every user may read.
- The prompt tells the model to treat documents as reference material and not to follow instructions inside them. This limits prompt injection but doesn't rule it out. The bot has no tools, so the worst case is a wrong answer.
- Keep `.env` private. The `documents/`, `index/` and `eval/results/` folders are excluded from git.

---

## 10. Differences from DESIGN.md

| DESIGN.md | This build | Reason |
|---|---|---|
| LLM open (proposal: Claude) | OpenAI `gpt-4.1-mini`, configurable | OpenAI was requested, which settles Q1 |
| Local embeddings (`bge-small` via `sentence-transformers`) | OpenAI `text-embedding-3-small` | One provider and a much lighter install (no PyTorch). It costs a fraction of a cent, but document text is sent to OpenAI (section 9). |
| `config.yaml` | Same, plus `chat_model`, `temperature` and `max_tokens` | Settings for the OpenAI call |
| Native citations API | Numbered `[n]` markers in the text, with sources listed under each answer | The OpenAI chat API has no document-citations feature |
| Deferred items (BM25, reranker, relevance gate, feedback buttons, lock) | Still deferred | 2-day scope. The **Re-index documents** button and the offline tests were cheap to add, so they are included. |
