# PensionBox HR Bot — v1

RAG-based internal HR chatbot. Employees ask questions in Slack, the bot searches your HR policy documents and answers using Claude.

---

## Project Structure

```
pensionbox-hr-bot/
├── docs/                  ← PUT YOUR HR POLICY FILES HERE (PDF, DOCX, TXT)
├── data/                  ← Auto-generated (FAISS index + metadata)
├── ingest/
│   └── build_index.py     ← Run this to build / rebuild the index
├── api/
│   ├── main.py            ← FastAPI server
│   └── rag.py             ← RAG engine (search + Claude)
├── slack_bot/
│   └── app.py             ← Slack Bolt bot
├── .env.example           ← Copy to .env and fill in your keys
└── requirements.txt
```

---

## Step 1 — Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Step 2 — Set up environment

```bash
cp .env.example .env
```

Open `.env` and fill in:
- `ANTHROPIC_API_KEY` — get from https://console.anthropic.com
- Slack keys (see Step 4 below)

---

## Step 3 — Add HR documents and build the index

1. Drop all your HR policy files into the `docs/` folder.
   Supported formats: `.pdf`, `.docx`, `.txt`

2. Build the FAISS index:
```bash
python -m ingest.build_index
```

3. Whenever you update or add new docs, drop the files in `docs/` and re-run the same command.

---

## Step 4 — Create the Slack App

### 4.1 Create a new app
1. Go to https://api.slack.com/apps → Create New App → From scratch
2. Name it `PensionBox HR Bot`, pick your workspace

### 4.2 Enable Socket Mode
1. Go to Socket Mode (left sidebar) → Enable it
2. Generate an App-Level Token with scope `connections:write`
3. Copy this token → paste as `SLACK_APP_TOKEN` in your `.env`

### 4.3 Add Bot Token Scopes
Go to OAuth & Permissions → Bot Token Scopes, add:
```
app_mentions:read
channels:history
chat:write
im:history
im:read
im:write
users:read
```

### 4.4 Enable Event Subscriptions
Go to Event Subscriptions → Enable → Subscribe to Bot Events:
```
app_mention
message.im
```

### 4.5 Install the app
Go to OAuth & Permissions → Install to Workspace → Authorize
Copy the Bot User OAuth Token (xoxb-...) → SLACK_BOT_TOKEN in .env
Copy the Signing Secret from Basic Information → SLACK_SIGNING_SECRET in .env

### 4.6 Add bot to a channel
In Slack: open the HR channel → /invite @PensionBox HR Bot

---

## Step 5 — Run the API server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Check: curl http://localhost:8000/health

---

## Step 6 — Run the Slack bot

```bash
python slack_bot/app.py
```

---

## Updating HR documents

1. Add/replace files in the docs/ folder on the server
2. Run: python -m ingest.build_index
   OR call: curl -X POST http://localhost:8000/reindex
3. No restart needed — API reloads the index automatically

---

## Architecture

Employee in Slack → Slack Bot → POST /ask → FastAPI
→ Embed question → FAISS top-5 search
→ Chunks + question → gpt-4o-mini
→ Answer + sources → Slack reply
