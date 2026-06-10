"""
watcher.py

HR Policy Change Watcher Agent
================================
Watches the /docs folder for any file additions, modifications, or deletions.
When a change is detected:
  1. Reads the changed file (PDF, DOCX, TXT)
  2. Diffs against the previous known version (stored in .cache/)
  3. Uses GPT-4o to generate a plain-English bulletin summarising what changed
  4. Posts the bulletin to a designated Slack channel
  5. Triggers a FAISS reindex so the Q&A bot reflects the new content

Run with:
    python watcher.py

Environment variables required (same .env as the rest of the bot):
    SLACK_BOT_TOKEN        — bot token (xoxb-...)
    SLACK_ANNOUNCEMENT_CHANNEL — channel ID to post bulletins (e.g. C12345678)
    OPENAI_API_KEY
    DOCS_DIR               — path to watch (default: ./docs)
    API_BASE_URL           — FastAPI server for reindex trigger (default: http://localhost:8000)
"""

import os
import time
import json
import hashlib
import logging
import difflib
import httpx
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# ── document loaders (reuse from build_index) ──────────────────────────────
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

# ── config ─────────────────────────────────────────────────────────────────
DOCS_DIR         = Path(os.getenv("DOCS_DIR", "./docs"))
CACHE_DIR        = Path(os.getenv("CACHE_DIR", "./data/.watcher_cache"))
POLL_INTERVAL    = int(os.getenv("POLL_INTERVAL_SECONDS", "30"))   # seconds between scans
API_BASE_URL     = os.getenv("API_BASE_URL", "http://localhost:8000")
ANNOUNCEMENT_CH  = os.getenv("SLACK_ANNOUNCEMENT_CHANNEL", "")     # e.g. C12345678
SLACK_TOKEN      = os.environ["SLACK_BOT_TOKEN"]
SUPPORTED_EXTS   = {".pdf", ".docx", ".txt"}

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# 1. FILE READING
# ═══════════════════════════════════════════════════════════════════════════

def extract_text(file_path: Path) -> str:
    """Extract raw text from PDF, DOCX, or TXT."""
    ext = file_path.suffix.lower()
    try:
        if ext == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif ext == ".docx":
            loader = Docx2txtLoader(str(file_path))
        elif ext == ".txt":
            loader = TextLoader(str(file_path))
        else:
            return ""
        docs = loader.load()
        return "\n\n".join(d.page_content for d in docs)
    except Exception as e:
        log.warning(f"Could not extract text from {file_path.name}: {e}")
        return ""


# ═══════════════════════════════════════════════════════════════════════════
# 2. CACHE — track file hashes and text snapshots
# ═══════════════════════════════════════════════════════════════════════════

def file_hash(file_path: Path) -> str:
    """MD5 of file bytes — fast change detection."""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cache() -> dict:
    cache_file = CACHE_DIR / "state.json"
    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)
    return {}   # {filename: {hash, text_snapshot}}


def save_cache(state: dict):
    with open(CACHE_DIR / "state.json", "w") as f:
        json.dump(state, f, indent=2)


def load_text_snapshot(filename: str) -> str:
    snap = CACHE_DIR / f"{filename}.txt"
    return snap.read_text(encoding="utf-8") if snap.exists() else ""


def save_text_snapshot(filename: str, text: str):
    snap = CACHE_DIR / f"{filename}.txt"
    snap.write_text(text, encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# 3. DIFF — find meaningful differences in text
# ═══════════════════════════════════════════════════════════════════════════

def compute_diff(old_text: str, new_text: str) -> str:
    """Return a human-readable unified diff (max 120 lines to keep prompt sane)."""
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff = list(difflib.unified_diff(
        old_lines, new_lines,
        fromfile="previous version",
        tofile="updated version",
        n=3,
    ))
    if not diff:
        return ""
    diff_text = "".join(diff[:120])
    if len(diff) > 120:
        diff_text += f"\n... (diff truncated, {len(diff) - 120} more lines)"
    return diff_text


# ═══════════════════════════════════════════════════════════════════════════
# 4. AI BULLETIN GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_bulletin(event: str, filename: str, new_text: str, diff_text: str) -> dict:
    """
    Ask GPT-4o to produce a structured bulletin as a JSON dict.
    Returns:
      {
        "summary": str,                  — one-line overview
        "modified_policies": [           — existing policies that changed
          {"name": str, "change": str},  — one line each
          ...
        ],
        "new_policies": [                — brand-new policies introduced
          {
            "name": str,
            "overview": str,
            "clauses": [
              {"clause": str, "detail": str, "subclauses": [str, ...]},
              ...
            ]
          },
          ...
        ],
        "action_required": str,          — what employees need to do (or "None")
        "effective_date": str,           — date or "Effective immediately"
      }
    event: "added" | "modified" | "deleted"
    """

    if event == "deleted":
        return {
            "summary": f"The document *{filename}* has been retired and is no longer in effect.",
            "modified_policies": [],
            "new_policies": [],
            "action_required": "No action required. Contact HR if you have questions.",
            "effective_date": "Immediately",
        }

    if event == "added":
        prompt = f"""A brand-new HR policy document has been published: {filename}

Full document content:
---
{new_text[:6000]}
---

Analyse this document carefully and return a JSON object with EXACTLY this structure:
{{
  "summary": "<one sentence: what this document covers overall>",
  "modified_policies": [],
  "new_policies": [
    {{
      "name": "<policy name / section title>",
      "overview": "<2-3 sentence description of what this policy is about>",
      "clauses": [
        {{
          "clause": "<clause title or number>",
          "detail": "<precise, complete explanation of this clause>",
          "subclauses": ["<subclause 1>", "<subclause 2>"]
        }}
      ]
    }}
  ],
  "action_required": "<what employees must do, or 'No action required'>",
  "effective_date": "<date from document, or 'Effective immediately'>"
}}

Rules:
- Extract EVERY policy section as a separate entry in new_policies
- For each policy, extract ALL clauses and subclauses precisely and completely — do not summarise or skip any
- Use the exact clause numbers/titles from the document
- subclauses array can be empty [] if there are none
- Return ONLY the JSON object, no markdown fences, no preamble
"""

    else:  # modified
        prompt = f"""An existing HR policy document has been updated: {filename}

What changed (unified diff):
---
{diff_text[:4000]}
---

Updated full document for context:
---
{new_text[:4000]}
---

Analyse the changes carefully and return a JSON object with EXACTLY this structure:
{{
  "summary": "<one sentence: overall nature of this update>",
  "modified_policies": [
    {{
      "name": "<name of the existing policy/section that changed>",
      "change": "<one precise sentence describing exactly what changed in this policy>"
    }}
  ],
  "new_policies": [
    {{
      "name": "<name of any brand-new policy/section introduced in this update>",
      "overview": "<2-3 sentence description of what this new policy is about>",
      "clauses": [
        {{
          "clause": "<clause title or number>",
          "detail": "<precise, complete explanation of this clause>",
          "subclauses": ["<subclause 1>", "<subclause 2>"]
        }}
      ]
    }}
  ],
  "action_required": "<what employees must do, or 'No action required'>",
  "effective_date": "<date from document, or 'Effective immediately'>"
}}

Rules:
- modified_policies = policies that ALREADY EXISTED and were changed — one line each, precise
- new_policies = policies that are BRAND NEW in this update — extract ALL clauses and subclauses completely
- If there are no new policies, new_policies = []
- If there are no modified policies, modified_policies = []
- Do not skip or summarise any clause for new policies
- Return ONLY the JSON object, no markdown fences, no preamble
"""

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are PensionBox's internal HR policy analyst. "
                    "You extract structured information from HR policy documents with complete precision. "
                    "You always return valid JSON only — no markdown, no extra text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=2000,
        temperature=0.1,
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if model adds them despite instructions
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning("GPT returned invalid JSON — falling back to plain text bulletin.")
        return {
            "summary": raw[:500],
            "modified_policies": [],
            "new_policies": [],
            "action_required": "Contact HR if you have questions.",
            "effective_date": "Effective immediately",
        }


# ═══════════════════════════════════════════════════════════════════════════
# 5. SLACK POSTING
# ═══════════════════════════════════════════════════════════════════════════

def build_slack_blocks(bulletin: dict, filename: str, event: str) -> list:
    """Convert the structured bulletin dict into rich Slack Block Kit blocks."""
    event_emoji = {"added": "🆕", "modified": "📝", "deleted": "🗑️"}.get(event, "📋")
    event_label = {"added": "New Policy Published", "modified": "Policy Updated", "deleted": "Policy Retired"}.get(event, "Policy Change")
    timestamp   = datetime.now().strftime("%d %b %Y, %I:%M %p")

    blocks = []

    # ── Header ──────────────────────────────────────────────────────────────
    blocks.append({
        "type": "header",
        "text": {"type": "plain_text", "text": f"{event_emoji} {event_label}: {filename}"},
    })
    blocks.append({"type": "divider"})

    # ── Summary ─────────────────────────────────────────────────────────────
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": f"*📌 Summary*\n{bulletin.get('summary', '')}"},
    })

    # ── Effective date + action required ────────────────────────────────────
    eff   = bulletin.get("effective_date", "Effective immediately")
    action = bulletin.get("action_required", "No action required.")
    blocks.append({
        "type": "section",
        "fields": [
            {"type": "mrkdwn", "text": f"*📅 Effective*\n{eff}"},
            {"type": "mrkdwn", "text": f"*⚡ Action required*\n{action}"},
        ],
    })
    blocks.append({"type": "divider"})

    # ── Modified policies (existing ones that changed) ───────────────────────
    modified = bulletin.get("modified_policies", [])
    if modified:
        lines = [f"*🔄 Changes to existing policies*"]
        for mp in modified:
            name   = mp.get("name", "")
            change = mp.get("change", "")
            lines.append(f"• *{name}* — {change}")
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "\n".join(lines)},
        })
        blocks.append({"type": "divider"})

    # ── New policies (brand new, with full clauses) ──────────────────────────
    new_policies = bulletin.get("new_policies", [])
    if new_policies:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*🆕 New policies introduced*"},
        })

        for policy in new_policies:
            pname    = policy.get("name", "Unnamed Policy")
            overview = policy.get("overview", "")
            clauses  = policy.get("clauses", [])

            # Policy header + overview
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*{pname}*\n_{overview}_"},
            })

            # Each clause
            for c in clauses:
                clause_title = c.get("clause", "")
                detail       = c.get("detail", "")
                subclauses   = c.get("subclauses", [])

                clause_text = f"*{clause_title}*\n{detail}"
                if subclauses:
                    sub_lines = "\n".join(f"  ◦ {s}" for s in subclauses)
                    clause_text += f"\n{sub_lines}"

                # Slack section blocks max 3000 chars — chunk if needed
                for i in range(0, len(clause_text), 2900):
                    blocks.append({
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": clause_text[i:i+2900]},
                    })

            blocks.append({"type": "divider"})

    # ── Footer ───────────────────────────────────────────────────────────────
    blocks.append({
        "type": "context",
        "elements": [{
            "type": "mrkdwn",
            "text": f"_Automatically detected by PensionBox HR Bot · {timestamp} · Contact HR for questions_",
        }],
    })

    # Slack allows max 50 blocks per message — trim if needed
    if len(blocks) > 50:
        blocks = blocks[:49]
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": "_Some sections truncated due to length. See the full document in /docs._"}],
        })

    return blocks


def post_to_slack(bulletin: dict, filename: str, event: str):
    """Post the structured bulletin to the announcement channel."""
    if not ANNOUNCEMENT_CH:
        log.warning("SLACK_ANNOUNCEMENT_CHANNEL not set — printing bulletin instead:\n" + json.dumps(bulletin, indent=2))
        return

    blocks = build_slack_blocks(bulletin, filename, event)

    with httpx.Client(timeout=15.0) as client:
        resp = client.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {SLACK_TOKEN}"},
            json={
                "channel": ANNOUNCEMENT_CH,
                "blocks": blocks,
                "text": f"HR Policy {event}: {filename}",
            },
        )
        data = resp.json()
        if data.get("ok"):
            log.info(f"Bulletin posted to Slack for: {filename}")
        else:
            log.error(f"Slack post failed: {data.get('error')} — {data}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. REINDEX TRIGGER
# ═══════════════════════════════════════════════════════════════════════════

def trigger_reindex():
    """Tell the FastAPI server to rebuild the FAISS index."""
    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(f"{API_BASE_URL}/reindex")
            if resp.status_code == 200:
                log.info("FAISS reindex completed successfully.")
            else:
                log.warning(f"Reindex returned {resp.status_code}: {resp.text}")
    except Exception as e:
        log.warning(f"Could not trigger reindex (is the API running?): {e}")


# ═══════════════════════════════════════════════════════════════════════════
# 7. MAIN WATCHER LOOP
# ═══════════════════════════════════════════════════════════════════════════

def scan_docs(state: dict) -> list[dict]:
    """
    Scan DOCS_DIR, compare against cached state.
    Returns a list of change events: [{event, path, filename}, ...]
    """
    events = []
    current_files = {}

    if not DOCS_DIR.exists():
        log.warning(f"Docs dir {DOCS_DIR} does not exist yet.")
        return events

    # Detect added / modified
    for file_path in DOCS_DIR.rglob("*"):
        if file_path.is_dir():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTS:
            continue

        fname = file_path.name
        fhash = file_hash(file_path)
        current_files[fname] = fhash

        if fname not in state:
            events.append({"event": "added", "path": file_path, "filename": fname})
        elif state[fname]["hash"] != fhash:
            events.append({"event": "modified", "path": file_path, "filename": fname})

    # Detect deleted
    for fname in list(state.keys()):
        if fname not in current_files:
            events.append({"event": "deleted", "path": None, "filename": fname})

    return events, current_files


def handle_event(ev: dict, state: dict):
    """Process a single change event end-to-end."""
    event    = ev["event"]
    filename = ev["filename"]
    path     = ev["path"]

    log.info(f"Change detected — {event.upper()}: {filename}")

    # Extract new text
    new_text = extract_text(path) if path else ""

    # Get old text from snapshot for diff
    old_text = load_text_snapshot(filename) if event == "modified" else ""
    diff_text = compute_diff(old_text, new_text) if event == "modified" else ""

    # Generate AI bulletin
    log.info(f"Generating bulletin for {filename}...")
    bulletin = generate_bulletin(event, filename, new_text, diff_text)
    log.info(f"Bulletin:\n{bulletin}\n")

    # Post to Slack
    post_to_slack(bulletin, filename, event)

    # Update cache
    if event == "deleted":
        state.pop(filename, None)
        snap = CACHE_DIR / f"{filename}.txt"
        if snap.exists():
            snap.unlink()
    else:
        fhash = file_hash(path)
        state[filename] = {"hash": fhash, "last_seen": datetime.now().isoformat()}
        save_text_snapshot(filename, new_text)


def run():
    log.info(f"PensionBox HR Watcher Agent started.")
    log.info(f"Watching: {DOCS_DIR.resolve()}")
    log.info(f"Poll interval: {POLL_INTERVAL}s")
    log.info(f"Announcement channel: {ANNOUNCEMENT_CH or '(not set)'}")

    state = load_cache()

    # On first run, silently snapshot all existing files (no announcements)
    if not state:
        log.info("First run — snapshotting existing docs (no announcements for current files).")
        for file_path in DOCS_DIR.rglob("*"):
            if file_path.is_dir() or file_path.suffix.lower() not in SUPPORTED_EXTS:
                continue
            fname = file_path.name
            fhash = file_hash(file_path)
            state[fname] = {"hash": fhash, "last_seen": datetime.now().isoformat()}
            save_text_snapshot(fname, extract_text(file_path))
        save_cache(state)
        log.info(f"Snapshotted {len(state)} existing file(s). Now watching for changes...")

    while True:
        try:
            events, current_files = scan_docs(state)

            if events:
                log.info(f"{len(events)} change(s) detected.")
                for ev in events:
                    handle_event(ev, state)
                save_cache(state)
                trigger_reindex()
            else:
                log.debug("No changes.")

        except Exception as e:
            log.error(f"Watcher loop error: {e}", exc_info=True)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()