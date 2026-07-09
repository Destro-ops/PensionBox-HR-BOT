"""
watcher.py

HR Policy Change Watcher Agent
================================
Watches the /docs folder for any file additions, modifications, or deletions.
When a change is detected:
  1. Reads the changed file (PDF, DOCX, TXT)
  2. Diffs against the previous known version (stored in .cache/)
  3. Uses GPT-4o to generate a clause-level bulletin summarising what changed,
     with before/after per clause, new clauses, removed clauses, and key positives
  4. Posts a two-column Slack bulletin to the announcement channel
  5. Triggers a FAISS/Qdrant reindex so the Q&A bot reflects the new content

Run with:
    python watcher.py

Environment variables required (same .env as the rest of the bot):
    SLACK_BOT_TOKEN              — bot token (xoxb-...)
    SLACK_ANNOUNCEMENT_CHANNEL   — channel ID to post bulletins (e.g. C12345678)
    OPENAI_API_KEY
    DOCS_DIR                     — path to watch (default: ./docs)
    API_BASE_URL                 — FastAPI server for reindex trigger (default: http://localhost:8000)
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

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

# ── config ─────────────────────────────────────────────────────────────────
DOCS_DIR         = Path(os.getenv("DOCS_DIR", "./docs"))
CACHE_DIR        = Path(os.getenv("CACHE_DIR", "./data/.watcher_cache"))
POLL_INTERVAL    = int(os.getenv("POLL_INTERVAL_SECONDS", "30"))
API_BASE_URL     = os.getenv("API_BASE_URL", "http://localhost:8000")
ANNOUNCEMENT_CH  = os.getenv("SLACK_ANNOUNCEMENT_CHANNEL", "")
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
    return {}


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
# 3. DIFF
# ═══════════════════════════════════════════════════════════════════════════

def compute_diff(old_text: str, new_text: str) -> str:
    """Return a unified diff, capped at 150 lines to keep the prompt sane."""
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff = list(difflib.unified_diff(
        old_lines, new_lines,
        fromfile="v1 (previous)",
        tofile="v2 (updated)",
        n=3,
    ))
    if not diff:
        return ""
    diff_text = "".join(diff[:150])
    if len(diff) > 150:
        diff_text += f"\n... (diff truncated — {len(diff) - 150} more lines)"
    return diff_text


# ═══════════════════════════════════════════════════════════════════════════
# 4. AI BULLETIN GENERATION
# ═══════════════════════════════════════════════════════════════════════════

# ── CHANGED: new schema for "modified" event — clause-level before/after ──

def generate_bulletin(event: str, filename: str, new_text: str, diff_text: str) -> dict:
    """
    Ask GPT-4o to produce a structured bulletin.

    For "added":
      {
        "summary": str,
        "new_policies": [{"name": str, "overview": str, "clauses": [{"clause": str, "detail": str}]}],
        "action_required": str,
        "effective_date": str
      }

    For "modified":  ← NEW schema — clause-level two-column diff
      {
        "summary": str,                          — one-line overall description
        "modified_clauses": [                    — clauses that changed
          {
            "clause": str,                       — clause number / title
            "section": str,                      — broader section name (e.g. "Non-compete")
            "before": str,                       — what v1 said (one line)
            "after": str,                        — what v2 says (one line)
            "positive": bool                     — true if the change benefits employees or reduces legal risk
          }
        ],
        "new_clauses": [                         — entirely new clauses added
          {"clause": str, "section": str, "detail": str, "positive": bool}
        ],
        "removed_clauses": [                     — clauses removed
          {"clause": str, "section": str, "detail": str}
        ],
        "positives_summary": [str],              — 2-4 bullet strings highlighting employee-friendly wins
        "action_required": str,
        "effective_date": str
      }

    For "deleted":
      plain retirement notice dict
    """

    if event == "deleted":
        return {
            "summary": f"The document *{filename}* has been retired and is no longer in effect.",
            "modified_clauses": [],
            "new_clauses": [],
            "removed_clauses": [],
            "positives_summary": [],
            "action_required": "No action required. Contact HR if you have questions.",
            "effective_date": "Immediately",
        }

    # ── ADDED ──────────────────────────────────────────────────────────────
    if event == "added":
        prompt = f"""A brand-new HR policy document has been published: {filename}

Full document content:
---
{new_text[:6000]}
---

Return a JSON object with EXACTLY this structure:
{{
  "summary": "<one sentence: what this document covers>",
  "modified_clauses": [],
  "new_clauses": [],
  "removed_clauses": [],
  "new_policies": [
    {{
      "name": "<policy/section title>",
      "overview": "<2-3 sentences>",
      "clauses": [
        {{"clause": "<clause title or number>", "detail": "<complete explanation>"}}
      ]
    }}
  ],
  "positives_summary": ["<employee-friendly highlight 1>", "<highlight 2>"],
  "action_required": "<what employees must do, or 'No action required'>",
  "effective_date": "<date from document, or 'Effective immediately'>"
}}

Rules:
- Extract EVERY policy section as a separate new_policies entry
- Extract ALL clauses precisely — do not skip or summarise
- Return ONLY the JSON, no markdown fences, no preamble
"""

    # ── MODIFIED ────────────────────────────────────────────────────────────
    else:
        prompt = f"""An existing HR policy document has been updated: {filename}

Unified diff (v1 → v2):
---
{diff_text[:4000]}
---

Full updated document (v2) for context:
---
{new_text[:4000]}
---

Your job: produce a clause-by-clause comparison like a legal analyst would.

Return a JSON object with EXACTLY this structure:
{{
  "summary": "<one line: overall nature of this update>",
  "modified_clauses": [
    {{
      "clause": "<clause number or short title, e.g. 'Cl. 6.2'>",
      "section": "<broader section name, e.g. 'Non-compete'>",
      "before": "<precise one-line description of what v1 said>",
      "after": "<precise one-line description of what v2 says>",
      "positive": <true if this change benefits employees or reduces legal risk, false otherwise>
    }}
  ],
  "new_clauses": [
    {{
      "clause": "<clause number or title>",
      "section": "<section name>",
      "detail": "<one-line description of what this new clause does>",
      "positive": <true if employee-friendly or legally protective, false otherwise>
    }}
  ],
  "removed_clauses": [
    {{
      "clause": "<clause number or title>",
      "section": "<section name>",
      "detail": "<one-line: what was removed and its significance>"
    }}
  ],
  "positives_summary": [
    "<highlight 1: most important employee-friendly win in this version>",
    "<highlight 2>",
    "<highlight 3 — omit if fewer than 3 genuine positives>"
  ],
  "action_required": "<one line or 'No action required'>",
  "effective_date": "<date or 'Effective immediately'>"
}}

Rules:
- Be precise and brief — one line per field
- modified_clauses must always have both before AND after
- positive=true only for genuinely employee-friendly or legally protective changes
- positives_summary should read like a lawyer's highlights memo — specific, not generic
- If nothing was removed, removed_clauses = []
- Return ONLY the JSON, no markdown, no preamble
"""

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are PensionBox's internal HR policy analyst and legal summariser. "
                    "You produce clause-level structured comparisons of HR policy documents. "
                    "You always return valid JSON only — no markdown, no extra text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=2500,
        temperature=0.1,
    )

    raw = response.choices[0].message.content.strip()
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
            "modified_clauses": [],
            "new_clauses": [],
            "removed_clauses": [],
            "positives_summary": [],
            "action_required": "Contact HR if you have questions.",
            "effective_date": "Effective immediately",
        }


# ═══════════════════════════════════════════════════════════════════════════
# 5. SLACK POSTING — two-column clause diff layout
# ═══════════════════════════════════════════════════════════════════════════

# ── CHANGED: build_slack_blocks now renders a two-column before/after layout ──

def build_slack_blocks(bulletin: dict, filename: str, event: str) -> list:
    """
    Render the bulletin as Slack Block Kit blocks.

    For "modified" events: two-column clause-level diff —
      Left column  = v1 (what it said before)
      Right column = v2 (what it says now)
    Plus a dedicated ✅ Positives section.
    """
    event_emoji = {"added": "🆕", "modified": "📝", "deleted": "🗑️"}.get(event, "📋")
    event_label = {
        "added":    "New Policy Published",
        "modified": "Policy Updated",
        "deleted":  "Policy Retired",
    }.get(event, "Policy Change")
    timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")

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

    # ── Effective date + action ─────────────────────────────────────────────
    eff    = bulletin.get("effective_date", "Effective immediately")
    action = bulletin.get("action_required", "No action required.")
    blocks.append({
        "type": "section",
        "fields": [
            {"type": "mrkdwn", "text": f"*📅 Effective*\n{eff}"},
            {"type": "mrkdwn", "text": f"*⚡ Action required*\n{action}"},
        ],
    })
    blocks.append({"type": "divider"})

    # ── ✅ Positives (employee-friendly highlights) ─────────────────────────
    positives = bulletin.get("positives_summary", [])
    if positives:
        lines = ["*✅ Key positives in this version*\n"]
        for p in positives:
            lines.append(f"• {p}")
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "\n".join(lines)},
        })
        blocks.append({"type": "divider"})

    # ── Two-column clause diff (modified clauses) ────────────────────────────
    #
    # Slack's "fields" array renders as two columns side-by-side.
    # We group clauses by section, emit a section header, then pairs of
    # [v1 cell, v2 cell] for each clause.
    #
    modified_clauses = bulletin.get("modified_clauses", [])
    if modified_clauses:
        # Column header row
        blocks.append({
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": "*🔴 v1 — previous*"},
                {"type": "mrkdwn", "text": "*🟢 v2 — updated*"},
            ],
        })

        # Group by section for readability
        sections: dict[str, list] = {}
        for c in modified_clauses:
            sec = c.get("section", "General")
            sections.setdefault(sec, []).append(c)

        for sec_name, clauses in sections.items():
            # Section label spanning both columns (context block)
            blocks.append({
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": f"*— {sec_name} —*"}],
            })
            for c in clauses:
                clause_label = c.get("clause", "")
                before = c.get("before", "")
                after  = c.get("after", "")
                positive_marker = " 🌟" if c.get("positive") else ""

                # Each clause = one fields block (two columns)
                blocks.append({
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*{clause_label}*\n❌ {before}"},
                        {"type": "mrkdwn", "text": f"*{clause_label}*{positive_marker}\n✅ {after}"},
                    ],
                })

        blocks.append({"type": "divider"})

    # ── New clauses ──────────────────────────────────────────────────────────
    new_clauses = bulletin.get("new_clauses", [])
    # Also handle "added" event new_policies format
    new_policies = bulletin.get("new_policies", [])

    if new_clauses:
        lines = ["*🆕 New clauses added*\n"]
        for c in new_clauses:
            star = " 🌟" if c.get("positive") else ""
            lines.append(f"• *{c.get('clause')}* ({c.get('section', '')}){star} — {c.get('detail')}")
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "\n".join(lines)},
        })
        blocks.append({"type": "divider"})

    elif new_policies:
        # "added" event — full policy breakdown
        lines = ["*🆕 New policies published*\n"]
        for p in new_policies:
            lines.append(f"*{p.get('name')}*")
            lines.append(p.get("overview", ""))
            for cl in p.get("clauses", []):
                lines.append(f"  • *{cl.get('clause')}* — {cl.get('detail')}")
            lines.append("")
        # Slack section text cap = 3000 chars; chunk if needed
        text = "\n".join(lines)
        for chunk_start in range(0, len(text), 2900):
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": text[chunk_start:chunk_start + 2900]},
            })
        blocks.append({"type": "divider"})

    # ── Removed clauses ──────────────────────────────────────────────────────
    removed_clauses = bulletin.get("removed_clauses", [])
    if removed_clauses:
        lines = ["*🗑️ Clauses removed*\n"]
        for c in removed_clauses:
            lines.append(f"• *{c.get('clause')}* ({c.get('section', '')}) — {c.get('detail')}")
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "\n".join(lines)},
        })
        blocks.append({"type": "divider"})

    # ── Footer ───────────────────────────────────────────────────────────────
    blocks.append({
        "type": "context",
        "elements": [{
            "type": "mrkdwn",
            "text": (
                f"_Automatically detected by PensionBox HR Bot · {timestamp} · "
                "🌟 = positive change · Contact HR for questions_"
            ),
        }],
    })

    # Slack hard limit: 50 blocks per message
    if len(blocks) > 50:
        blocks = blocks[:49]
        blocks.append({
            "type": "context",
            "elements": [{
                "type": "mrkdwn",
                "text": "_Some clauses truncated due to length. See the full document in /docs._",
            }],
        })

    return blocks


def post_to_slack(bulletin: dict, filename: str, event: str):
    if not ANNOUNCEMENT_CH:
        log.warning("SLACK_ANNOUNCEMENT_CHANNEL not set — printing bulletin:\n" + json.dumps(bulletin, indent=2))
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
    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(f"{API_BASE_URL}/reindex")
            if resp.status_code == 200:
                log.info("Reindex completed successfully.")
            else:
                log.warning(f"Reindex returned {resp.status_code}: {resp.text}")
    except Exception as e:
        log.warning(f"Could not trigger reindex (is the API running?): {e}")


# ═══════════════════════════════════════════════════════════════════════════
# 7. MAIN WATCHER LOOP
# ═══════════════════════════════════════════════════════════════════════════

def scan_docs(state: dict):
    events = []
    current_files = {}

    if not DOCS_DIR.exists():
        log.warning(f"Docs dir {DOCS_DIR} does not exist yet.")
        return events, current_files

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

    for fname in list(state.keys()):
        if fname not in current_files:
            events.append({"event": "deleted", "path": None, "filename": fname})

    return events, current_files


def handle_event(ev: dict, state: dict):
    event    = ev["event"]
    filename = ev["filename"]
    path     = ev["path"]

    log.info(f"Change detected — {event.upper()}: {filename}")

    new_text  = extract_text(path) if path else ""
    old_text  = load_text_snapshot(filename) if event == "modified" else ""
    diff_text = compute_diff(old_text, new_text) if event == "modified" else ""

    log.info(f"Generating bulletin for {filename}...")
    bulletin = generate_bulletin(event, filename, new_text, diff_text)
    log.info(f"Bulletin: {json.dumps(bulletin, indent=2)}")

    post_to_slack(bulletin, filename, event)

    if event == "deleted":
        state.pop(filename, None)
        snap = CACHE_DIR / f"{filename}.txt"
        if snap.exists():
            snap.unlink()
    else:
        state[filename] = {
            "hash": file_hash(path),
            "last_seen": datetime.now().isoformat(),
        }
        save_text_snapshot(filename, new_text)


def run():
    log.info("PensionBox HR Watcher Agent started.")
    log.info(f"Watching: {DOCS_DIR.resolve()}")
    log.info(f"Poll interval: {POLL_INTERVAL}s")
    log.info(f"Announcement channel: {ANNOUNCEMENT_CH or '(not set)'}")

    state = load_cache()

    if not state:
        log.info("First run — snapshotting existing docs (no announcements).")
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