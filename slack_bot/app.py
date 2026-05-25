"""
slack_bot/app.py

Slack Bolt app — listens for:
  - @PensionBox HR Bot mentions in any channel
  - Direct messages to the bot

Run with:
    python slack_bot/app.py
"""

import os
import logging
import httpx
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

app = App(
    token=os.environ["SLACK_BOT_TOKEN"],
    signing_secret=os.environ["SLACK_SIGNING_SECRET"],
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def ask_hr_bot(question: str, employee_name: str) -> dict:
    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            f"{API_BASE_URL}/ask",
            json={"question": question, "employee_name": employee_name},
        )
        response.raise_for_status()
        return response.json()


def format_reply(answer: str, sources: list[str]) -> str:
    text = answer
    if sources:
        sources_text = "\n".join(f"• {src}" for src in sources)
        text += f"\n\n📄 *Sources:*\n{sources_text}"
    return text


def get_display_name(client, user_id: str) -> str:
    try:
        result = client.users_info(user=user_id)
        profile = result["user"]["profile"]
        return profile.get("display_name") or profile.get("real_name") or "Employee"
    except Exception:
        return "Employee"


@app.event("app_mention")
def handle_mention(event, say, client):
    user_id = event["user"]
    bot_user_id = client.auth_test()["user_id"]

    question = event.get("text", "").replace(f"<@{bot_user_id}>", "").strip()

    if not question:
        say("Hi! Ask me anything about PensionBox HR policies. 👋")
        return

    employee_name = get_display_name(client, user_id)
    log.info(f"Mention from {employee_name}: {question}")

    say("_Looking that up for you..._ 🔍")

    try:
        result = ask_hr_bot(question, employee_name)
        say(format_reply(result["answer"], result["sources"]))
    except Exception as e:
        log.error(f"Error: {e}", exc_info=True)
        say("Sorry, something went wrong. Please contact HR directly for now.")


@app.event("message")
def handle_dm(event, say, client):
    # Only handle DMs, ignore bot messages
    if event.get("channel_type") != "im":
        return
    if event.get("bot_id"):
        return

    user_id = event["user"]
    question = event.get("text", "").strip()

    if not question:
        return

    employee_name = get_display_name(client, user_id)
    log.info(f"DM from {employee_name}: {question}")

    say("_Looking that up for you..._ 🔍")

    try:
        result = ask_hr_bot(question, employee_name)
        say(format_reply(result["answer"], result["sources"]))
    except Exception as e:
        log.error(f"Error: {e}", exc_info=True)
        say("Sorry, something went wrong. Please contact HR directly for now.")


if __name__ == "__main__":
    log.info("Starting PensionBox HR Bot...")
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start()
