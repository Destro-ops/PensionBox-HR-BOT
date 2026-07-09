"""
slack_bot/app.py

Slack Bolt app — listens for:
  - @PensionBox HR Bot mentions in any channel
  - Direct messages to the bot

Run with:
    python slack_bot/app.py
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import schedule
import time
import requests
import os
import logging
import httpx
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from datetime import date, datetime
import json
import re
from openai import OpenAI

load_dotenv()
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
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

def detect_intent(text: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
               "content": """Classify the user's message into one of these intents:
- "leave" → asking about leave status, who is on leave, attendance records
- "duration" → asking about work hours, active duration, how long someone worked, productive percent, productivity, performance metrics, employee statistics
- "discrepancy" → asking about anomalies, mismatches, attendance issues, who didn't show up, work discrepancies
- "hr_policy" → anything else related to HR policies, benefits, salary, general HR questions

Return ONLY one word: leave, duration, discrepancy, or hr_policy"""
            },
            {"role": "user", "content": text}
        ],
        max_tokens=10,
        temperature=0
    )
    return response.choices[0].message.content.strip().lower()


def get_display_name(client, user_id: str) -> str:
    try:
        result = client.users_info(user=user_id)
        profile = result["user"]["profile"]
        return profile.get("display_name") or profile.get("real_name") or "Employee"
    except Exception:
        return "Employee"
    
def extract_dates(text: str) -> list[str]:
    today = date.today().isoformat()
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"""Today is {today} ({date.today().strftime('%A')}). 

The user is asking about employee attendance or leave. Your job is to figure out which date(s) they are referring to.

Think step by step:
- What time period is the user asking about?
- Convert that to specific calendar dates
- Skip weekends (Saturday and Sunday) since there's no attendance data for those days

Return ONLY a JSON array of dates in YYYY-MM-DD format. No explanation, no preamble.

Examples of how to think:
- "today" → just today's date
- "tomorrow" → just tomorrow's date  
- "this week" → Monday to Friday of the current week
- "next week" → Monday to Friday of next week
- "last 3 days" → the 3 most recent weekdays including today
- "12th june" → just that date if it's a weekday
- "this month" → all weekdays from the 1st of the current month to today
- "last month" → all weekdays of the previous month
- no date mentioned → just today

Current date: {today}
Day of week: {date.today().strftime('%A')}"""
            },
            {"role": "user", "content": text}
        ],
        max_tokens=300,
        temperature=0.2
    )
    
    extracted = response.choices[0].message.content.strip()
    try:
        dates = json.loads(extracted)
        return dates
    except ValueError:
        return [today]

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
        intent = detect_intent(question)
        AUTHORIZED_IDS = os.environ.get("AUTHORIZED_USER_IDS", "").split(",")
        if intent in ["duration", "leave", "discrepancy"] and user_id not in AUTHORIZED_IDS:
            say("Sorry, you are not authorized to access this information.")
            return
        if intent == "duration":
          say(get_work_durations(question))
        elif intent == "leave":
          say(get_leave_status(question))
        elif intent == "discrepancy":
          say(get_discrepancies(question))
        else:
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
        intent = detect_intent(question)
        AUTHORIZED_IDS = os.environ.get("AUTHORIZED_USER_IDS", "").split(",")
        print(f"Detected intent: {intent}, user_id: {user_id}, authorized_ids: {AUTHORIZED_IDS}")
        if intent in ["duration", "leave", "discrepancy"] and user_id not in AUTHORIZED_IDS:
            say("Sorry, you are not authorized to access this information.")
            return
        if intent == "duration":
          say(get_work_durations(question))
        elif intent == "leave":
          say(get_leave_status(question))
        elif intent == "discrepancy":
          say(get_discrepancies(question))
        else:
          result = ask_hr_bot(question, employee_name)
          say(format_reply(result["answer"], result["sources"]))
    except Exception as e:
       log.error(f"Error: {e}", exc_info=True)
       say("Sorry, something went wrong. Please contact HR directly for now.")

def get_we360_token() -> str:
    resp = requests.post(
        "https://auth.in.we360.ai/realms/ind-prod/protocol/openid-connect/token",
        data={
            "client_id": os.environ["WE360_CUSTOMER_ID"],
            "username": os.environ["WE360_EMAIL"],
            "password": os.environ["WE360_PASSWORD"],
            "grant_type": "password"
        }
    )
    resp.raise_for_status()
    return resp.json()["access_token"]

def get_work_durations(question: str = "") -> str:
    target_dates = extract_dates(question)
    token = get_we360_token()
    url = os.environ["WE360_API_URL"]
    
    all_results = {}  # {date: [employees]}
    
    for target_date in target_dates:
        payload = {
            "start_date": f"{target_date}T00:00:00",
            "end_date": f"{target_date}T23:59:59",
            "mode": "detailed",
            "columns": ["first_name", "active_duration", "productive_percent"],
            "limit": 100,
            "page": 1
        }
        headers = {"Authorization": f"Bearer {token}"}
        
        results = []
        while True:
            r = requests.post(url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
            results.extend(data["data"])
            if not data["pagination"]["has_next"]:
                break
            payload["page"] += 1
        
        all_results[target_date] = results
    
    # Send to GPT to analyze and answer the actual question
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are an HR analyst. Analyze employee productivity data and answer the HR's question directly.
Be concise. Use employee first names only.
If asked for highest/lowest, rank them clearly.
If asked for a summary, give a clean overview.
Format nicely for Slack."""
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nData:\n{json.dumps(all_results, indent=2)}"
            }
        ],
        max_tokens=800,
        temperature=0.2
    )
    
    return response.choices[0].message.content





def fetch_employee_date(emp, target_date):
    payload = {
        "auth": {
            "id": int(os.environ["RAZORPAY_PAYROLL_ID"]),
            "key": os.environ["RAZORPAY_PAYROLL_KEY"]
        },
        "request": {"type": "attendance", "sub-type": "fetch"},
        "data": {"email": emp["email"], "date": target_date}
    }
    r = requests.post("https://payroll.razorpay.com/api/att", json=payload)
    response = r.json()
    
    if "error" in response:
        return (target_date, emp["name"], "No record found", "")
    
    data = response["data"]
    status = data.get("status", {}).get("description", "Unknown")
    leave = data.get("leave-type", {}).get("description", "N/A")
    return (target_date, emp["name"], status, leave)




def get_leave_status(question: str = "") -> str:
    # If question implies ranking/aggregation, default to current month weekdays
    aggregation_keywords = ["most", "least", "highest", "lowest", "top", "maximum", "minimum", "rank"]
    if any(kw in question.lower() for kw in aggregation_keywords):
        from datetime import timedelta
        today = date.today()
        start = today.replace(day=1)
        target_dates = []
        d = start
        while d <= today:
            if d.weekday() < 5:
                target_dates.append(d.isoformat())
            d += timedelta(days=1)
    else:
        target_dates = extract_dates(question)

    print("target_dates:", target_dates)
    
    with open("data/employees.json", "r") as f:
        employees = json.load(f)
    
    tasks = [(emp, d) for d in target_dates for emp in employees]
    results = {}
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_employee_date, emp, d): (emp, d) for emp, d in tasks}
        for future in as_completed(futures):
            target_date, name, status, leave = future.result()
            results.setdefault(target_date, []).append((name, status, leave))
    
    print("results:", results)
    
    leave_count: dict[str, int] = {}
    for d, records in results.items():
        for name, status, leave in records:
            if status and any(s in status.lower() for s in ["leave"]):
                leave_count[name] = leave_count.get(name, 0) + 1

    if not leave_count:
        return "*Leave records:*\nNo leaves taken in the requested period."

    sorted_leaves = sorted(leave_count.items(), key=lambda x: x[1], reverse=True)

    lines = ["*🏖️ Leave count:*"]
    for name, count in sorted_leaves:
        lines.append(f"• {name}: {count} day{'s' if count > 1 else ''}")

    return "\n".join(lines)


def get_raw_leave_data(target_date: str) -> dict:
    """Returns {email: {name, status, leave_type}} for a given date."""
    with open("data/employees.json", "r") as f:
        employees = json.load(f)
    
    result = {}
    
    def fetch(emp):
        payload = {
         "auth": {
            "id": int(os.environ["RAZORPAY_PAYROLL_ID"]),
            "key": os.environ["RAZORPAY_PAYROLL_KEY"]
         },
         "request": {"type": "attendance", "sub-type": "fetch"},
         "data": {"email": emp["email"], "date": target_date}
        }
        try:
           r = requests.post("https://payroll.razorpay.com/api/att", json=payload)
           resp = r.json()
        except Exception:
           return emp["email"], {"name": emp["name"], "status": "no_record", "leave_type": None}
    
        if "error" in resp:
           return emp["email"], {"name": emp["name"], "status": "no_record", "leave_type": None}
        data = resp["data"]
        return emp["email"], {
           "name": emp["name"],
           "status": data.get("status", {}).get("description", "unknown"),
           "leave_type": data.get("leave-type", {}).get("description", None)
    }
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch, emp) for emp in employees]
        for future in as_completed(futures):
            email, data = future.result()
            result[email] = data
    
    return result


def get_raw_we360_data(target_date: str) -> dict:
    """Returns {first_name_lower: {name, active_duration}} for a given date."""
    token = get_we360_token()
    url = os.environ["WE360_API_URL"]
    
    payload = {
      "start_date": f"{target_date}T00:00:00",
      "end_date": f"{target_date}T23:59:59",
      "mode": "detailed",
      "columns": ["first_name", "last_name", "active_duration", "productive_percent"],
      "limit": 100,
      "page": 1
    }
    headers = {"Authorization": f"Bearer {token}"}
    
    result = {}
    while True:
        r = requests.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        for e in data["data"]:
            key = e["first_name"].split()[0].lower()
            result[key] = {
                "name": f"{e['first_name']} {e['last_name']}",
                "active_duration": e.get("active_duration", "unavailable"),
                "productive_percent": e.get("productive_percent", "unavailable")
            }
        if not data["pagination"]["has_next"]:
            break
        payload["page"] += 1
    
    return result

def get_discrepancies(question: str = "") -> str:
    target_date = date.today().isoformat()
    
    leave_data = get_raw_leave_data(target_date)
    we360_data = get_raw_we360_data(target_date)
    
    # Merge both datasets
   # Merge both datasets — only include employees tracked by we360
    merged = []
    for email, leave in leave_data.items():
      first = leave["name"].split()[0].lower()
      activity = we360_data.get(first, {})
    
      if not activity:  # skip employees not in we360
        continue
    
      merged.append({
         "name": leave["name"],
         "date": target_date,
         "leave_status": leave["status"],
         "leave_type": leave["leave_type"],
         "active_duration": activity.get("active_duration", "unavailable"),
         "productive_percent": activity.get("productive_percent", "unavailable")
    })
    
    # Send to GPT
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {   "role": "system",
                "content": """You are an HR analyst. Analyze employee attendance and activity data and flag discrepancies.

Flag these cases:
1. Employee is on leave but has significant productive_percent (> 10%)
2. Employee has no leave record and productive_percent is 0 or very low (< 10%) on a weekday
3. Employee is marked present but productive_percent is suspiciously low (< 10%)
4. If productive_percent is "unavailable", only analyze based on leave status

For each discrepancy, provide:
- Employee name
- What the issue is
- A brief recommendation

If no discrepancies found, say "No discrepancies found for today."
Be concise and clear."""
            },
            {
                "role": "user",
                "content": f"Analyze this attendance data for {target_date}:\n{json.dumps(merged, indent=2)}"
            }
        ],
        max_tokens=1000,
        temperature=0.2
    )
    
    return f"*🚨 Attendance Discrepancy Report — {target_date}*\n\n" + response.choices[0].message.content

def post_daily_discrepancy_report():
    try:
        report = get_discrepancies()
        from slack_sdk import WebClient
        client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
        
        AUTHORIZED_IDS = os.environ.get("AUTHORIZED_USER_IDS", "").split(",")
        for user_id in AUTHORIZED_IDS:
            if user_id.strip():
                client.chat_postMessage(
                    channel=user_id.strip(),  # sending to user_id directly opens a DM
                    text=report
                )
        log.info("Daily discrepancy report sent to authorized users.")
    except Exception as e:
        log.error(f"Failed to post daily report: {e}", exc_info=True)

def run_scheduler():
    schedule.every().day.at("15:28").do(post_daily_discrepancy_report)
    while True:
        schedule.run_pending()
        time.sleep(60)





if __name__ == "__main__":
    threading.Thread(target=run_scheduler, daemon=True).start()
    log.info("Starting PensionBox HR Bot...")
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start()
