#!/usr/bin/env python3
"""
notify.py — send a completion notification for the ST pipeline.

Channels:
  1. Email via Gmail SMTP   (GMAIL_USER + GMAIL_APP_PASSWORD + NOTIFY_EMAIL_TO)
  2. macOS desktop notification (osascript)

Both are best-effort: a failure in one does not block the other.

Usage:
  python notify.py "Subject line" "Body text"
  echo "body" | python notify.py "Subject line"
"""
import os
import sys
import ssl
import smtplib
import subprocess
from email.message import EmailMessage


def desktop(subject: str, body: str) -> None:
    snippet = body.replace('"', "'").replace("\n", " ")[:230]
    title = subject.replace('"', "'")[:120]
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{snippet}" with title "{title}"'],
            check=False, timeout=15,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[notify] desktop notification failed: {e}", file=sys.stderr)


def email(subject: str, body: str) -> None:
    user = os.environ.get("GMAIL_USER", "").strip()
    pw = os.environ.get("GMAIL_APP_PASSWORD", "").replace(" ", "").strip()
    to = os.environ.get("NOTIFY_EMAIL_TO", user).strip()
    if not (user and pw and to):
        print("[notify] email skipped: set GMAIL_USER, GMAIL_APP_PASSWORD, "
              "NOTIFY_EMAIL_TO in .env", file=sys.stderr)
        return
    msg = EmailMessage()
    msg["From"] = user
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as s:
            s.starttls(context=ctx)
            s.login(user, pw)
            s.send_message(msg)
        print(f"[notify] email sent to {to}")
    except Exception as e:  # noqa: BLE001
        print(f"[notify] email FAILED: {e}", file=sys.stderr)


if __name__ == "__main__":
    subject = sys.argv[1] if len(sys.argv) > 1 else "Spatial Transcriptomics pipeline"
    body = sys.argv[2] if len(sys.argv) > 2 else sys.stdin.read()
    desktop(subject, body)
    email(subject, body)
