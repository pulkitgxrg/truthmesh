"""Print the AICredits wallet balance (GET https://aicredits.in/api/v1/credits) and append it to a log."""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from cvsm.llm.providers import load_dotenv

load_dotenv()
req = urllib.request.Request("https://aicredits.in/api/v1/credits",
                             headers={"Authorization": f"Bearer {os.environ['AICREDITS_API_KEY']}", "User-Agent": "curl/8.7.1"})
data = json.loads(urllib.request.urlopen(req, timeout=30).read())["data"]
data.pop("label", None)
row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "note": " ".join(sys.argv[1:]), **data}
Path("runs").mkdir(exist_ok=True)
with open("runs/credits_log.jsonl", "a") as f:
    f.write(json.dumps(row) + "\n")
print(json.dumps(row))
