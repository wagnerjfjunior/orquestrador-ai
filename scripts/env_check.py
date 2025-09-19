import os, sys

required = ["OPENAI_API_KEY", "GEMINI_API_KEY"]
missing = [k for k in required if not os.getenv(k)]
if missing:
    print("ERROR: missing env vars:", ", ".join(missing))
    sys.exit(1)

print("OK: all required env vars are present.")
for k in required + ["OPENAI_MODEL", "GEMINI_MODEL", "LOG_LEVEL"]:
    v = os.getenv(k)
    if v:
        print(f"{k}={v[:6]}... (len={len(v)})" if "KEY" in k else f"{k}={v}")
