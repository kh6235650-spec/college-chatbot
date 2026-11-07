"""بيجرّب مفتاح Gemini (من api_key.txt) على كذا موديل ويطبع النتيجة. مابيطبعش المفتاح."""
from google import genai

key = open("api_key.txt", encoding="utf-8").read().strip()
print(f"المفتاح اللي اتقرا ينتهي بـ: ...{key[-4:]}  (طوله {len(key)} حرف)\n")
client = genai.Client(api_key=key)

names = []
for m in client.models.list():
    n = (m.name or "").replace("models/", "")
    acts = getattr(m, "supported_actions", None) or []
    if ("flash" in n or "gemma" in n) and "generateContent" in acts and not any(
        s in n for s in ("image", "tts", "live", "audio", "robotics", "computer", "transcribe")
    ):
        names.append(n)

for n in names[:10]:
    try:
        r = client.models.generate_content(model=n, contents="قول مرحبا")
        print(f"[شغال]  {n}")
    except Exception as e:
        msg = str(e).replace("\n", " ")
        tag = "الحصة اليومية خلصت" if "PerDay" in msg else msg[:110]
        print(f"[فاشل]  {n}  ->  {tag}")
