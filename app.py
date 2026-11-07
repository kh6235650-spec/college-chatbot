"""
شات بوت الكلية (Nabeeh) - مشروع جامعي
- الواجهة: Streamlit
- المحرك: Claude (Anthropic) أو Gemini (Google) - تختار من الشريط الجانبي،
  ولو واحد فيهم وقع التاني بيكمّل مكانه تلقائياً.
- المعرفة: ملف knowledge.md (RAG بسيط باستخدام TF-IDF) لأسئلة الكلية،
  وأي سؤال تاني بيتجاوب عليه من معرفة الموديل نفسه (مش محدود بالكلية).
- التخزين: SQLite (سجل المحادثات) + زر لتفريغه من الشريط الجانبي.

التشغيل:
    pip install -r requirements.txt
    streamlit run app.py

المفاتيح (بالترتيب: ملف ← متغير بيئة ← خانة في الشريط الجانبي):
    Claude : ملف anthropic_key.txt  أو  ANTHROPIC_API_KEY
    Gemini : ملف api_key.txt        أو  GEMINI_API_KEY
المفتاح عمره ما بيتكتب جوه الكود.
"""

import base64
import os
import re
import sqlite3
import time
from datetime import datetime

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import anthropic
except ImportError:  # الحزمة مش متسطبة
    anthropic = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

# ----------------------------- الإعدادات -----------------------------
KNOWLEDGE_FILE = "knowledge.md"
DB_FILE = "chat_history.db"
GEMINI_KEY_FILE = "AQ.Ab8RN6I-nzQHRp5zSVc2zef45FHEPZr9DxA9-OKdZQUJl0eugw"
CLAUDE_KEY_FILE = "anthropic_key.txt"

# "gemini-flash-latest" اسم بديل بيشاور دايماً على أحدث flash، فمابيتشالش زي الأسماء القديمة
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
CLAUDE_MODELS = {
    "Claude Sonnet 5.5 (متوازن)": "claude-sonnet-5-5",
    "Claude Haiku 5.5 (أسرع وأرخص)": "claude-haiku-5-5",
    "Claude Opus 5.5 (الأقوى)": "claude-opus-5-5",
}
CLAUDE_FALLBACK = "claude-haiku-5-5"

TOP_K = 3            # عدد الأجزاء اللي بنجيبها من قاعدة المعرفة
MIN_SCORE = 0.08     # أقل درجة تشابه عشان نعتبر الجزء مفيد
MAX_HISTORY = 20     # آخر كام رسالة نبعتها للموديل
MAX_PDF_MB = 20      # أقصى حجم لملف المحاضرة
MAX_TOKENS = 4096
SUMMARY_QUESTION = "لخّص المحاضرة اللي رفعتها في نقاط مرتبة"

SYSTEM_PROMPT = """أنت "نبيه"، مساعد ذكي شغال على طلاب الكلية، وتقدر تساعد في أي موضوع.
القواعد:
1. جاوب بنفس لغة الطالب. لو بيكتب عربي استخدم لهجة مصرية مبسطة ومحترمة. خلي الرد منظم ومختصر،
   وتوسّع لو السؤال محتاج شرح أو كود أو خطوات.
2. تقدر تجاوب على أي سؤال: مذاكرة، شرح مواد، برمجة، ترجمة، كتابة، أفكار، ثقافة عامة، نصايح... مفيش موضوع ممنوع
   طالما مش ضار. ماتعتذرش عن سؤال لمجرد إنه مش عن الكلية.
3. لو السؤال عن حاجة خاصة بالكلية (مواعيد، تسجيل، مصروفات، امتحانات، غياب، شهادات...) اعتمد على
   "المعلومات المرجعية" المرفقة. ولو المعلومة مش موجودة فيها قول بصراحة إنك مش متأكد وانصح الطالب
   يرجع لشؤون الطلاب أو الموقع الرسمي. ماتخترعش مواعيد أو أرقام أو أسماء خاصة بالكلية.
4. لو في محاضرة PDF مرفقة مع السؤال: جاوب منها، ولو طلب الطالب تلخيص اعمل ملخص منظم
   (فكرة المحاضرة، أهم النقاط، التعريفات والقوانين المهمة). لو المعلومة مش في المحاضرة قول كده.
5. لو الطالب طلب تلخيص محاضرة ومفيش ملف مرفق، قوله يرفع ملف PDF من الشريط الجانبي.
6. لو مش متأكد من معلومة عامة قول كده بدل ما تخمّن.
"""


# ----------------------------- قاعدة البيانات -----------------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            created_at TEXT
        )"""
    )
    conn.commit()
    return conn


def save_message(session_id, role, content):
    conn = init_db()
    conn.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
        (session_id, role, content, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()


def message_count():
    conn = init_db()
    n = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    conn.close()
    return n


def clear_messages():
    """بيمسح كل الرسائل المخزنة ويرجّع العداد يبدأ من 1."""
    conn = init_db()
    conn.execute("DELETE FROM messages")
    conn.execute("DELETE FROM sqlite_sequence WHERE name = 'messages'")
    conn.commit()
    conn.execute("VACUUM")
    conn.close()


# ----------------------------- قاعدة المعرفة (RAG) -----------------------------
@st.cache_resource
def load_knowledge():
    """بيقسم knowledge.md لأجزاء (كل عنوان ## جزء) ويبني فهرس TF-IDF."""
    if not os.path.exists(KNOWLEDGE_FILE):
        return [], None, None
    with open(KNOWLEDGE_FILE, encoding="utf-8") as f:
        text = f.read()
    chunks = [c.strip() for c in re.split(r"\n(?=## )", text) if c.strip()]
    if not chunks:
        return [], None, None
    # char n-grams بتشتغل كويس مع العربي واختلاف الكتابة
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    matrix = vectorizer.fit_transform(chunks)
    return chunks, vectorizer, matrix


def retrieve(question, previous_question=""):
    """بيجيب أجزاء الكلية المناسبة. بنضيف السؤال اللي قبله عشان أسئلة زي "ومصاريفها؟" تتفهم."""
    chunks, vectorizer, matrix = load_knowledge()
    if not chunks:
        return []
    query = f"{previous_question} {question}".strip()
    scores = cosine_similarity(vectorizer.transform([query]), matrix)[0]
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    return [chunks[i] for i in ranked[:TOP_K] if scores[i] >= MIN_SCORE]


# ----------------------------- مفاتيح وأخطاء -----------------------------
def read_key(filename, env_name, state_key):
    """ملف ← متغير بيئة ← خانة الشريط الجانبي."""
    if os.path.exists(filename):
        with open(filename, encoding="utf-8") as f:
            key = f.read().strip()
        if key:
            return key
    key = os.environ.get(env_name, "").strip()
    if key:
        return key
    return st.session_state.get(state_key, "").strip()


def status_of(e):
    """كود الـ HTTP من الاستثناء (أدق من البحث جوه النص)."""
    for attr in ("status_code", "code"):
        v = getattr(e, attr, None)
        if isinstance(v, int):
            return v
    return None


CONN_WORDS = ("10054", "10060", "forcibly closed", "connection reset",
              "remoteprotocolerror", "connection aborted", "connection error")
TIMEOUT_WORDS = ("timed out", "timeout", "deadline")


class ProviderError(Exception):
    """كل المحركات فشلت؛ الرسالة جاهزة للعرض."""


def friendly_error(e):
    """بيحول الأخطاء لرسالة مفهومة بالعربي."""
    if isinstance(e, ProviderError):
        return str(e)
    text = str(e)
    low = text.lower()
    code = status_of(e)
    if "getaddrinfo" in low or "connecterror" in low or "temporary failure" in low:
        return "مفيش اتصال بالإنترنت. اتأكد من النت وجرب تاني."
    if any(w in low for w in TIMEOUT_WORDS):
        return "الرد اتأخر. جرب تبعت السؤال تاني بعد شوية."
    if any(w in low for w in CONN_WORDS):
        return "الاتصال اتقطع في النص. غالباً النت مش ثابت: جرب شبكة تانية أو اقفل أي VPN."
    if "credit balance" in low:
        return "رصيد حساب Anthropic خلص. اشحن رصيد من console.anthropic.com أو استخدم Gemini."
    if code in (401, 403) or "api key" in low or "api_key" in low or "authentication" in low:
        return "في مشكلة في مفتاح الـ API (غلط أو اتلغى أو منتهي). اعمل مفتاح جديد وحطه مكان القديم."
    if code == 404 or "not_found" in low:
        return "اسم الموديل مش متاح. غيّره من الشريط الجانبي أو من الإعدادات في أول app.py."
    if code == 429 or "resource_exhausted" in low:
        if "PerDay" in text:
            return "خلصت الحصة اليومية المجانية للموديل ده. جرب بكرة أو استخدم Claude."
        return "وصلنا للحد المسموح للطلبات. استنى دقيقة وجرب تاني."
    if code in (500, 502, 503, 504, 529) or "unavailable" in low or "overloaded" in low:
        return "الموديل عليه ضغط عالي دلوقتي. استنى دقيقة وابعت سؤالك تاني."
    return f"حصل خطأ أثناء الاتصال بالموديل: {e}"


# ----------------------------- بناء الرسالة -----------------------------
def compose_user_text(question, history, lecture):
    """بنضيف معلومات الكلية بس لو لقينا حاجة مناسبة، وإلا السؤال يروح زي ما هو."""
    prev = ""
    for m in reversed(history):
        if m["role"] == "user":
            prev = m["content"]
            break
    chunks = retrieve(question, prev)
    if chunks:
        text = (
            "المعلومات المرجعية عن الكلية (استخدمها بس لو السؤال عن الكلية، وتجاهلها لو مالهاش علاقة):\n"
            + "\n\n".join(chunks) + f"\n\nسؤال الطالب: {question}"
        )
    else:
        text = question
    if lecture:
        text = (
            f'(الطالب رافع محاضرة PDF اسمها "{lecture["name"]}" وهي مرفقة مع الرسالة. '
            "لو السؤال عنها جاوب منها، ولو طلب تلخيص اعمل ملخص منظم في نقاط.)\n\n" + text
        )
    return text, chunks


def clean_history(history):
    """آخر MAX_HISTORY رسالة، بتبدأ برسالة طالب (شرط Claude)."""
    h = [m for m in history[-MAX_HISTORY:] if m.get("content")]
    while h and h[0]["role"] != "user":
        h.pop(0)
    return h


# ----------------------------- Claude -----------------------------
def ask_claude(key, model, history, user_text, lecture):
    client = anthropic.Anthropic(api_key=key, timeout=90.0, max_retries=2)
    messages = [{"role": m["role"], "content": m["content"]} for m in clean_history(history)]
    if lecture:
        data = base64.standard_b64encode(lecture["bytes"]).decode("ascii")
        content = [
            {"type": "document",
             "source": {"type": "base64", "media_type": "application/pdf", "data": data}},
            {"type": "text", "text": user_text},
        ]
    else:
        content = user_text
    messages.append({"role": "user", "content": content})

    last_error = None
    for name in dict.fromkeys([model, CLAUDE_FALLBACK]):
        try:
            resp = client.messages.create(
                model=name, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT, messages=messages
            )
            st.session_state.last_model = name
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            return text or "مقدرتش أطلع رد، جرب تسأل بشكل تاني."
        except anthropic.APIStatusError as e:
            last_error = e
            if e.status_code in (404, 429, 500, 502, 503, 529):
                continue  # جرب الموديل البديل
            raise
        except anthropic.APITimeoutError as e:
            last_error = e
            continue
    raise last_error


# ----------------------------- Gemini -----------------------------
def get_gemini_client(key):
    try:
        # لو Google ما ردتش في 45 ثانية نقطع ونجرب موديل تاني
        return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=45000))
    except Exception:
        return genai.Client(api_key=key)


def list_flash_models(client):
    """موديلات flash المتاحة لمفتاحك. مابنخزنش النتيجة لو فشل الطلب (عشان مايفضلش فاضي)."""
    cached = st.session_state.get("flash_models")
    if cached:
        return cached
    skip = ("image", "tts", "live", "audio", "embed", "robotics", "computer", "veo", "imagen",
            "transcribe", "omni", "preview")  # omni وpreview حصتها المجانية غالباً صفر
    names = []
    try:
        for m in client.models.list():
            name = (m.name or "").replace("models/", "")
            actions = getattr(m, "supported_actions", None) or []
            if "flash" in name and "generateContent" in actions and not any(s in name for s in skip):
                names.append(name)
    except Exception:
        return []
    names = sorted(set(names), reverse=True)
    if names:
        st.session_state.flash_models = names
    return names


def ask_gemini(key, history, user_text, lecture):
    client = get_gemini_client(key)
    contents = []
    for m in clean_history(history):
        role = "user" if m["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
    parts = []
    if lecture:
        parts.append(types.Part.from_bytes(data=lecture["bytes"], mime_type="application/pdf"))
    parts.append(types.Part(text=user_text))
    contents.append(types.Content(role="user", parts=parts))

    candidates = [GEMINI_MODEL] + [n for n in list_flash_models(client) if n != GEMINI_MODEL][:2]
    last_error = None
    attempts = []  # سجل كل محاولة وسبب فشلها (للتشخيص)
    for model_name in candidates:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        max_output_tokens=MAX_TOKENS,
                    ),
                )
                st.session_state.last_model = model_name
                return response.text or "مقدرتش أطلع رد، جرب تسأل بشكل تاني."
            except Exception as e:
                last_error = e
                attempts.append(f"{model_name} (محاولة {attempt + 1}): {str(e)[:300]}")
                try:
                    e.attempts = attempts
                except Exception:
                    pass
                low = str(e).lower()
                code = status_of(e)
                if code == 404 or "not_found" in low:
                    break  # الموديل مش متاح
                if any(w in low for w in TIMEOUT_WORDS):
                    break  # بطيء، جرب اللي بعده
                if code == 429 or "resource_exhausted" in low:
                    break  # الحصة خلصت على الموديل ده
                busy = code in (500, 502, 503, 504) or "unavailable" in low or any(w in low for w in CONN_WORDS)
                if not busy:
                    raise  # خطأ تاني (مفتاح غلط مثلاً)
                if attempt == 0:
                    time.sleep(2)
    raise last_error


# ----------------------------- التنسيق بين المحركات -----------------------------
def ask_model(providers, history, question, lecture):
    """providers: قائمة مرتبة [(اسم, مفتاح, موديل)]. لو الأول فشل بنجرب اللي بعده."""
    user_text, chunks = compose_user_text(question, history, lecture)
    errors = []
    for name, key, model in providers:
        try:
            if name == "Claude":
                answer = ask_claude(key, model, history, user_text, lecture)
            else:
                answer = ask_gemini(key, history, user_text, lecture)
            st.session_state.last_provider = name
            return answer, chunks, errors
        except Exception as e:
            errors.append((name, e))
    if len(errors) == 1:
        raise errors[0][1]
    raise ProviderError(" | ".join(f"{n}: {friendly_error(e)}" for n, e in errors)) from errors[0][1]


# ----------------------------- الواجهة -----------------------------
st.set_page_config(page_title="مساعد الكلية", page_icon="🎓")
st.markdown("<style>.stApp{direction:rtl;text-align:right;}</style>", unsafe_allow_html=True)
st.title("🎓 Nabeeh")
st.caption("اسألني عن الكلية والمحاضرات، أو عن أي حاجة تانية")

if "session_id" not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
if "history" not in st.session_state:
    st.session_state.history = []

summarize_clicked = False


def do_clear():
    """بيتنفذ كـ callback (قبل ما الصفحة تتبني تاني) عشان نقدر نغيّر حالة الـ checkbox."""
    clear_messages()
    st.session_state.history = []
    st.session_state.confirm_clear = False


with st.sidebar:
    st.header("⚙️ الإعدادات")

    # --- المفاتيح ---
    if not os.path.exists(CLAUDE_KEY_FILE) and not os.environ.get("ANTHROPIC_API_KEY"):
        st.text_input("مفتاح Claude (Anthropic)", type="password", key="claude_key_input")
    if not os.path.exists(GEMINI_KEY_FILE) and not os.environ.get("GEMINI_API_KEY"):
        st.text_input("مفتاح Gemini (اختياري)", type="password", key="gemini_key_input")

    claude_key = read_key(CLAUDE_KEY_FILE, "ANTHROPIC_API_KEY", "claude_key_input")
    gemini_key = read_key(GEMINI_KEY_FILE, "GEMINI_API_KEY", "gemini_key_input")
    has_claude = bool(claude_key) and anthropic is not None
    has_gemini = bool(gemini_key) and genai is not None
    if claude_key and anthropic is None:
        st.warning("مكتبة Claude مش متسطبة. شغّل: pip install anthropic")
    if gemini_key and genai is None:
        st.warning("مكتبة Gemini مش متسطبة. شغّل: pip install google-genai")

    # --- اختيار المحرك ---
    available = (["Claude"] if has_claude else []) + (["Gemini"] if has_gemini else [])
    claude_model = os.environ.get("CLAUDE_MODEL") or list(CLAUDE_MODELS.values())[0]
    if len(available) > 1:
        engine = st.radio("المحرك", available, horizontal=True)
    else:
        engine = available[0] if available else None
    if has_claude:
        labels = list(CLAUDE_MODELS)
        pick = st.selectbox("موديل Claude", labels, index=0)
        claude_model = CLAUDE_MODELS[pick]

    providers = []
    if engine:
        order = [engine] + [a for a in available if a != engine]
        for name in order:
            providers.append(("Claude", claude_key, claude_model) if name == "Claude"
                             else ("Gemini", gemini_key, GEMINI_MODEL))

    st.divider()
    st.metric("إجمالي الرسائل المخزنة", message_count())
    show_sources = st.checkbox("إظهار مصادر الكلية المستخدمة", value=False)
    if st.session_state.get("last_model"):
        st.caption(f"آخر رد: {st.session_state.get('last_provider', '')} / {st.session_state.last_model}")

    with st.expander("🗑️ تفريغ سجل الرسائل"):
        st.caption("بيمسح كل الرسائل المخزنة في قاعدة البيانات نهائياً.")
        confirm_clear = st.checkbox("أنا متأكد", key="confirm_clear")
        st.button("تفريغ الآن", disabled=not confirm_clear, on_click=do_clear)

    st.divider()
    st.subheader("📄 المحاضرات")
    uploaded = st.file_uploader("ارفع ملف محاضرة (PDF)", type=["pdf"])
    if uploaded is not None:
        pdf_bytes = uploaded.getvalue()
        if len(pdf_bytes) > MAX_PDF_MB * 1024 * 1024:
            st.error(f"الملف كبير. أقصى حجم {MAX_PDF_MB} ميجا.")
            st.session_state.pop("lecture", None)
        else:
            st.session_state.lecture = {"name": uploaded.name, "bytes": pdf_bytes}
    else:
        st.session_state.pop("lecture", None)
    if st.session_state.get("lecture"):
        st.success(f"المحاضرة جاهزة: {st.session_state.lecture['name']}")
        summarize_clicked = st.button("📝 لخّص المحاضرة")
    st.divider()
    if st.button("🆕 محادثة جديدة"):
        st.session_state.history = []
        st.session_state.session_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        st.rerun()

if not providers:
    st.info("حط مفتاح Claude أو Gemini في الشريط الجانبي (على اليمين) عشان تبدأ.")
    st.stop()

for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.write(m["content"])

typed_question = st.chat_input("اكتب سؤالك هنا...")
question = typed_question or (SUMMARY_QUESTION if summarize_clicked else None)
if question:
    with st.chat_message("user"):
        st.write(question)
    save_message(st.session_state.session_id, "user", question)

    with st.chat_message("assistant"):
        with st.spinner("بفكر..."):
            ok = True
            err_text = None
            try:
                answer, sources, _ = ask_model(
                    providers, st.session_state.history, question, st.session_state.get("lecture")
                )
            except Exception as e:  # أخطاء الشبكة أو المفتاح
                ok = False
                tried = "\n".join(getattr(e, "attempts", []) or [])
                err_text = (
                    f"المفتاح المستخدم بينتهي بـ: ...{(gemini_key or '')[-4:]}\n"
                    f"نوع الخطأ: {type(e).__name__}\n{str(e)[:600]}\n\n"
                    f"المحاولات:\n{tried}"
                )
                answer, sources = friendly_error(e), []
        if ok:
            st.write(answer)
        else:
            st.error(answer)
        if err_text:
            with st.expander("🔧 تفاصيل تقنية للخطأ (صوّرها وابعتها)", expanded=True):
                st.code(err_text[:2500], language=None)
        if show_sources and sources:
            with st.expander("مصادر الكلية"):
                for s in sources:
                    st.markdown(s)

    # رسايل الخطأ مابتتحطش في سجل المحادثة عشان ماتلخبطش الموديل في الأسئلة الجاية
    if ok:
        st.session_state.history.append({"role": "user", "content": question})
        st.session_state.history.append({"role": "assistant", "content": answer})
        save_message(st.session_state.session_id, "assistant", answer)
