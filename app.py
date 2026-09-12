import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

# ====================================================
# ۰. تنظیمات اولیه
# ====================================================
# مسیر .env را صریحاً کنار همین فایل app.py مشخص می‌کنیم تا مستقل از اینکه
# استریم‌لیت از کدام پوشه اجرا شده، فایل .env پیدا شود.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

if not GROQ_API_KEY or not SERPER_API_KEY:
    st.set_page_config(page_title="خطای تنظیمات", page_icon="⚠️")
    st.error(
        "کلید(های) API پیدا نشد. لطفاً یک فایل `.env` دقیقاً در همان پوشه‌ی "
        "`app.py` بسازید و مقادیر زیر را در آن قرار دهید:\n\n"
        "```\nGROQ_API_KEY=...\nSERPER_API_KEY=...\n```\n\n"
        "سپس برنامه را دوباره اجرا کنید."
    )
    st.stop()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    groq_api_key=GROQ_API_KEY,
    max_tokens=8000,  # جلوگیری از قطع‌شدن گزارش‌های بلند و چندبخشی وسط راه
)

# کلید را مستقیم به ولیدیتور pydantic پاس می‌دهیم تا وابسته به خواندن
# متغیر محیطی توسط خودِ کتابخانه نباشیم (همان چیزی که خطای قبلی را ایجاد می‌کرد).
serper = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)


# ====================================================
# ۱. ابزار جستجوی زنده گوگل (Serper API)
# ====================================================
@tool
def google_search(query: str) -> str:
    """جستجوی زنده در نتایج واقعی گوگل با Serper API و بازگرداندن مهم‌ترین نتایج (عنوان، خلاصه و منبع)."""
    try:
        data = serper.results(query)
        organic = data.get("organic", [])[:5]
        if not organic:
            return "نتیجه‌ای یافت نشد."

        formatted = []
        # جعبه پاسخ سریع گوگل (در صورت وجود) اولویت بالایی برای آمار دقیق دارد
        answer_box = data.get("answerBox")
        if answer_box:
            formatted.append(
                f"پاسخ سریع گوگل: {answer_box.get('title', '')} — {answer_box.get('answer') or answer_box.get('snippet', '')}\n---"
            )

        for r in organic:
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            link = r.get("link", "")
            formatted.append(f"عنوان: {title}\nخلاصه: {snippet}\nمنبع: {link}\n---")

        return "\n".join(formatted)
    except Exception as e:
        return f"خطا در جستجو: {e}"


# ====================================================
# ۲. معماری دو ایجنت
# ====================================================
def build_search_query(topic: str) -> str:
    """موضوع کاربر را به یک کوئری کوتاه و تمیز مناسب گوگل تبدیل می‌کند."""
    prompt = f"""موضوع زیر را به یک عبارت جستجوی کوتاه، دقیق و مناسب گوگل (حداکثر ۸ کلمه) تبدیل کن.
فقط عبارت جستجو را برگردان، بدون هیچ توضیح اضافه یا علامت نقل‌قول.

موضوع: {topic}
"""
    res = llm.invoke([HumanMessage(content=prompt)])
    return res.content.strip().strip('"').strip("«»")


def researcher_agent(topic: str) -> str:
    """ایجنت ۱: محقق گوگل — کوئری می‌سازد، سرچ می‌کند و یادداشت تحلیلی مستند تولید می‌کند."""
    query = build_search_query(topic)
    search_data = google_search.invoke(query)

    prompt = f"""شما محقق ارشد هستید. بر اساس نتایج جستجوی گوگل زیر درباره «{topic}»، مهم‌ترین
آمارها، ارقام، رویدادها، منابع و نقل‌قول‌های کلیدی را استخراج کنید و در قالب یک یادداشت
تحلیلی دقیق و مستند جمع‌بندی کنید. برای هر نکته‌ی مهم، منبع آن را ذکر کنید.

کوئری جستجوی استفاده‌شده: {query}

نتایج خام گوگل:
{search_data}
"""
    res = llm.invoke([HumanMessage(content=prompt)])
    return res.content


def editor_agent(topic: str, notes: str, feedback: str = "") -> str:
    """ایجنت ۲: سردبیر و منتقد ارشد — گزارش راهبردی ساختاریافته تدوین می‌کند."""
    prompt = f"""شما سردبیر ارشد و منتقد یک نشریه تخصصی اقتصادی/فناوری هستید.
بر اساس یادداشت‌های محقق زیر، یک گزارش راهبردی رسمی درباره «{topic}» دقیقاً و **فقط** با این
چهار بخش تدوین کنید (بخش اضافه‌ای مثل مرور تاریخی، روش‌شناسی جداگانه یا چند جدول پشت‌سرهم اضافه نکنید):

## چکیده اجرایی
## جدول سناریوها و آمار کلیدی
## تحلیل روندها
## نتیجه‌گیری و پیشنهاد راهبردی

نکات مهم:
- گزارش را مختصر و فشرده نگه دار؛ هر بخش حداکثر چند پاراگراف یا یک جدول کوتاه باشد.
- جدول را با سینتکس استاندارد مارک‌داون بنویس و آن را حداکثر در ۵-۶ ردیف نگه دار.
- هر سلول جدول باید کوتاه و تک‌خطی باشد؛ هرگز داخل جدول از تگ HTML (مثل <br>) یا بک‌تیک
  استفاده نکن. اگر چند نکته برای یک سلول داری، آن‌ها را با ویرگول یا نقطه‌ویرگول در یک خط بنویس.
- مهم‌ترین قانون: گزارش باید همیشه کامل و با نتیجه‌گیری واقعی پایان یابد؛ هرگز وسط یک جمله،
  جدول یا فرمت مارک‌داون (مثل ** باز بدون بسته شدن) قطع نشود.

یادداشت‌های محقق:
{notes}

نظرات و اصلاحات ناظر انسانی (در صورت وجود، حتماً اعمال شود):
{feedback}
"""
    res = llm.invoke([HumanMessage(content=prompt)])
    return res.content


def _looks_truncated(text: str) -> bool:
    """تشخیص ساده‌ی گزارش‌های ناقص‌مانده (مثل ** باز نشده یا جدول نصفه)."""
    stripped = text.strip()
    if not stripped:
        return True
    if stripped.count("**") % 2 != 0:
        return True
    if stripped.endswith(("|", "-", "**", "،", ":")):
        return True
    return False


# ====================================================
# ۳. خروجی سند شرکتی (HTML تیره و راست‌چین)
# ====================================================
def _sanitize_markdown(text: str) -> str:
    """رفع مشکل رایج تگ‌های <br> که به‌اشتباه داخل بک‌تیک نوشته می‌شوند و به‌جای
    شکستن خط، به‌صورت متن خام «<br>» روی صفحه نمایش داده می‌شوند."""
    text = re.sub(r"`\s*<br\s*/?>\s*`", "<br>", text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "<br>", text, flags=re.IGNORECASE)
    return text


def generate_executive_html(report_text: str) -> str:
    try:
        import markdown

        clean_text = _sanitize_markdown(report_text)
        html_body = markdown.markdown(clean_text, extensions=["tables", "fenced_code"])
    except ImportError:
        html_body = f"<pre style='white-space: pre-wrap;'>{report_text}</pre>"

    return f"""
    <!DOCTYPE html>
    <html lang="fa" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <title>گزارش هوش مصنوعی چندعاملی</title>
        <style>
            * {{ box-sizing: border-box; }}
            body {{
                font-family: 'Tahoma', sans-serif;
                background-color: #0f172a;
                color: #e2e8f0;
                padding: 40px;
                unicode-bidi: plaintext;
            }}
            .card {{
                max-width: 950px;
                margin: auto;
                background: #1e293b;
                padding: 45px;
                border-radius: 14px;
                box-shadow: 0 8px 24px rgba(0,0,0,0.4);
                line-height: 1.9;
                border: 1px solid #334155;
                font-size: 16px;
            }}
            h1 {{ font-size: 26px; }}
            h2 {{ font-size: 21px; }}
            h3 {{ font-size: 18px; }}
            h1, h2, h3 {{
                color: #93c5fd;
                border-bottom: 2px solid #334155;
                padding-bottom: 10px;
                margin-top: 28px;
                line-height: 1.6;
            }}
            p, li {{ color: #cbd5e1; font-size: 15px; }}
            .table-wrap {{
                overflow-x: auto;
                margin: 25px 0;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                table-layout: fixed;
                font-size: 14px;
            }}
            th, td {{
                border: 1px solid #334155;
                padding: 12px 14px;
                text-align: right;
                vertical-align: top;
                word-wrap: break-word;
                overflow-wrap: break-word;
                unicode-bidi: plaintext;
            }}
            th {{
                background-color: #273449;
                font-weight: bold;
                color: #93c5fd;
            }}
            tr:nth-child(even) {{ background-color: #1a2537; }}
            code {{
                background-color: #0f172a;
                color: #7dd3fc;
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 13px;
            }}
            .badge {{
                background: #2563eb;
                color: white;
                padding: 6px 14px;
                border-radius: 20px;
                font-size: 13px;
                font-weight: bold;
            }}
            .footer {{
                margin-top: 30px;
                font-size: 12px;
                color: #64748b;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <span class="badge">گزارش تحلیلی هوش مصنوعی (Multi-Agent)</span>
            <div class="table-wrap">
                {html_body}
            </div>
            <div class="footer">تولید شده به‌صورت خودکار توسط سیستم چندعاملی | تمام حقوق محفوظ است</div>
        </div>
    </body>
    </html>
    """


# ====================================================
# ۴. رابط کاربری استریم‌لیت (RTL و تیره)
# ====================================================
st.set_page_config(page_title="اتاق فکر چندعاملی", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background-color: #0f172a;
        }
        .stApp, .stMarkdown, p, h1, h2, h3, h4, h5, h6, span, label, div {
            direction: rtl !important;
            text-align: right !important;
            font-family: 'Tahoma', 'Vazirmatn', sans-serif !important;
        }
        .stTextInput input, .stTextArea textarea {
            direction: rtl !important;
            text-align: right !important;
            background-color: #1e293b !important;
            color: #e2e8f0 !important;
            border: 1px solid #334155 !important;
        }
        .stButton button {
            width: 100%;
            border-radius: 8px;
        }
        div[data-testid="stTable"] table {
            direction: rtl !important;
            text-align: right !important;
        }
        .stAlert {
            direction: rtl !important;
            text-align: right !important;
        }
        section[data-testid="stSidebar"] { direction: rtl !important; }

        /* رفع فاصله‌ی نامتقارن بین نشانه‌ی لیست (•) و متن در حالت راست‌چین */
        .stApp ul, .stApp ol {
            padding-right: 1.4em !important;
            padding-left: 0 !important;
            margin-right: 0 !important;
        }
        .stApp li {
            margin-bottom: 6px;
            padding-right: 0.2em;
        }

        /* فاصله‌ی بیشتر بین دو ستون اصلی صفحه، به‌همراه یک خط جداکننده‌ی ظریف */
        div[data-testid="stHorizontalBlock"] {
            gap: 3rem;
        }
        div[data-testid="column"] {
            background-color: #141f38;
            border-radius: 12px;
            padding: 20px 24px;
            border: 1px solid #253148;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 اتاق فکر چندعاملی (Multi-Agent Research Lab)")
st.caption("سیستم مشارکتی ۲ ایجنت مستقل (محقق گوگل + سردبیر) همراه با تایید ناظر انسانی (Human-in-the-Loop)")

# --- حافظه سشن ---
if "topic" not in st.session_state:
    st.session_state.topic = ""
if "notes" not in st.session_state:
    st.session_state.notes = ""
if "draft" not in st.session_state:
    st.session_state.draft = ""


col1, col2 = st.columns([1, 1], gap="large")

# ستون ۱: ایجنت محقق
with col1:
    st.subheader("🔎 ۱. ایجنت محقق گوگل (Live Google Researcher)")
    topic_input = st.text_input(
        "موضوع گزارش را وارد کنید:",
        placeholder="مثال: وضعیت ثبت‌نام و قیمت جدید خودروهای وارداتی",
    )

    if st.button("🚀 فعال‌سازی ایجنت محقق", type="primary") and topic_input:
        st.session_state.topic = topic_input
        with st.spinner("ایجنت محقق در حال سرچ گوگل و جمع‌آوری داده‌ها..."):
            st.session_state.notes = researcher_agent(topic_input)
            st.session_state.draft = editor_agent(topic_input, st.session_state.notes)
        st.rerun()

    if st.session_state.notes:
        st.success("✅ یادداشت‌های اولیه محقق آماده شد:")
        st.info(st.session_state.notes)

# ستون ۲: ایجنت سردبیر + تایید انسانی
with col2:
    st.subheader("🧐 ۲. ایجنت سردبیر و منتقد ارشد (Editor Agent)")

    if st.session_state.draft:
        if _looks_truncated(st.session_state.draft):
            st.warning(
                "⚠️ به‌نظر می‌رسد گزارش وسط راه قطع شده (مثلاً یک جدول یا جمله نصفه مونده). "
                "پیشنهاد می‌شه دکمه‌ی «بازنویسی توسط سردبیر» رو بدون فیدبک خاصی بزنی تا دوباره کامل تولید بشه."
            )
        st.markdown("### 📄 پیش‌نویس گزارش تدوین‌شده:")
        st.write(st.session_state.draft)

        st.markdown("---")
        st.subheader("👤 نظارت انسانی (Human-in-the-Loop)")
        feedback = st.text_input(
            "دستور بازنویسی یا نکته تکمیلی (اختیاری):",
            placeholder="مثلاً: جدول آمار را دقیق‌تر کن یا نتیجه‌گیری اضافه کن",
        )

        if st.button("🔄 بازنویسی توسط سردبیر"):
            with st.spinner("سردبیر در حال اعمال نظرات شما..."):
                st.session_state.draft = editor_agent(
                    st.session_state.topic, st.session_state.notes, feedback
                )
            st.rerun()

        html_report = generate_executive_html(st.session_state.draft)
        st.download_button(
            label="📥 دانلود گزارش رسمی (HTML)",
            data=html_report,
            file_name=f"Report_{st.session_state.topic[:15]}.html",
            mime="text/html",
        )
