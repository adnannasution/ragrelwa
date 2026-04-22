import os
import re
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# 1. LOAD CONFIGURATION
load_dotenv()
DATABASE_URL   = os.getenv("DATABASE_URL")
DINOIKI_API_KEY = os.getenv("DINOIKI_API_KEY")
FONNTE_TOKEN   = os.getenv("FONNTE_TOKEN")          # Token device dari dashboard Fonnte

# Whitelist nomor WA (format internasional tanpa +, pisahkan koma)
# Contoh isi di Railway: 6281234567890,6289876543210
ALLOWED_NUMBERS_RAW = os.getenv("ALLOWED_NUMBERS", "")
ALLOWED_NUMBERS = [n.strip() for n in ALLOWED_NUMBERS_RAW.split(",") if n.strip()]

# 2. SETUP AI ENGINE  (identik dengan versi Telegram)
WA_CUSTOM_PROMPT = """You are a PostgreSQL expert and a helpful AI Assistant.
Given an input question, create a syntactically correct PostgreSQL query to run.
HANYA BERIKAN QUERY SQL MURNI, TANPA MARKDOWN ATAU BACKTICK.

Setelah mendapatkan hasil dari database, berikan jawaban akhir dalam Bahasa Indonesia yang profesional.

ATURAN SQL:
- Selalu gunakan NULLIF(pembagi, 0) pada posisi penyebut untuk menghindari division by zero.
- WAJIB gunakan casting ::numeric untuk fungsi ROUND.
  Contoh: ROUND((hasil_perhitungan)::numeric, 2)
- Pastikan query kompatibel dengan PostgreSQL.

ATURAN FORMAT JAWABAN (KHUSUS WHATSAPP):
1. JAWABAN HARUS FULL NARASI: JANGAN gunakan tabel HTML atau format [CHART].
2. Gunakan poin-poin (•) jika data lebih dari satu agar tetap rapi di layar HP.
3. Tebalkan poin penting dengan *teks* (format bold WhatsApp).
4. Tambahkan emoticon yang relevan (📊, ✅, ⚠️) agar interaktif.

Table structure: {table_info}
Question: {input}"""

db       = SQLDatabase.from_uri(DATABASE_URL)
llm      = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=DINOIKI_API_KEY,
    base_url="https://ai.dinoiki.com/v1",
    temperature=0.7,
)
PROMPT   = PromptTemplate(input_variables=["input", "table_info"], template=WA_CUSTOM_PROMPT)
db_chain = SQLDatabaseChain.from_llm(llm, db, prompt=PROMPT, verbose=True, return_direct=False)

# 3. HELPER FUNCTIONS
def clean_response(text: str) -> str:
    text = re.sub(r'\[CHART\].*?\[/CHART\]', '', text, flags=re.DOTALL)
    text = re.sub(r'<table.*?>.*?</table>', '', text, flags=re.DOTALL)
    text = text.replace("```sql", "").replace("```", "").strip()
    return text

def send_wa(target: str, message: str) -> dict:
    """Kirim pesan WhatsApp via Fonnte API."""
    response = requests.post(
        "https://api.fonnte.com/send",
        headers={"Authorization": FONNTE_TOKEN},
        data={
            "target":  target,
            "message": message,
        },
        timeout=30,
    )
    return response.json()

# 4. FLASK APP & WEBHOOK ENDPOINT
app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def webhook():
    data    = request.get_json(force=True, silent=True) or {}

    sender  = data.get("sender", "")     # Nomor pengirim, contoh: 6281234567890
    message = data.get("message", "")    # Isi teks pesan
    # Field lain dari Fonnte yang tersedia jika dibutuhkan:
    # device, name, member (grup), text (button), location,
    # pollname, choices, timestamp, inboxid, url, filename, extension

    # PROTEKSI: Abaikan jika bukan nomor terdaftar
    if sender not in ALLOWED_NUMBERS:
        print(f"Akses ditolak untuk nomor: {sender}")
        return jsonify({"status": "ignored"}), 200

    # Abaikan pesan kosong
    if not message.strip():
        return jsonify({"status": "empty"}), 200

    try:
        response    = db_chain.invoke({"query": message})
        raw_answer  = response.get("result", response)
        final_answer = clean_response(raw_answer)
    except Exception as e:
        final_answer = f"⚠️ Kendala teknis: {str(e)}"

    send_wa(sender, final_answer)
    return jsonify({"status": "ok"}), 200

# Health-check endpoint (dipakai Railway untuk port detection)
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "WA Bot is running 🚀"}), 200

# 5. RUN
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"🚀 WA Webhook Bot berjalan di port {port}...")
    app.run(host="0.0.0.0", port=port)
