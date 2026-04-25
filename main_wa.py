import os
import re
import json
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# 1. LOAD CONFIGURATION
load_dotenv()
DATABASE_URL    = os.getenv("DATABASE_URL")
DINOIKI_API_KEY = os.getenv("DINOIKI_API_KEY")
FONNTE_TOKEN    = os.getenv("FONNTE_TOKEN")
PRISMA_URL      = os.getenv("PRISMA_URL", "")
CHATBOT_API_KEY = os.getenv("CHATBOT_API_KEY", "")
PRISMA_HEADERS  = {"x-chatbot-key": CHATBOT_API_KEY}

ALLOWED_NUMBERS_RAW = os.getenv("ALLOWED_NUMBERS", "")
ALLOWED_NUMBERS = [n.strip() for n in ALLOWED_NUMBERS_RAW.split(",") if n.strip()]

# 2. SETUP AI ENGINE
db = SQLDatabase.from_uri(DATABASE_URL, sample_rows_in_table_info=0)

llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=DINOIKI_API_KEY,
    base_url="https://ai.dinoiki.com/v1",
    temperature=0.7,
)

# 3. PRISMA INTEGRATION
def query_prisma(sql: str) -> dict:
    if not PRISMA_URL:
        return {"ok": False, "error": "PRISMA_URL belum dikonfigurasi"}
    try:
        r = requests.post(
            f"{PRISMA_URL}/chatbot/query",
            headers=PRISMA_HEADERS,
            json={"sql": sql},
            timeout=30
        )
        return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}

def fetch_prisma_schema() -> str:
    """
    Fetch schema PRISMA via information_schema — sama konsepnya
    dengan db.get_table_info() untuk database lokal.
    Dipanggil sekali saat startup.
    """
    if not PRISMA_URL:
        return ""
    try:
        sql = """
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name IN (
                'taex_reservasi', 'prisma_reservasi',
                'kumpulan_summary', 'sap_pr', 'sap_po', 'work_order'
              )
            ORDER BY table_name, ordinal_position
        """
        result = query_prisma(sql)
        if not result.get("ok"):
            print(f"[PRISMA SCHEMA] Gagal: {result.get('error')}")
            return ""

        lines = ["SCHEMA PRISMA TA-ex (query via query_prisma, BUKAN DB lokal):"]
        current_table = None
        for row in result.get("data", []):
            tbl   = row.get("table_name", "")
            col   = row.get("column_name", "")
            dtype = row.get("data_type", "")
            if tbl != current_table:
                current_table = tbl
                lines.append(f"\n  Tabel: {current_table}")
            col_display = f'"{col}"' if col == "order" else col
            lines.append(f"    - {col_display} ({dtype})")

        print("[PRISMA SCHEMA] Loaded OK")
        return "\n".join(lines)
    except Exception as e:
        print(f"[PRISMA SCHEMA] Exception: {e}")
        return ""

# Fetch schema PRISMA sekali saat startup (seperti get_table_info local)
PRISMA_SCHEMA = fetch_prisma_schema()

SYSTEM_PROMPT = """You are a PostgreSQL expert and a helpful AI Assistant for a refinery company.
Kamu memiliki memori percakapan — gunakan konteks dari pesan sebelumnya jika relevan.

DATABASE LOKAL:
{table_info}

{prisma_schema}

ATURAN ROUTING:
- Jika pertanyaan relevan dengan tabel PRISMA di atas → query ke PRISMA
- Jika relevan dengan tabel lokal → query database lokal
- JANGAN campur keduanya

ATURAN QUERY SQL:
- Pilih tabel yang paling relevan berdasarkan nama tabel dan kolom yang tersedia.
- Jika tabel relevan kosong, jawab: "Data belum tersedia."
- Kolom RU antar tabel mungkin berbeda format, gunakan ILIKE '%RU II%' saat JOIN.
- Selalu gunakan NULLIF(kolom_penyebut, 0) untuk menghindari division by zero.
- Gunakan ROUND(nilai::numeric, 2) untuk pembulatan.
- JANGAN query SELECT * tanpa LIMIT. Selalu agregasi, filter, atau LIMIT 20.
- Kolom "order" di tabel PRISMA WAJIB pakai tanda kutip ganda: "order"
- Untuk bad_actor_monitoring: kolom utama adalah ru, tag_number, status, problem, action_plan, progress, target_date.
- Untuk icu_monitoring: kolom utama adalah ru, icu_status (Medium/High/Critical/Low), tag_no, issue, mitigation, permanent_solution, progress, target_closed, report_date.
- Untuk anggaran_maintenance: kolom nilai_usd adalah nilai dalam USD. Selalu tampilkan dengan format USD dan pemisah ribuan, contoh: 1,234,567.89 USD.
- Untuk tkdn: kolom nominal dan kdn adalah nilai dalam IDR. Selalu tampilkan dengan format Rp dan pemisah ribuan.
- Untuk program_kerja_atg: kolom refinery_unit, type, atg_eksisting, program_2024, prokja, action_plan_category, target, month_update.
- Untuk paf: Plant Availability Factor — kolom type, ru, target_realisasi, value, plan_unplan, month.
- Untuk zero_clamp: kolom ru, area, unit, tag_no_ln, type_damage, type_perbaikan, status, tanggal_dipasang, tanggal_rencana_perbaikan.
- Untuk issue_paf: kolom type (Primary/Secondary Unit), ru, date, issue.
- Untuk power_stream: kolom refinery_unit, type_equipment, equipment, status_operation, desain, kapasitas_max, average_actual.
- Untuk jumlah_eqp_utl: kolom refinery_unit, type_equipment, status_equipment, jumlah.
- Untuk critical_eqp_utl: kolom refinery_unit, type_equipment, highlight_issue, corrective_action, mitigasi_action, target_corrective.
- Untuk critical_eqp_prim_sec: kolom refinery_unit, unit_proses, equipment, highlight_issue, corrective_action, mitigasi_action.
- Untuk monitoring_operasi: kolom refinery_unit, unit_proses, unit, design, minimal_capacity, plant_readiness, actual, target_sts.
- Untuk inspection_plan: kolom refinery_unit, area, tag_no_ln, type_equipment, type_inspection, due_date, plan_date, actual_date, result_remaining_life, grand_result.
- Untuk tkdn: Tingkat Kandungan Dalam Negeri — kolom refinery_unit, bulan, nominal (IDR), kdn (IDR), persentase (%), tahun.
- Untuk rcps_rekomendasi: kolom kilang, rcps_no, judul_rcps, rekomendasi, traffic, pic, target, remark.
- Untuk rcps: kolom kilang, traffic, sum_of_progress, disiplin, judul_rcps, rcps_no, criticallity.
- Untuk boc: Basis of Comparison — kolom ru, area, unit, equipment, status, frequency, running_hours, mttr, mtbf, hasil.
- Untuk readiness_jetty: kolom refinery_unit, tag_no, status_operation, status_tuks, expired_tuks, status_ijin_ops, status_isps, status_struktur, status_trestle, status_mla, status_fire_protection, month_update.
- Untuk workplan_jetty: kolom refinery_unit, tag_no, item, status_item, remark, rtl_action_plan, target, status_rtl, month_update.
- Untuk readiness_tank: kolom refinery_unit, tag_number, type_tangki, service_tangki, prioritas, status_operational, atg_certification_validity, status_coi, status_atg, status_grounding, status_shell_course, status_roof, status_cathodic, month_update.
- Untuk workplan_tank: kolom unit, tag_no, item, remark, rtl_action_plan, target, status_rtl, month_update.
- Untuk readiness_spm: kolom refinery_unit, tag_no, status_operation, status_laik_operasi, expired_laik_operasi, status_ijin_spl, status_mbc, status_lds, status_mooring_hawser, status_floating_hose, status_cathodic_spl, month_update.
- Untuk spm_workplan: kolom refinery_unit, tag_no, item, remark, rtl_action_plan, target, status_rtl, month_update.
- Untuk status procurement PRISMA (join taex_reservasi + sap_po ON sap_po.purchreq = taex_reservasi.pr):
    no-pr      : pr IS NULL atau pr = ''
    pr-created : pr ada, belum ada PO
    po-created : PO ada, qty_delivered = 0
    partial    : qty_delivered > 0 tapi < po_quantity
    complete   : qty_delivered >= po_quantity

ATURAN FORMAT JAWABAN (KHUSUS WHATSAPP — NARASI SAJA):
1. JAWABAN FULL NARASI — JANGAN gunakan tabel HTML, JANGAN format [CHART].
2. Jika hasil lebih dari 10 item, tampilkan ringkasan/highlight saja.
3. Gunakan poin-poin (•) jika data lebih dari satu.
4. Tebalkan poin penting dengan *teks* (bold WhatsApp).
5. Tambahkan emoticon relevan (🏭, 💰, 📊, ✅, ⚠️, 🔧, 🛢️, 🚨).
6. Maksimal 5 poin per jawaban agar tidak terlalu panjang di layar HP."""

# 4. MEMORY PER NOMOR WA
MAX_HISTORY = 10
wa_histories: dict[str, list] = {}

def get_history(number: str) -> list:
    return wa_histories.get(number, [])

def add_history(number: str, question: str, answer: str):
    history = wa_histories.get(number, [])
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
    wa_histories[number] = history

def clear_history(number: str):
    wa_histories.pop(number, None)

# 5. PRE-FILTER
OUT_OF_SCOPE = [
    "cuaca", "berita", "news", "resep", "masak", "film", "musik",
    "olahraga", "politik", "saham", "crypto", "bitcoin", "translate",
    "siapa presiden", "capital of", "ibukota",
]
DUMP_KEYWORDS = [
    "tampilkan semua", "lihat semua", "show all", "list semua",
    "seluruh isi", "semua baris", "semua data", "semua record",
    "export semua", "ceritakan semua",
]

# 6. CORE AI FUNCTION
def run_wa(question: str, sender: str) -> str:
    q_lower = question.lower()

    # Pre-filter out of scope
    if any(k in q_lower for k in OUT_OF_SCOPE):
        return ("⚠️ Maaf, saya hanya dapat membantu analisis data maintenance kilang.\n"
                "Silakan ajukan pertanyaan yang berkaitan dengan data yang tersedia.")

    # Pre-filter dump
    if any(k in q_lower for k in DUMP_KEYWORDS):
        return ("📊 Menampilkan semua data dalam chat kurang efisien.\n\n"
                "Coba persempit pertanyaan:\n"
                "• Berapa jumlah per RU?\n"
                "• Mana yang statusnya bermasalah?\n"
                "• Mana yang sudah melewati target date?\n\n"
                "💡 Untuk data lengkap, minta admin download via web.")

    history    = get_history(sender)
    table_info = db.get_table_info()

    # Build messages dengan history
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(
        table_info=table_info,
        prisma_schema=PRISMA_SCHEMA if PRISMA_SCHEMA else "(PRISMA tidak tersedia)",
    )}]
    for msg in history:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})

    # Routing: tanya LLM berdasarkan schema — tidak pakai keyword lagi
    is_prisma = False
    if PRISMA_URL and PRISMA_SCHEMA:
        try:
            routing = llm.invoke(messages + [{"role": "user", "content": (
                f"Berdasarkan schema di atas, pertanyaan berikut perlu query ke tabel mana?\n"
                f"Pertanyaan: '{question}'\n"
                f"Jawab HANYA satu kata: 'PRISMA' atau 'LOCAL'"
            )}])
            is_prisma = "PRISMA" in routing.content.strip().upper()
            print(f"[ROUTING] {'PRISMA' if is_prisma else 'LOCAL'} ← '{question[:60]}'")
        except Exception as e:
            print(f"[ROUTING ERROR] {e} — fallback LOCAL")

    if is_prisma:
        # Generate SQL untuk PRISMA
        sql_messages = messages + [{"role": "user", "content": (
            f"Berikan HANYA query SQL PostgreSQL untuk tabel PRISMA TA-ex "
            f"(taex_reservasi, prisma_reservasi, kumpulan_summary, sap_pr, sap_po, work_order). "
            f"Kolom 'order' WAJIB pakai tanda kutip ganda. LIMIT 50. SQL murni saja.\n"
            f"Pertanyaan: {question}"
        )}]
        sql_resp  = llm.invoke(sql_messages)
        sql_query = sql_resp.content.replace("```sql", "").replace("```", "").strip()
        print(f"[PRISMA SQL] {sql_query}")

        prisma_result = query_prisma(sql_query)
        if prisma_result.get("ok"):
            db_result = f"Hasil PRISMA ({prisma_result.get('rows', 0)} baris):\n{prisma_result.get('data', [])}"
        else:
            err = prisma_result.get('error')
            print(f"[PRISMA ERROR] {err} | SQL: {sql_query}")
            db_result = f"Query PRISMA gagal: {err}"
    else:
        # Generate SQL untuk DB lokal
        sql_messages = messages + [{"role": "user", "content": (
            f"Berikan HANYA query SQL PostgreSQL yang valid untuk: {question}. "
            f"Tanpa penjelasan, tanpa markdown."
        )}]
        sql_resp  = llm.invoke(sql_messages)
        sql_query = sql_resp.content.replace("```sql", "").replace("```", "").strip()
        print(f"[LOCAL SQL] {sql_query}")

        try:
            db_result = db.run(sql_query)
        except Exception as e:
            db_result = f"Query error: {str(e)}"

    # Generate jawaban final
    answer_messages = messages + [
        {"role": "user", "content": question},
        {"role": "user", "content": (
            f"Hasil query:\n{db_result}\n\n"
            f"Berikan jawaban final dalam Bahasa Indonesia sesuai aturan format WhatsApp. "
            f"Ingat: narasi saja, tidak ada tabel HTML atau [CHART]."
        )}
    ]
    final  = llm.invoke(answer_messages)
    answer = final.content.replace("```sql", "").replace("```", "").strip()
    answer = re.sub(r'\[CHART\].*?\[/CHART\]', '', answer, flags=re.DOTALL)
    answer = re.sub(r'<table.*?>.*?</table>', '', answer, flags=re.DOTALL)
    answer = re.sub(r'\[DOWNLOAD:\w+\]', '', answer).strip()

    # Simpan ke history
    if answer and "⚠️ Maaf" not in answer:
        add_history(sender, question, answer)

    return answer

# 7. HELPER
def send_wa(target: str, message: str) -> dict:
    response = requests.post(
        "https://api.fonnte.com/send",
        headers={"Authorization": FONNTE_TOKEN},
        data={"target": target, "message": message},
        timeout=30,
    )
    return response.json()

# 8. FLASK APP
app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def webhook():
    data    = request.get_json(force=True, silent=True) or {}
    sender  = data.get("sender", "")
    message = data.get("message", "").strip()

    if sender not in ALLOWED_NUMBERS:
        print(f"Akses ditolak: {sender}")
        return jsonify({"status": "ignored"}), 200

    if not message:
        return jsonify({"status": "empty"}), 200

    # Command reset history
    if message.lower() in ["/reset", "reset", ".reset"]:
        clear_history(sender)
        send_wa(sender, "🔄 *Percakapan direset.* Memori sesi sebelumnya dihapus.")
        return jsonify({"status": "ok"}), 200

    # Command reload schema PRISMA manual
    if message.lower() in ["/reload", ".reload"]:
        global PRISMA_SCHEMA
        PRISMA_SCHEMA = fetch_prisma_schema()
        status = "✅ Schema PRISMA berhasil di-reload." if PRISMA_SCHEMA else "⚠️ Gagal reload schema PRISMA."
        send_wa(sender, status)
        return jsonify({"status": "ok"}), 200

    answer = run_wa(message, sender)
    send_wa(sender, answer)
    return jsonify({"status": "ok"}), 200

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "WA Bot is running 🚀",
        "prisma_schema_loaded": bool(PRISMA_SCHEMA),
    }), 200

# 9. RUN
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"🚀 WA Bot berjalan di port {port}...")
    print(f"📋 PRISMA Schema: {'✅ loaded' if PRISMA_SCHEMA else '❌ not available'}")
    app.run(host="0.0.0.0", port=port)