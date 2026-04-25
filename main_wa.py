import os
import re
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# ─── 1. LOAD CONFIGURATION ────────────────────────────────────────────────────
load_dotenv()
DATABASE_URL    = os.getenv("DATABASE_URL")
DINOIKI_API_KEY = os.getenv("DINOIKI_API_KEY")
FONNTE_TOKEN    = os.getenv("FONNTE_TOKEN")
PRISMA_URL      = os.getenv("PRISMA_URL", "")
CHATBOT_API_KEY = os.getenv("CHATBOT_API_KEY", "")
PRISMA_HEADERS  = {"x-chatbot-key": CHATBOT_API_KEY}

ALLOWED_NUMBERS_RAW = os.getenv("ALLOWED_NUMBERS", "")
ALLOWED_NUMBERS = [n.strip() for n in ALLOWED_NUMBERS_RAW.split(",") if n.strip()]

# ─── 2. SETUP AI ENGINE ───────────────────────────────────────────────────────
db_engine = SQLDatabase.from_uri(DATABASE_URL, sample_rows_in_table_info=0)

llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=DINOIKI_API_KEY,
    base_url="https://ai.dinoiki.com/v1",
    temperature=0.7,
)

# ─── 3. PRISMA INTEGRATION ────────────────────────────────────────────────────
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

# ─── 4. SYSTEM PROMPT — sama persis dengan web, tambah aturan WA ──────────────
# {table_info} dan {input} diisi saat runtime
CUSTOM_PROMPT = """You are a PostgreSQL expert and a helpful AI Assistant for a refinery company.
Given an input question, create a syntactically correct PostgreSQL query to run.
HANYA BERIKAN QUERY SQL MURNI, TANPA MARKDOWN ATAU BACKTICK.

Setelah mendapatkan hasil dari database, berikan jawaban akhir dalam Bahasa Indonesia yang profesional.

STRUKTUR TABEL TERSEDIA:
{table_info}

ATURAN QUERY SQL:
- Pilih tabel yang paling relevan berdasarkan nama tabel dan kolom yang tersedia.
- Jika tabel relevan kosong, jawab: "Data belum tersedia, silakan upload datanya terlebih dahulu."
- Jangan query tabel yang tidak relevan dengan pertanyaan.
- Kolom RU antar tabel mungkin berbeda format, gunakan ILIKE '%RU II%' saat JOIN.
- Selalu gunakan NULLIF(kolom_penyebut, 0) untuk menghindari division by zero.
- Gunakan ROUND(nilai::numeric, 2) untuk pembulatan.
- Jika pertanyaan melibatkan lebih dari satu tabel, gunakan JOIN yang sesuai.
- PENTING: Jangan pernah query SELECT * tanpa LIMIT. Selalu gunakan agregasi, filter, atau LIMIT 20.
- DETEKSI PERTANYAAN TIDAK PRODUKTIF: Jika user meminta salah satu dari berikut, JANGAN query — langsung tolak dengan sopan:
  * "tampilkan semua", "list semua", "show all", "lihat semua", "ceritakan semua"
  * "tampilkan seluruh isi tabel", "dump data", "export semua"
  * Pertanyaan di luar konteks maintenance kilang (cuaca, berita, pengetahuan umum, coding, dll)
  Untuk pertanyaan di luar konteks → jawab: "Maaf, saya hanya dapat membantu analisis data maintenance kilang."
- Untuk icu_monitoring: kolom utama adalah ru, icu_status (Medium/High/Critical/Low), tag_no, issue, mitigation, permanent_solution, progress, target_closed, report_date.
- Untuk program_kerja_atg: kolom utama adalah refinery_unit, type, atg_eksisting, program_2024, prokja (progress), action_plan_category, target, month_update.
- Untuk paf: Plant Availability Factor — kolom type, ru, target_realisasi, value (angka PAF), plan_unplan, month.
- Untuk zero_clamp: monitoring temporary repair zero clamp — kolom ru, area, unit, tag_no_ln, type_damage, type_perbaikan, status, tanggal_dipasang, tanggal_rencana_perbaikan.
- Untuk issue_paf: daftar issue yang mempengaruhi PAF — kolom type (Primary/Secondary Unit), ru, date, issue.
- Untuk power_stream: status operasi equipment power & steam — kolom refinery_unit, type_equipment, equipment, status_operation, desain, kapasitas_max, average_actual.
- Untuk jumlah_eqp_utl: jumlah equipment utility per status — kolom refinery_unit, type_equipment, status_equipment, jumlah.
- Untuk critical_eqp_utl: critical equipment utility — kolom refinery_unit, type_equipment, highlight_issue, corrective_action, mitigasi_action, target_corrective.
- Untuk critical_eqp_prim_sec: critical equipment primary & secondary — kolom refinery_unit, unit_proses, equipment, highlight_issue, corrective_action, mitigasi_action.
- Untuk monitoring_operasi: monitoring kapasitas operasi unit proses — kolom refinery_unit, unit_proses, unit, design, minimal_capacity, plant_readiness, actual, target_sts.
- Untuk inspection_plan: rencana & realisasi inspeksi equipment — kolom refinery_unit, area, tag_no_ln, type_equipment, type_inspection, due_date, plan_date, actual_date, result_remaining_life, grand_result.
- Untuk tkdn: Tingkat Kandungan Dalam Negeri — kolom refinery_unit, bulan, nominal (IDR), kdn (IDR), persentase (%), tahun. Selalu tampilkan nominal dan kdn dengan format Rp dan pemisah ribuan.
- Untuk anggaran_maintenance: kolom ru, tahun, kategori, tipe, nilai_usd (USD). Selalu tampilkan nilai_usd dengan format USD dan pemisah ribuan, contoh: 1,234,567.89 USD.
- Untuk rcps_rekomendasi: rekomendasi dari RCPS — kolom kilang, rcps_no, judul_rcps, rekomendasi, traffic, pic, target, remark.
- Untuk rcps: daftar RCPS — kolom kilang, traffic, sum_of_progress, disiplin, judul_rcps, rcps_no, criticallity.
- Untuk boc: Basis of Comparison equipment — kolom ru, area, unit, equipment, status, frequency, running_hours, mttr, mtbf, hasil.
- Untuk readiness_jetty: kesiapan operasional jetty — kolom refinery_unit, tag_no, status_operation, status_tuks, expired_tuks, status_ijin_ops, status_isps, status_struktur, status_trestle, status_mla, status_fire_protection, month_update.
- Untuk workplan_jetty: workplan perbaikan item jetty — kolom refinery_unit, tag_no, item, status_item, remark, rtl_action_plan, target, status_rtl, month_update.
- Untuk readiness_tank: kesiapan operasional tangki — kolom refinery_unit, tag_number, type_tangki, service_tangki, prioritas, status_operational, atg_certification_validity, status_coi, status_atg, status_grounding, status_shell_course, status_roof, status_cathodic, month_update.
- Untuk workplan_tank: workplan perbaikan tangki — kolom unit, tag_no, item, remark, rtl_action_plan, target, status_rtl, month_update.
- Untuk readiness_spm: kesiapan operasional SPM — kolom refinery_unit, tag_no, status_operation, status_laik_operasi, expired_laik_operasi, status_ijin_spl, status_mbc, status_lds, status_mooring_hawser, status_floating_hose, status_cathodic_spl, month_update.
- Untuk spm_workplan: workplan perbaikan SPM — kolom refinery_unit, tag_no, item, remark, rtl_action_plan, target, status_rtl, month_update.

TABEL EKSTERNAL PRISMA TA-ex (data procurement material Turnaround):
Untuk pertanyaan tentang material TA, reservasi, PR, PO, work order turnaround — gunakan query_prisma(sql).
Tabel yang tersedia di sistem PRISMA (BUKAN di database lokal ini):
- taex_reservasi: reservasi material utama TA-ex
  kolom: plant, equipment, "order" (pakai tanda kutip!), reservno, material, material_description,
         qty_reqmts, qty_stock, pr, item, qty_pr, del, fis, ict, pg, reqmts_date, uom, res_price, res_curr
- prisma_reservasi: subset taex aktif (ict=L)
  kolom: plant, equipment, "order", material, qty_reqmts, qty_stock_onhand,
         pr_prisma, qty_pr_prisma, code_kertas_kerja
- kumpulan_summary: ringkasan kebutuhan material per kertas kerja
  kolom: material, material_description, qty_req, qty_stock, qty_pr, qty_to_pr, code_tracking
- sap_pr: Purchase Request dari SAP
  kolom: plant, pr, material, material_description, qty_pr, req_date, release_date, tracking_no
- sap_po: Purchase Order dari SAP
  kolom: plnt, purchreq (=nomor PR), material, po, po_quantity, qty_delivered, deliv_date, net_price, crcy
- work_order: Work Order dari SAP
  kolom: plant, "order", equipment, description, system_status, planner_group,
         basic_start_date, basic_finish_date, total_plan_cost, total_act_cost

STATUS PROCUREMENT (join taex + sap_po ON sap_po.purchreq = taex_reservasi.pr):
- no-pr:      pr IS NULL atau pr = ''
- pr-created: pr ada, belum ada PO
- po-created: PO ada, qty_delivered = 0
- partial:    qty_delivered > 0 tapi < po_quantity
- complete:   qty_delivered >= po_quantity

ATURAN QUERY PRISMA:
- Kolom "order" WAJIB ditulis dengan tanda kutip ganda: "order"
- Selalu gunakan LIMIT maksimal 50
- Keyword PRISMA: turnaround, TA, material, reservasi, PR, PO, kertas kerja, work order TA
- JANGAN query tabel PRISMA ke database lokal — gunakan query_prisma()

ATURAN FORMAT JAWABAN (KHUSUS WHATSAPP — NARASI SAJA):
1. JAWABAN FULL NARASI — JANGAN gunakan tabel HTML, JANGAN format [CHART].
2. Jika hasil lebih dari 10 item, tampilkan ringkasan/highlight saja.
3. Gunakan poin-poin (•) jika data lebih dari satu.
4. Tebalkan poin penting dengan *teks* (bold WhatsApp).
5. Tambahkan emoticon relevan (🏭, 💰, 📊, ✅, ⚠️, 🔧, 🛢️, 🚨).
6. Maksimal 5 poin per jawaban agar tidak terlalu panjang di layar HP.

Question: {input}"""

# ─── 5. PRISMA KEYWORDS — sama persis dengan web ──────────────────────────────
PRISMA_KEYWORDS = [
    "turnaround", "ta-ex", "taex", "reservasi", "material ta",
    "purchase request", " pr ", "purchase order", " po ",
    "kertas kerja", "kumpulan summary", "work order ta",
    "belum pr", "sudah pr", "delivery material", "stock onhand",
    "sap pr", "sap po", "procurement",
]

# ─── 6. MEMORY PER NOMOR WA ───────────────────────────────────────────────────
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

# ─── 7. CORE FUNCTION — sama persis logika run_with_memory web ────────────────
def run_wa(question: str, sender: str) -> str:
    history    = get_history(sender)
    table_info = db_engine.get_table_info()

    # Build messages dengan history — sama persis dengan web
    messages = [{"role": "system", "content": CUSTOM_PROMPT.replace("{table_info}", table_info).replace("{input}", "")}]
    for msg in history:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})

    # Deteksi PRISMA — sama persis dengan web
    is_prisma = any(kw in question.lower() for kw in PRISMA_KEYWORDS)

    if is_prisma and PRISMA_URL:
        # ── PRISMA PATH — sama persis dengan web ──
        sql_messages = messages + [{"role": "user", "content": (
            f"Berikan HANYA query SQL PostgreSQL yang valid untuk pertanyaan berikut "
            f"menggunakan tabel PRISMA TA-ex. "
            f"Tabel tersedia: taex_reservasi, prisma_reservasi, kumpulan_summary, sap_pr, sap_po, work_order. "
            f"ATURAN WAJIB:\n"
            f"1. Kolom 'order' SELALU ditulis dengan tanda kutip ganda: \"order\"\n"
            f"2. Selalu tambahkan LIMIT 50 di akhir query\n"
            f"3. Untuk hitung yang sudah PR: WHERE pr IS NOT NULL AND pr != ''\n"
            f"4. Untuk hitung yang belum PR: WHERE pr IS NULL OR pr = ''\n"
            f"5. Untuk status PO: JOIN sap_po ON sap_po.purchreq = taex_reservasi.pr\n"
            f"6. Gunakan COUNT(*) atau COUNT(DISTINCT ...) untuk agregasi\n"
            f"7. HANYA output SQL murni, tanpa penjelasan, tanpa markdown, tanpa backtick\n"
            f"\nPertanyaan: {question}"
        )}]
        sql_response = llm.invoke(sql_messages)
        sql_query    = sql_response.content.replace("```sql", "").replace("```", "").strip()
        print(f"[PRISMA SQL] {sql_query}")

        prisma_result = query_prisma(sql_query)
        if prisma_result.get("ok"):
            rows      = prisma_result.get('rows', 0)
            data      = prisma_result.get('data', [])
            db_result = f"Hasil dari PRISMA TA-ex ({rows} baris):\n{data}"
        else:
            err = prisma_result.get('error', 'Unknown error')
            print(f"[PRISMA ERROR] SQL: {sql_query}")
            print(f"[PRISMA ERROR] Error: {err}")
            db_result = (
                f"Query PRISMA gagal. SQL yang dicoba: {sql_query}. "
                f"Error: {err}. "
                f"Coba perbaiki query atau arahkan user ke aplikasi PRISMA langsung."
            )
    else:
        # ── LOCAL PATH — sama persis dengan web ──
        sql_messages = messages + [{"role": "user", "content": (
            f"Berikan HANYA query SQL PostgreSQL yang valid untuk: {question}. "
            f"Tanpa penjelasan, tanpa markdown."
        )}]
        sql_response = llm.invoke(sql_messages)
        sql_query    = sql_response.content.replace("```sql", "").replace("```", "").strip()
        print(f"[LOCAL SQL] {sql_query}")

        try:
            db_result = db_engine.run(sql_query)
        except Exception as e:
            db_result = f"Query error: {str(e)}"

    # Generate jawaban final — format WhatsApp, bukan HTML
    answer_messages = messages + [
        {"role": "user", "content": question},
        {"role": "user", "content": (
            f"Hasil query SQL:\n{db_result}\n\n"
            f"Berikan jawaban final dalam Bahasa Indonesia sesuai aturan format WhatsApp. "
            f"Ingat: narasi saja, tidak ada tabel HTML atau [CHART]."
        )}
    ]
    final_response = llm.invoke(answer_messages)
    answer = final_response.content.replace("```sql", "").replace("```", "").strip()

    # Bersihkan artefak format web yang tidak relevan di WA
    answer = re.sub(r'\[CHART\].*?\[/CHART\]', '', answer, flags=re.DOTALL)
    answer = re.sub(r'<table.*?>.*?</table>', '', answer, flags=re.DOTALL)
    answer = re.sub(r'<[^>]+>', '', answer)
    answer = re.sub(r'\[DOWNLOAD:\w+\]', '', answer).strip()

    # Simpan ke history
    add_history(sender, question, answer)

    return answer

# ─── 8. HELPER ────────────────────────────────────────────────────────────────
def send_wa(target: str, message: str) -> dict:
    response = requests.post(
        "https://api.fonnte.com/send",
        headers={"Authorization": FONNTE_TOKEN},
        data={"target": target, "message": message},
        timeout=30,
    )
    return response.json()

# ─── 9. FLASK APP ─────────────────────────────────────────────────────────────
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

    answer = run_wa(message, sender)
    send_wa(sender, answer)
    return jsonify({"status": "ok"}), 200

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "WA Bot is running 🚀"}), 200

# ─── 10. RUN ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"🚀 WA Bot berjalan di port {port}...")
    app.run(host="0.0.0.0", port=port)