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

INFORMASI TABEL YANG TERSEDIA:

TABEL UTAMA:
- anggaran_maintenance: rekap anggaran maintenance — kolom ru, tahun, kategori, tipe, nilai_usd.
- pipeline_inspection: inspeksi pipeline — kolom refinery_unit, area, tag_number, last_inspection_date, next_inspection_date, rem_life_years, jumlah_temporary_repair.
- rotor_monitoring: monitoring rotor — kolom refinery_unit, bulan, rotor, program, status_readiness_spare, status_workplan, action_plan_category.
- atg_monitoring: monitoring ATG — kolom refinery_unit, tag_no_tangki, tag_no_atg, status_atg, cert_no_atg, date_expired_atg, status_rtl, month_update.
- bad_actor_monitoring: bad actor equipment — kolom ru, tag_number, status, problem, action_plan, progress, target_date.
- icu_monitoring: Integrity Concern Unit — kolom ru, icu_status, tag_no, issue, mitigation, permanent_solution, progress, target_closed, report_date.
- metering_monitoring: monitoring metering — kolom refinery_unit, tag_number, status_metering, cert_no_metering, date_expired_metering, status_rtl, month_update.
- program_kerja_atg: program kerja ATG — kolom refinery_unit, type, atg_eksisting, program_2024, prokja, action_plan_category, target, month_update.
- paf: Plant Availability Factor — kolom type, ru, target_realisasi, value, plan_unplan, month.
- zero_clamp: zero clamp monitoring — kolom ru, area, unit, tag_no_ln, type_damage, type_perbaikan, status, tanggal_dipasang, tanggal_rencana_perbaikan.
- issue_paf: issue PAF — kolom type (Primary/Secondary Unit), ru, date, issue.
- power_stream: power stream equipment — kolom refinery_unit, type_equipment, equipment, status_operation, desain, kapasitas_max, average_actual.
- jumlah_eqp_utl: jumlah equipment utilitas — kolom refinery_unit, type_equipment, status_equipment, jumlah.
- critical_eqp_utl: critical equipment utilitas — kolom refinery_unit, type_equipment, highlight_issue, corrective_action, mitigasi_action, target_corrective.
- critical_eqp_prim_sec: critical equipment primer/sekunder — kolom refinery_unit, unit_proses, equipment, highlight_issue, corrective_action, mitigasi_action.
- monitoring_operasi: monitoring operasi — kolom refinery_unit, unit_proses, unit, design, minimal_capacity, plant_readiness, actual, target_sts.
- inspection_plan: rencana inspeksi — kolom refinery_unit, area, tag_no_ln, type_equipment, type_inspection, due_date, plan_date, actual_date, result_remaining_life, grand_result.
- tkdn: Tingkat Kandungan Dalam Negeri — kolom refinery_unit, bulan, nominal, kdn, persentase, tahun.
- rcps_rekomendasi: rekomendasi RCPS — kolom kilang, rcps_no, judul_rcps, rekomendasi, traffic, pic, target, remark.
- rcps: daftar RCPS — kolom kilang, traffic, sum_of_progress, disiplin, judul_rcps, rcps_no, criticallity.
- boc: Basis of Comparison equipment — kolom ru, area, unit, equipment, status, frequency, running_hours, mttr, mtbf, hasil.

TABEL BARU (JETTY, TANK, SPM):
- readiness_jetty: kesiapan operasional jetty — kolom refinery_unit, tag_no, status_operation, status_tuks, expired_tuks, status_ijin_ops, status_isps, status_struktur, status_trestle, status_mla, status_fire_protection, month_update.
- workplan_jetty: workplan perbaikan jetty — kolom refinery_unit, tag_no, item, status_item, remark, rtl_action_plan, target, status_rtl, month_update.
- readiness_tank: kesiapan operasional tangki — kolom refinery_unit, tag_number, type_tangki, service_tangki, prioritas, status_operational, atg_certification_validity, status_coi, status_atg, status_grounding, status_shell_course, status_roof, status_cathodic, month_update.
- workplan_tank: workplan perbaikan tangki — kolom unit, tag_no, item, remark, rtl_action_plan, target, status_rtl, month_update.
- readiness_spm: kesiapan operasional SPM — kolom refinery_unit, tag_no, status_operation, status_laik_operasi, expired_laik_operasi, status_ijin_spl, status_mbc, status_lds, status_mooring_hawser, status_floating_hose, status_cathodic_spl, month_update.
- spm_workplan: workplan perbaikan SPM — kolom refinery_unit, tag_no, item, remark, rtl_action_plan, target, status_rtl, month_update.

Table structure: {table_info}
Question: {input}"""

db = SQLDatabase.from_uri(
    DATABASE_URL,
    include_tables=[
        # Tabel lama
        "anggaran_maintenance",
        "pipeline_inspection",
        "rotor_monitoring",
        "atg_monitoring",
        "bad_actor_monitoring",
        "icu_monitoring",
        "metering_monitoring",
        "program_kerja_atg",
        "paf",
        "zero_clamp",
        "issue_paf",
        "power_stream",
        "jumlah_eqp_utl",
        "critical_eqp_utl",
        "critical_eqp_prim_sec",
        "monitoring_operasi",
        "inspection_plan",
        "tkdn",
        "rcps_rekomendasi",
        "rcps",
        "boc",
        # Tabel baru
        "readiness_jetty",
        "workplan_jetty",
        "readiness_tank",
        "workplan_tank",
        "readiness_spm",
        "spm_workplan",
    ]
)
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