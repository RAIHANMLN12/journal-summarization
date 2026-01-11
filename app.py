import streamlit as st
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# =========================
# Fungsi Load Model
# =========================
@st.cache_resource
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return tokenizer, model

# =========================
# Model Options
# =========================
model_options = {
    "Model Pertama": "/model_jurnal_ilmu_sosial_ketiga",
    "Model Kedua":   "/model_jurnal_gabungan_ketiga",
}

# =========================
# Streamlit Setup
# =========================
st.set_page_config(page_title="Ringkas Jurnal", layout="wide")

st.title("📘 Aplikasi Ringkas Jurnal")

# Session state
if "summaries" not in st.session_state:
    st.session_state.summaries = []


# =========================
# Sidebar untuk pilih model
# =========================
with st.sidebar:
    st.header("⚙️ Pengaturan Model")
    selected_model_name = st.selectbox(
        "Pilih Model:",
        options=list(model_options.keys())
    )
    selected_model_path = model_options[selected_model_name]

    st.success(f"Model aktif: **{selected_model_name}**")

# Load model sesuai pilihan
tokenizer, model = load_model(selected_model_path)


# =========================
# Fungsi Ringkas Teks Panjang
# =========================
def summarize_long_text(text, chunk_size=512):
    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    total_len = len(tokens)
    summaries = []

    for i in range(0, total_len, chunk_size):
        chunk_tokens = tokens[i:i+chunk_size].unsqueeze(0)
        summary_ids = model.generate(
            chunk_tokens,
            max_length=420,
            min_length=100,
            num_beams=6,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            top_k=50,
            length_penalty=1.5,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            early_stopping=True,
        )
        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary_text)

    return " ".join(summaries)


# Format waktu
def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    else:
        return f"{m}m {s}s"


# =========================
# Layout Utama 2 Kolom
# =========================
col1, col2 = st.columns([1, 1])

with col1:
    st.header("✏️ Input Jurnal")
    sub_judul = st.text_input("Sub Judul")
    isi_sub_judul = st.text_area("Isi Sub Judul", height=250)

    if st.button("Ringkas"):
        if isi_sub_judul.strip() != "":
            start_time = time.time()

            final_summary = summarize_long_text(isi_sub_judul)

            elapsed = time.time() - start_time
            formatted_time = format_time(elapsed)

            # Hitung pengurangan
            original_len = len(isi_sub_judul.split())
            summary_len = len(final_summary.split())
            reduction_percent = round((1 - summary_len / original_len) * 100, 2)

            # Simpan hasil
            st.session_state.summaries.append(
                (sub_judul, final_summary, reduction_percent, formatted_time, selected_model_name)
            )

            sub_judul = ""
            isi_sub_judul = ""


with col2:
    st.header("📄 Hasil Ringkasan")

    for i, (judul, ringkas, reduction, waktu, model_used) in enumerate(st.session_state.summaries):
        st.subheader(f"{judul}  —  _(Model: {model_used})_")
        st.write(ringkas)

        st.info(f"**{reduction}% pengurangan** — waktu proses **{waktu}**")

        st.button("Copy", key=f"copy_{i}")
