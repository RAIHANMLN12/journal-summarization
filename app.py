import streamlit as st
import time
import fitz
import easyocr
import torch
from pdf2image import convert_from_bytes
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO


# =========================
# Load Model
# =========================
@st.cache_resource
def load_model(model_path):

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path
    )

    return tokenizer, model


# =========================
# Load OCR
# =========================
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['id','en'])


reader = load_ocr()


# =========================
# Smart PDF Extraction
# =========================
def extract_text_smart(uploaded_file):

    pdf_bytes = uploaded_file.read()

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    text = ""
    pages_with_text = 0

    for page in doc:
        page_text = page.get_text().strip()
        text += page_text + "\n"

        if len(page_text) > 50:
            pages_with_text += 1

    # ---------- OCR fallback ----------
    if pages_with_text < len(doc) * 0.3:

        images = convert_from_bytes(pdf_bytes)

        text = ""

        for img in images:

            results = reader.readtext(img)

            for detection in results:
                text += detection[1] + " "

            text += "\n"

    return text


# =========================
# Chunking Summarization
# =========================
def summarize_chunking(text, chunk_size=512):

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

    # Gabungkan semua ringkasan chunk
    final_summary = " ".join(summaries)

    return final_summary


# =========================
# Model Options
# =========================
model_options = {
    "Model Ilmu Sosial": "./model_jurnal_sosial_pertama",
    "Model Gabungan": "./model_jurnal_gabungan_ketiga",
}


# =========================
# Streamlit Setup
# =========================
st.set_page_config(page_title="Ringkas Jurnal", layout="wide")


# =========================
# Header
# =========================
col_title, col_model = st.columns([3,1])

with col_title:
    st.title("📘 Aplikasi Ringkas Jurnal")

with col_model:
    selected_model_name = st.selectbox(
        "Model",
        options=list(model_options.keys())
    )

selected_model_path = model_options[selected_model_name]


# =========================
# Load Model
# =========================
tokenizer, model = load_model(selected_model_path)


# =========================
# Session State
# =========================
if "summaries" not in st.session_state:
    st.session_state.summaries = []

if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""


# =========================
# Time Formatter
# =========================
def format_time(seconds):

    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)

    if h > 0:
        return f"{h}h {m}m {s}s"
    else:
        return f"{m}m {s}s"


# =========================
# Export Summary to PDF
# =========================
def create_summary_pdf(summary_text, model_name, reduction, time_used):

    buffer = BytesIO()

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Ringkasan Jurnal", styles['Title']))
    story.append(Spacer(1,20))

    story.append(Paragraph(f"<b>Pengurangan:</b> {reduction}%", styles['Normal']))
    story.append(Paragraph(f"<b>Waktu Proses:</b> {time_used}", styles['Normal']))

    story.append(Spacer(1,20))

    paragraphs = summary_text.split("\n")

    for p in paragraphs:
        story.append(Paragraph(p, styles['BodyText']))
        story.append(Spacer(1,10))

    doc = SimpleDocTemplate(buffer)
    doc.build(story)

    buffer.seek(0)

    return buffer


status_placeholder = st.empty()
progress_bar = st.empty()


# =========================
# Layout
# =========================
col1, col2 = st.columns([1,1])


# =========================
# INPUT
# =========================
with col1:

    st.header("✏️ Input")

    input_method = st.radio(
        "Pilih Metode Input",
        ["Copy Paste Teks", "Upload PDF"]
    )


    # =========================
    # Copy Paste
    # =========================
    if input_method == "Copy Paste Teks":

        text_input = st.text_area(
            "Masukkan teks jurnal",
            height=300
        )


    # =========================
    # Upload PDF
    # =========================
    else:

        uploaded_file = st.file_uploader(
            "Upload file PDF",
            type=["pdf"]
        )

        if uploaded_file is not None:

            if st.button("Ekstrak Teks dari PDF"):

                with st.spinner("Ekstraksi teks..."):

                    extracted = extract_text_smart(uploaded_file)

                    st.session_state.extracted_text = extracted

                st.success("Ekstraksi selesai!")

        text_input = st.text_area(
            "Teks hasil ekstraksi (bisa diedit)",
            value=st.session_state.extracted_text,
            height=300
        )


    # =========================
    # Button Ringkas
    # =========================
    if st.button("Ringkas"):

        if text_input.strip() != "":

            status_placeholder.info("⏳ Sedang meringkas dokumen...")

            start_time = time.time()

            final_summary = summarize_chunking(text_input)

            status_placeholder.success("✅ Ringkasan selesai dibuat!")

            elapsed = time.time() - start_time

            formatted_time = format_time(elapsed)

            original_len = len(text_input.split())
            summary_len = len(final_summary.split())

            reduction_percent = round(
                (1 - summary_len / original_len) * 100,
                2
            )

            st.session_state.summaries.append(
                (
                    final_summary,
                    reduction_percent,
                    formatted_time,
                    selected_model_name
                )
            )


# =========================
# OUTPUT
# =========================
with col2:

    st.header("📄 Hasil Ringkasan")

    status_placeholder = st.empty()

    for i, (ringkas, reduction, waktu, model_used) in enumerate(st.session_state.summaries):

        st.subheader(f"Model: {model_used}")

        st.write(ringkas)

        st.info(
            f"**{reduction}% pengurangan** — waktu proses **{waktu}**"
        )

        col_copy, col_export = st.columns(2)

        with col_copy:
            st.button("Copy", key=f"copy_{i}")

        with col_export:

            pdf_file = create_summary_pdf(
                ringkas,
                model_used,
                reduction,
                waktu
            )

            st.download_button(
                label="Export PDF",
                data=pdf_file,
                file_name=f"ringkasan_{i}.pdf",
                mime="application/pdf",
                key=f"export_{i}"
            )