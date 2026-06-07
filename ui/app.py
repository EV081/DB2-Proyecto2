import streamlit as st

st.set_page_config(page_title="Multimodal Search — Demo", layout="wide")

st.title("Búsqueda Multimodal (Demo Mock)")
st.markdown("Hito 1 — Esqueleto de interfaz. Los resultados son simulados.")

tab_audio, tab_image, tab_text = st.tabs(["🎵 Audio", "🖼️ Imagen", "📄 Texto"])

MOCK_RESULTS = [
    {"rank": i + 1, "id": f"item-{i:04d}", "score": round(0.95 - i * 0.08, 3)}
    for i in range(5)
]

def show_tab(label: str, accept_types: list[str]):
    st.header(f"Búsqueda por {label}")
    uploaded = st.file_uploader(f"Sube un archivo de {label.lower()}", type=accept_types)

    if uploaded is not None:
        st.success(f"Archivo recibido: {uploaded.name}")
    else:
        st.info("Carga un archivo para ver resultados simulados.")
        return

    st.subheader("Top-K Resultados (Mock)")
    for r in MOCK_RESULTS:
        st.write(f"  **#{r['rank']}** — `{r['id']}`  (similitud: {r['score']})")

with tab_audio:
    show_tab("Audio", ["wav", "mp3", "flac"])

with tab_image:
    show_tab("Imagen", ["jpg", "jpeg", "png"])

with tab_text:
    show_tab("Texto", ["txt", "json", "csv"])

st.divider()
st.caption("Modo mock — Hito 1. Conecta con la API real en hitos posteriores.")
