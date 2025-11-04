# app_simple.py
import re
from collections import Counter
import pandas as pd
import streamlit as st

st.set_page_config(page_title="🎯 Simple Topic Analyzer", page_icon="🎯", layout="wide")

st.title("🎯 Analizador de temas y palabras clave")
st.caption("Versión ligera sin dependencias pesadas ni API externa.")

def tokenize(text):
    toks = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", text.lower())
    return [t for t in toks if len(t) > 2]

def analyze_text(text):
    words = tokenize(text)
    common = [w for w, _ in Counter(words).most_common(50)]
    mid_tail = list(dict.fromkeys([" ".join(words[i:i+2]) for i in range(len(words)-1)]))[:20]
    long_tail = list(dict.fromkeys([" ".join(words[i:i+3]) for i in range(len(words)-2)]))[:20]
    related = [w for w in common if w not in ("que","para","con","los","las","por","del","una","como","pero")][:10]
    plan = [
        "Post con cita destacada",
        "Carrusel con los 3 puntos clave",
        "Short de 30s con momento WOW",
        "Mini artículo tipo resumen",
        "Encuesta sobre el tema principal"
    ]
    return {
        "temas": related,
        "short_tail": common[:10],
        "mid_tail": mid_tail[:10],
        "long_tail": long_tail[:10],
        "plan": plan
    }

with st.expander("ℹ️ Instrucciones"):
    st.markdown("Pega una transcripción o texto largo para obtener temas y sugerencias básicas.")

text = st.text_area("Pega tu contenido:", height=200)
run = st.button("Analizar")

if run:
    if not text.strip():
        st.warning("Por favor ingresa un texto.")
        st.stop()

    result = analyze_text(text)
    st.subheader("Temas principales")
    st.write(", ".join(result["temas"]) or "—")

    st.subheader("Palabras clave")
    st.write("**Short tail:**", ", ".join(result["short_tail"]))
    st.write("**Mid tail:**", ", ".join(result["mid_tail"]))
    st.write("**Long tail:**", ", ".join(result["long_tail"]))

    st.subheader("Ideas de contenido")
    for i, idea in enumerate(result["plan"], 1):
        st.markdown(f"{i}. {idea}")

    kw_df = pd.DataFrame({
        "short_tail": pd.Series(result["short_tail"]),
        "mid_tail": pd.Series(result["mid_tail"]),
        "long_tail": pd.Series(result["long_tail"]),
    })
    st.download_button("⬇️ Descargar CSV", data=kw_df.to_csv(index=False), file_name="keywords_simple.csv", mime="text/csv")
