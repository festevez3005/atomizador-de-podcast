# app.py
"""
Streamlit app para ayudar a creadores a convertir un tema o transcript de podcast
en un plan de contenidos completo (SEO + redes + LinkedIn) con enfoque de
atomización. Usa la API de OpenAI vía el SDK oficial.

Seguridad de claves:
- Lee la clave desde st.secrets["OPENAI_API_KEY"] o un campo seguro del usuario.
- Si defines APP_PASSWORD en el entorno, activa un gate de acceso simple.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, Any
import streamlit as st
from openai import OpenAI

# ------------------ Configuración de página ------------------
st.set_page_config(
    page_title="Atomizador de Contenidos (Podcast → Contenido)",
    page_icon="🎙️",
    layout="wide",
)

# --------- Gate opcional con contraseña (simple) ---------
APP_PASSWORD = os.getenv("APP_PASSWORD")
if APP_PASSWORD:
    st.sidebar.markdown("### 🔒 Acceso")
    pw = st.sidebar.text_input("Contraseña", type="password")
    if pw != APP_PASSWORD:
        st.warning("App protegida. Ingresa la contraseña correcta para continuar.")
        st.stop()

# ------------------ Configuración ------------------
DEFAULT_MODEL = "gpt-4.1-mini"

SYSTEM_INSTRUCTIONS = """
Eres un estratega de contenidos y SEO senior. Recibirás un transcript o tema.
Devuelve EXCLUSIVAMENTE un JSON con:
- Intención de búsqueda (TOFU/MOFU/BOFU)
- Temas relacionados
- Clusters de keywords
- Recomendaciones de artículos SEO
- Ideas para redes/LinkedIn
- Mapa de atomización
Idioma del input = idioma de salida.
""".strip()

PROMPT_TEMPLATE = """
INPUT:
---
{user_text}
---
Genera un JSON estructurado con los datos solicitados.
""".strip()

@dataclass
class LLMConfig:
    api_key: str
    model: str = DEFAULT_MODEL
    temperature: float = 0.2
    max_output_tokens: int = 2000

def get_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)

def call_llm(cfg: LLMConfig, user_text: str) -> Dict[str, Any]:
    client = get_openai_client(cfg.api_key)
    prompt = PROMPT_TEMPLATE.replace("{user_text}", user_text)

    resp = client.responses.create(
        model=cfg.model,
        input=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ],
        temperature=cfg.temperature,
        max_output_tokens=cfg.max_output_tokens,
    )
    text = resp.output_text.strip()
    if text.startswith("```"):
        text = text.strip("`\n ").removeprefix("json").strip()
    return json.loads(text)

# ------------------ Sidebar ------------------
st.sidebar.header("⚙️ Configuración")
secret_key = st.secrets.get("OPENAI_API_KEY", None)
use_custom_key = st.sidebar.checkbox("Usar mi propia OpenAI API Key", value=(secret_key is None))
api_key = st.sidebar.text_input("Tu OpenAI API Key", type="password") if use_custom_key else secret_key
model = st.sidebar.selectbox("Modelo", ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o"])
temp = st.sidebar.slider("Creatividad (temperature)", 0.0, 1.0, 0.2, 0.05)
max_toks = st.sidebar.slider("Límite de tokens de salida", 500, 8000, 2000, 100)

# ------------------ Entrada ------------------
st.title("🎙️ Atomizador de Contenidos desde Transcript")
st.write("Convierte un tema o transcript de podcast en ideas SEO y redes sociales.")

input_mode = st.radio("¿Qué vas a ingresar?", ["Tema general", "Transcript (pegar)", "Transcript (archivo .txt/.md)"])
user_text = ""
if input_mode == "Tema general":
    user_text = st.text_area("Tema o idea central", height=140)
elif input_mode == "Transcript (pegar)":
    user_text = st.text_area("Pega aquí el transcript completo", height=260)
else:
    up = st.file_uploader("Subí un .txt o .md con el transcript", type=["txt", "md"])
    if up is not None:
        user_text = up.read().decode("utf-8", errors="ignore")
        st.success(f"Archivo cargado: {up.name} — {len(user_text)} caracteres")

run = st.button("🚀 Generar plan y contenidos", type="primary", disabled=not bool(user_text.strip()))
results = None

if run:
    if not api_key:
        st.error("Falta la OpenAI API Key. Configúrala en el sidebar o en secrets.")
    else:
        with st.spinner("Analizando transcript..."):
            try:
                cfg = LLMConfig(api_key=api_key, model=model, temperature=temp, max_output_tokens=max_toks)
                results = call_llm(cfg, user_text)
                st.session_state["results"] = results
            except Exception as e:
                st.exception(e)

if results is None and "results" in st.session_state:
    results = st.session_state["results"]

# ------------------ Resultados ------------------
if results:
    st.success("¡Listo! Aquí están tus entregables.")
    st.json(results)
    st.download_button(
        "⬇️ Descargar JSON",
        data=json.dumps(results, ensure_ascii=False, indent=2),
        file_name="atomizacion_contenidos.json",
        mime="application/json",
    )

st.caption("💡 Configura tu clave en 'Secrets' o pégala temporalmente para usar la app.")
