# app_oss.py
import os
import json
import time
import torch
import pandas as pd
import streamlit as st
from typing import Dict, Any, List, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

APP_TITLE = "🎯 Content Atomizer (OSS)"
APP_DESC = """
 Carga un modelo pequeño de Hugging Face(local o al vuelo) y produce salidas estructuradas en **JSON**.
"""

# ---- Config ----
DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
GEN_KWARGS = {
    "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", "800")),
    "temperature": float(os.getenv("TEMPERATURE", "0.2")),
    "top_p": float(os.getenv("TOP_P", "0.95")),
    "do_sample": True
}

SYSTEM_PROMPT = """You are an expert content strategist for Spanish-speaking creators.
Given a transcript or general topic, return concise, actionable outputs in Spanish.
Return **ONLY** strictly valid JSON. No prose outside JSON.
"""

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "search_intent": {"type": "string"},
        "related_topics": {"type": "array", "items": {"type": "string"}},
        "keywords": {
            "type": "object",
            "properties": {
                "short_tail": {"type": "array", "items": {"type": "string"}},
                "mid_tail": {"type": "array", "items": {"type": "string"}},
                "long_tail": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["short_tail","mid_tail","long_tail"]
        },
        "content_recommendations": {
            "type": "object",
            "properties": {
                "seo_articles": {"type": "array", "items": {"type": "string"}},
                "social_posts": {"type": "array", "items": {"type": "string"}},
                "linkedin_posts": {"type": "array", "items": {"type": "string"}},
                "email_newsletter": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["seo_articles","social_posts","linkedin_posts","email_newsletter"]
        },
        "atomization_plan": {"type": "array", "items": {"type": "string"}},
        "metadata": {
            "type": "object",
            "properties": {
                "reading_levels": {"type": "array", "items": {"type": "string"}},
                "posting_cadence_recommendation": {"type": "string"},
                "notes": {"type": "string"}
            },
            "required": ["posting_cadence_recommendation"]
        }
    },
    "required": ["search_intent","related_topics","keywords","content_recommendations","atomization_plan","metadata"]
}

# --- Heurístico local (fallback) ---
import re
from collections import Counter
def _tokenize(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", text.lower())
    return [t for t in toks if len(t) > 2]

def local_topicize(text: str) -> Dict[str, Any]:
    words = _tokenize(text)
    common = [w for w, _ in Counter(words).most_common(60)]
    mid_tail = list(dict.fromkeys([" ".join(words[i:i+2]) for i in range(len(words)-1)]))[:30]
    long_tail = list(dict.fromkeys([" ".join(words[i:i+3]) for i in range(len(words)-2)]))[:30]
    related = [w for w in common if w not in set(("que","para","con","los","las","por","del","una","como","pero"))][:12]
    out = {
        "search_intent": "informacional (heurístico)",
        "related_topics": related,
        "keywords": {
            "short_tail": common[:12],
            "mid_tail": mid_tail[:12],
            "long_tail": long_tail[:12],
        },
        "content_recommendations": {
            "seo_articles": [f"Guía esencial sobre {related[0]}" if related else "Guía esencial del tema"],
            "social_posts": ["Idea de post: 3 aprendizajes clave + CTA a episodio"],
            "linkedin_posts": ["Post tipo caso: problema → hipótesis → resultado → CTA"],
            "email_newsletter": ["Resumen del episodio + 2 recursos útiles + pregunta para responder"]
        },
        "atomization_plan": [
            "Hilo (7 tweets) con insights",
            "Carrusel (6 slides) con marco mental",
            "2 shorts (30-45s) con momentos WOW del podcast",
            "Artículo breve (600-800 palabras) con los puntos clave",
            "Infografía comparativa (X vs Y)",
            "Checklist descargable",
            "Post de comunidad con pregunta abierta",
            "Guión para video de 3 minutos con CTA",
            "Snippet con cita destacada",
            "Idea para encuesta de redes"
        ],
        "metadata": {
            "reading_levels": ["divulgativo", "intermedio"],
            "posting_cadence_recommendation": "3-4 piezas por semana derivadas del episodio.",
            "notes": "Resultados generados en modo heurístico."
        }
    }
    return out

# --- Carga de modelo ---
@st.cache_resource(show_spinner=False)
def load_pipe(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        # Important to prevent very long ram use
        max_new_tokens=GEN_KWARGS["max_new_tokens"],
        do_sample=GEN_KWARGS["do_sample"],
        temperature=GEN_KWARGS["temperature"],
        top_p=GEN_KWARGS["top_p"]
    )

def llm_topicize(pipe, text: str, min_items: int = 10, language: str = "es") -> Dict[str, Any]:
    # System + user in a simple instruct style (works with many small instruct models)
    schema_str = json.dumps(JSON_SCHEMA, ensure_ascii=False)
    prompt = f"""[SYSTEM]
{SYSTEM_PROMPT}

[USER]
Contenido fuente (idioma {language}):
---
{text.strip()[:12000]}
---

Instrucciones:
1) Analiza el contenido y determina intención de búsqueda y temas relacionados.
2) Genera listas de palabras clave (short/mid/long tail).
3) Recomienda formatos y titulares concretos para: artículos SEO, publicaciones sociales, LinkedIn y newsletter.
4) Propón una atomización detallada (≥{min_items} piezas) que cubra carruseles, hilos, shorts, clips, infografías, etc.
5) Devuelve SOLO JSON que cumpla EXACTAMENTE este esquema:
{schema_str}
"""
    raw = pipe(prompt)[0]["generated_text"]
    # Intenta extraer el primer bloque JSON
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        js_str = raw[start:end+1]
        try:
            return json.loads(js_str)
        except Exception:
            pass
    # Fallback
    return local_topicize(text)

def export_markdown(topic_input: str, js: Dict[str, Any]) -> str:
    lines = [f"# Plan de contenidos (OSS)",
             f"_Fecha: {time.strftime('%Y-%m-%d')}_",
             "",
             "## Input",
             "```",
             topic_input.strip()[:2000],
             "```",
             "",
             "## Intención de búsqueda",
             f"- {js.get('search_intent', '')}",
             "",
             "## Temas relacionados"]
    for t in js.get("related_topics", []):
        lines.append(f"- {t}")
    lines += ["", "## Palabras clave", "### Short tail"]
    for t in js.get("keywords", {}).get("short_tail", []):
        lines.append(f"- {t}")
    lines += ["", "### Mid tail"]
    for t in js.get("keywords", {}).get("mid_tail", []):
        lines.append(f"- {t}")
    lines += ["", "### Long tail"]
    for t in js.get("keywords", {}).get("long_tail", []):
        lines.append(f"- {t}")
    lines += ["", "## Recomendaciones de contenido"]
    cr = js.get("content_recommendations", {})
    for bucket in ["seo_articles", "social_posts", "linkedin_posts", "email_newsletter"]:
        if bucket in cr:
            lines.append(f"### {bucket}")
            for item in cr[bucket]:
                lines.append(f"- {item}")
    lines += ["", "## Plan de atomización"]
    for a in js.get("atomization_plan", []):
        lines.append(f"- {a}")
    md = "\n".join(lines)
    return md

def run():
    st.set_page_config(page_title="Content Atomizer (OSS)", page_icon="🎯", layout="wide")
    st.title(APP_TITLE)
    st.caption(APP_DESC)

    with st.sidebar:
        st.subheader("Modelo OSS")
        model_id = st.text_input("Hugging Face model id", value=DEFAULT_MODEL_ID, help="Ej.: Qwen/Qwen2.5-0.5B-Instruct o TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        tokens = st.number_input("max_new_tokens", min_value=128, max_value=2048, value=GEN_KWARGS["max_new_tokens"], step=32)
        temperature = st.slider("Creatividad", 0.0, 1.0, float(GEN_KWARGS["temperature"]), 0.05)
        st.session_state["gen_cfg"] = {"max_new_tokens": int(tokens), "temperature": float(temperature)}
        st.write("Si falla o la salida no es JSON válido, usaremos un **heurístico local**.")

    tab1, tab2 = st.tabs(["➕ Entrada", "📊 Resultados"])
    with tab1:
        mode = st.radio("Modo de entrada", ["Tema general", "Transcripción de episodio"], horizontal=True)
        text = st.text_area("Pega el contenido", height=240)
        uploaded = st.file_uploader("…o sube .txt / .md", type=["txt","md"])
        if uploaded and not text.strip():
            text = uploaded.read().decode("utf-8", errors="ignore")
        min_items = st.slider("Mínimo de piezas de atomización", 5, 30, 12)
        run_btn = st.button("🚀 Generar plan (OSS)", type="primary", use_container_width=True)

    if run_btn:
        if not text.strip():
            st.error("Por favor pega o sube contenido.")
            st.stop()
        with st.spinner(f"Cargando/ejecutando {model_id}… (puede tardar la primera vez)"):
            try:
                pipe = load_pipe(model_id)
            except Exception as e:
                st.error(f"No se pudo cargar el modelo {model_id}: {e}")
                st.stop()
        with st.spinner("Generando…"):
            js = llm_topicize(pipe, text, min_items=min_items)
        st.session_state["last_input"] = text
        st.session_state["result_json"] = js
        st.success("¡Listo! Revisa la pestaña Resultados.")

    with tab2:
        js = st.session_state.get("result_json")
        if not js:
            st.info("Genera un plan primero en la pestaña Entrada.")
        else:
            a,b = st.columns([2,1])
            with a:
                st.subheader("Intención de búsqueda")
                st.write(js.get("search_intent",""))
                st.subheader("Temas relacionados")
                st.write(", ".join(js.get("related_topics", [])) or "—")
                st.subheader("Palabras clave")
                kw = js.get("keywords", {})
                st.write("**Short tail**:", ", ".join(kw.get("short_tail", [])) or "—")
                st.write("**Mid tail**:", ", ".join(kw.get("mid_tail", [])) or "—")
                st.write("**Long tail**:", ", ".join(kw.get("long_tail", [])) or "—")
            with b:
                st.subheader("Cadencia sugerida")
                st.write(js.get("metadata",{}).get("posting_cadence_recommendation","—"))
                st.subheader("Notas")
                st.write(js.get("metadata",{}).get("notes","—"))

            st.divider()
            st.subheader("Recomendaciones de generación")
            for bucket in ["seo_articles", "social_posts", "linkedin_posts", "email_newsletter"]:
                items = js.get("content_recommendations", {}).get(bucket, [])
                with st.expander(bucket.replace("_"," ").title()):
                    for i, idea in enumerate(items, 1):
                        st.markdown(f"{i}. {idea}")

            st.subheader("Plan de atomización")
            for i, idea in enumerate(js.get("atomization_plan", []), 1):
                st.markdown(f"{i}. {idea}")

            # Export
            md = export_markdown(st.session_state.get("last_input",""), js)
            st.download_button("⬇️ Descargar Markdown", data=md, file_name="plan_contenidos_oss.md", mime="text/markdown")

            kw = js.get("keywords", {})
            kw_df = pd.DataFrame({
                "short_tail": pd.Series(kw.get("short_tail", [])),
                "mid_tail": pd.Series(kw.get("mid_tail", [])),
                "long_tail": pd.Series(kw.get("long_tail", [])),
            })
            st.download_button("⬇️ Descargar palabras clave (CSV)", data=kw_df.to_csv(index=False), file_name="keywords_oss.csv", mime="text/csv")

if __name__ == "__main__":
    run()
