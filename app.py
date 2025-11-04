# app.py
import os
import json
import time
import pandas as pd
import streamlit as st
from typing import Dict, Any, List, Optional

# Try OpenAI SDK v1
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

APP_TITLE = "🎯 Content Atomizer & Topicizer (Podcast → Multi‑channel)"
APP_DESC = """
Paste a transcript or a general topic and get:
- Search intent & related subtopics
- Keyword lists (short, mid, long-tail)
- Content recommendations (SEO articles, social, LinkedIn)
- An atomization plan to repurpose across formats

Works best with an OpenAI API key. If not provided, a lightweight local fallback will generate heuristic outputs.
"""

def get_openai_client() -> Optional["OpenAI"]:
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if _HAS_OPENAI and api_key:
        return OpenAI(api_key=api_key)
    return None

LLM_MODEL_DEFAULT = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBED_MODEL_DEFAULT = os.getenv("EMBED_MODEL", "text-embedding-3-small")

SYSTEM_PROMPT = """You are an expert content strategist for Spanish-speaking creators.
Given a transcript or general topic, produce concise, actionable outputs in Spanish.
Be practical. Avoid fluff. Return valid JSON following the provided schema exactly.
"""

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "search_intent": {"type": "string", "description": "Primary search intent behind the topic (informational, navigational, transactional, commercial investigation)."},
        "related_topics": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Topically related ideas or subtopics."
        },
        "keywords": {
            "type": "object",
            "properties": {
                "short_tail": {"type": "array", "items": {"type": "string"}},
                "mid_tail": {"type": "array", "items": {"type": "string"}},
                "long_tail": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["short_tail", "mid_tail", "long_tail"]
        },
        "content_recommendations": {
            "type": "object",
            "properties": {
                "seo_articles": {"type": "array", "items": {"type": "string"}},
                "social_posts": {"type": "array", "items": {"type": "string"}},
                "linkedin_posts": {"type": "array", "items": {"type": "string"}},
                "email_newsletter": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["seo_articles", "social_posts", "linkedin_posts", "email_newsletter"]
        },
        "atomization_plan": {
            "type": "array",
            "description": "List of smaller content pieces derived from the main asset; include format and angle.",
            "items": {"type": "string"}
        },
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
    "required": ["search_intent", "related_topics", "keywords", "content_recommendations", "atomization_plan", "metadata"]
}

def llm_topicize(client, text: str, language: str = "es") -> Dict[str, Any]:
    """Call OpenAI to get structured topicalization JSON, with retry/repair if needed."""
    user_prompt = f"""Contenido fuente (idioma {language}):
---
{text.strip()[:12000]}
---

Instrucciones:
1) Analiza el contenido y determina intención de búsqueda y temas relacionados.
2) Genera listas de palabras clave (short/mid/long tail).
3) Recomienda formatos y titulares concretos para: artículos SEO, publicaciones sociales, LinkedIn y newsletter.
4) Propón una atomización detallada (≥10 piezas) que cubra carruseles, hilos, shorts, clips, infografías, etc.
5) Devuelve SOLO JSON que cumpla EXACTAMENTE este esquema:
{json.dumps(JSON_SCHEMA, ensure_ascii=False)}
"""
    # Try 2 attempts
    for _ in range(2):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL_DEFAULT,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            js = resp.choices[0].message.content
            return json.loads(js)
        except Exception as e:
            time.sleep(0.8)
            continue
    # Final fallback raises
    raise RuntimeError("No se pudo obtener una respuesta JSON válida del LLM.")

# --- Lightweight local fallbacks (heuristic) ---
import re
from collections import Counter
def _tokenize(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", text.lower())
    return [t for t in toks if len(t) > 2]

def local_topicize(text: str) -> Dict[str, Any]:
    words = _tokenize(text)
    common = [w for w, _ in Counter(words).most_common(60)]
    # naive n-grams
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
            "notes": "Resultados generados sin LLM (modo fallback)."
        }
    }
    return out

def json_to_frames(js: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    kw = js.get("keywords", {})
    frames = {}
    frames["Palabras clave"] = pd.DataFrame({
        "short_tail": pd.Series(kw.get("short_tail", [])),
        "mid_tail": pd.Series(kw.get("mid_tail", [])),
        "long_tail": pd.Series(kw.get("long_tail", [])),
    })
    frames["Temas relacionados"] = pd.DataFrame({"related_topics": js.get("related_topics", [])})
    for k, v in js.get("content_recommendations", {}).items():
        frames[f"Recomendaciones · {k}"] = pd.DataFrame({k: v})
    frames["Atomización"] = pd.DataFrame({"pieza": js.get("atomization_plan", [])})
    return frames

def export_markdown(topic_input: str, js: Dict[str, Any]) -> str:
    lines = [f"# Plan de contenidos (generado)",
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
    st.set_page_config(page_title="Content Atomizer", page_icon="🎯", layout="wide")
    st.title(APP_TITLE)
    st.caption(APP_DESC)

    with st.sidebar:
        st.subheader("LLM")
        st.write("Provee tu **OPENAI_API_KEY** en *Secrets* o como variable de entorno.")
        model = st.text_input("Modelo (chat)", value=LLM_MODEL_DEFAULT)
        embed_model = st.text_input("Modelo (embeddings)", value=EMBED_MODEL_DEFAULT)
        st.session_state["model"] = model
        st.session_state["embed_model"] = embed_model
        st.divider()
        export_btn = st.button("📝 Exportar a Markdown", key="export_md", disabled=True)

    tab1, tab2 = st.tabs(["➕ Entrada", "📊 Resultados"])

    with tab1:
        mode = st.radio("Modo de entrada", ["Tema general", "Transcripción de episodio"], horizontal=True)
        text = st.text_area(
            "Pega aquí tu texto",
            height=240,
            placeholder="Ej.: Inteligencia artificial aplicada al marketing de contenidos... o pega la transcripción de tu episodio."
        )
        uploaded = st.file_uploader("…o sube un archivo .txt / .md", type=["txt", "md"])

        if uploaded and not text.strip():
            text = uploaded.read().decode("utf-8", errors="ignore")

        colA, colB = st.columns([1,1])
        with colA:
            min_items = st.slider("Mínimo de ideas por sección", 5, 20, 10)
        with colB:
            temperature = st.slider("Creatividad (temperature)", 0.0, 1.0, 0.3, 0.1)

        run_btn = st.button("🚀 Generar plan", type="primary", use_container_width=True)

    if run_btn:
        if not (text and text.strip()):
            st.error("Por favor pega o sube contenido.")
            st.stop()

        client = get_openai_client()
        if client:
            # adjust global model based on sidebar
            global LLM_MODEL_DEFAULT
            LLM_MODEL_DEFAULT = st.session_state.get("model", LLM_MODEL_DEFAULT)
            try:
                with st.spinner("Consultando LLM…"):
                    js = llm_topicize(client, text)
            except Exception as e:
                st.warning("Fallo del LLM. Usando heurística local…")
                js = local_topicize(text)
        else:
            st.info("No se detectó OPENAI_API_KEY. Usando heurística local.")
            js = local_topicize(text)

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

            # Exports
            md = export_markdown(st.session_state.get("last_input",""), js)
            md_path = "plan_contenidos.md"
            st.download_button("⬇️ Descargar Markdown", data=md, file_name=md_path, mime="text/markdown")

            # Keyword CSV
            kw_df = pd.DataFrame({
                "short_tail": pd.Series(kw.get("short_tail", [])),
                "mid_tail": pd.Series(kw.get("mid_tail", [])),
                "long_tail": pd.Series(kw.get("long_tail", [])),
            })
            st.download_button("⬇️ Descargar palabras clave (CSV)", data=kw_df.to_csv(index=False), file_name="keywords.csv", mime="text/csv")

if __name__ == "__main__":
    run()
