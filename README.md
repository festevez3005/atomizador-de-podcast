# 🎯 Content Atomizer & Topicizer (Podcast → Multi‑channel)

Una app en **Streamlit Cloud** para creadores: pega un **tema** o la **transcripción** de tu podcast y recibe intención de búsqueda, temas relacionados, listas de keywords y recomendaciones de contenidos (SEO, redes, LinkedIn), más un **plan de atomización** listo para ejecutar.

## 🚀 Demo rápida (local)
```bash
python -m venv .venv && source .venv/bin/activate  # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-xxxxx  # opcional; sin clave usa modo heurístico
streamlit run app.py
```

## ☁️ Deploy en Streamlit Cloud
1. Sube este repo a GitHub.
2. En Streamlit Cloud, crea una nueva app apuntando a `app.py`.
3. En **Secrets**, agrega:
   ```toml
   OPENAI_API_KEY = "sk-xxxxx"   # opcional pero recomendado
   ```
4. (Opcional) Cambia el modelo en la barra lateral (`gpt-4o-mini` por defecto).

## 💡 ¿Cómo funciona?
- **Con LLM (recomendado):** llama a OpenAI (chat) con un prompt estructurado y exige **JSON** con:
  - intención de búsqueda
  - temas relacionados
  - keywords (short/mid/long tail)
  - ideas para artículos SEO, sociales, LinkedIn y newsletter
  - plan de **atomización** (≥10 piezas)
- **Sin LLM (fallback):** aplica heurísticas locales para extraer n‑gramas y frecuencias; útil para bosquejos rápidos.

## 📦 Archivos
- `app.py` → la app Streamlit
- `requirements.txt` → dependencias mínimas
- `README.md` → este archivo

## 🔐 Variables de entorno
- `OPENAI_API_KEY` (o en Secrets).

## 🧰 Stack
- Python 3.10+
- Streamlit
- OpenAI SDK v1
- Pandas

## 📝 Notas
- Este proyecto no hace scraping ni SEO a SERPs: se enfoca en **topicalización** del contenido fuente.
- Si el JSON del modelo falla, hay reintentos y reparación mínima. Si aún así falla, se usa modo heurístico.

## 🗺️ Roadmap sugerido
- Embeddings + clustering de segmentos para subtemas finos.
- Plantillas exportables (Notion/Markdown) por canal.
- Integración con programadores (Buffer, Hootsuite) mediante export.
