# Hybrid Chatbot System

A Python-based hybrid chatbot prototype combining deterministic responses, semantic retrieval and an LLM-backed response path. Includes a Streamlit interface, example test queries and sample evaluation output.

## Features
- Hybrid chatbot response logic and retrieval-oriented flow
- Streamlit interface
- Sample test queries and evaluation output
- Environment-variable based model credentials

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run src/streamlit_app.py
```

Review `src/chatbot.py` for model configuration and dependencies before running. Set `OPENAI_API_KEY` in your environment if your selected execution path requires it. Never commit API keys or `.env` files.

## Repository layout

```text
src/       chatbot and Streamlit UI
data/     sample test queries
outputs/  sample evaluation output
scripts/  query runner
```

## Evaluation note
The included results are sample artifacts. Re-run evaluations in your own environment before making performance claims. This is a learning/portfolio project and should be validated before production use.

## Attribution
This repository is a curated portfolio version of a project archive. Review dependencies, third-party model licenses and source attribution before redistribution or commercial use.
