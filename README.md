# Hybrid Chatbot System

A Python portfolio prototype combining rule-based intent handling, semantic FAQ retrieval, an LLM fallback, sentiment analysis, multilingual responses, and a Streamlit chat interface.

## Features
- Rule-based, semantic-retrieval and LLM response paths
- Multilingual response option
- Conversation preference/context prototype
- Sample query dataset
- Script to run sample queries and write raw responses to CSV

## Setup

Use Python 3.10 or 3.11 in a virtual environment. Install the packages and English spaCy model:

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Set `OPENAI_API_KEY` in your environment before using the LLM response path. Never commit API keys or `.env` files.

Run the Streamlit interface from the repository root:

```bash
streamlit run src/streamlit_app.py
```

Run sample queries from the repository root:

```bash
python scripts/run_queries.py
```

The script writes raw responses to `outputs/run_results.csv`. Model downloads and API calls may take time or incur usage charges.

## Evaluation and limitations

The query CSV contains test prompts and expected responses, not verified model benchmark results. Run the evaluation in your own environment before reporting accuracy or performance. The prototype loads several NLP models and may require substantial first-run downloads and memory. Review the implementation before production use.

## Attribution

This is a curated portfolio copy of a supplied project archive. Review third-party package and model licenses before redistribution or commercial use.
