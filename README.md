# Hybrid Chatbot System

A Python portfolio prototype combining rule-based intent handling, semantic FAQ retrieval, an LLM fallback, sentiment analysis, multilingual responses, and a Streamlit chat interface.

> **Project status: In progress**
>
> This project is being developed as part of my technical portfolio. Current priorities include reviewing NLP dependencies, validating the Streamlit interface, and testing the rule-based, semantic retrieval, and LLM response paths. End-to-end testing is still pending.

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

## Current Status & Roadmap

- [x] Prepare portfolio README and setup guidance
- [x] Add a lightweight GitHub Actions Python syntax check
- [ ] Complete end-to-end local testing of the Streamlit interface
- [ ] Validate rule-based, semantic retrieval, and LLM response paths
- [ ] Capture screenshots from the running application
- [ ] Document any remaining runtime issues and evaluation results

## Project Notes

I maintain this repository as part of my technical portfolio. I am continuing to review, test, and document the implementation. The roadmap reflects current priorities and will be updated as work progresses. Third-party packages, models, and other included materials remain subject to their respective licenses.
