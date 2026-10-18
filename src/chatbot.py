# -*- coding: utf-8 -*-
"""
Hybrid Chatbot System
defined for:
- structured test query execution
- richer output logging
- primary data generation
"""

# Required Libraries
import os
import re
import warnings
import openai
import spacy
import wikipediaapi
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from transformers import pipeline
from textblob import TextBlob
from googletrans import Translator

# Suppress warnings
warnings.filterwarnings("ignore")

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize NLP tools
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="EnhancedChatbot/2.0 (https://example.com; contact@example.com)"
)

# Memory for multi-turn conversations
user_memory = []
user_preferences = {}
translator = Translator()

# Initialize zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Rule-based responses
rule_based_responses = {
    "greet": "Hello! How can I assist you today?",
    "farewell": "Goodbye! Have a great day!",
    "name": "I’m your friendly assistant chatbot.",
    "business_hours": "Our business hours are 9 AM to 5 PM, Monday to Friday.",
    "pricing": "Our products range from $10 to $500, depending on your requirements.",
    "returns": "You can return items within 30 days of purchase.",
    "shipping": "We offer free shipping for orders above $50."
}

# Expanded FAQs and Their Embeddings
faq_data = {
    "business_hours": "Our business operates from 9 AM to 5 PM, Monday to Friday.",
    "pricing": "Our products range from $10 to $500, depending on your requirements.",
    "returns": "You can return items within 30 days of purchase.",
    "shipping": "We offer free shipping for orders above $50.",
    "support": "You can reach out to our support team at support@example.com.",
    "payment_methods": "We accept Visa, MasterCard, PayPal, and Apple Pay.",
    "order_changes": "You can modify your order within 24 hours by contacting our support team.",
    "account_reset": "You can reset your password by clicking 'Forgot Password' on the login page.",
    "international_shipping": "Yes, we offer international shipping to select countries.",
    "warranty": "All our products come with a one-year warranty."
}

# Precompute FAQ embeddings
faq_keys = list(faq_data.keys())
faq_embeddings = [embedder.encode(faq) for faq in faq_keys]


# ---------------------------------------------------
# Helper: standard output format for dissertation logs
# ---------------------------------------------------
def format_chatbot_output(response, component_used, matched_intent_or_faq="", confidence_score=""):
    return {
        "response": response,
        "component_used": component_used,
        "matched_intent_or_faq": matched_intent_or_faq,
        "confidence_score": confidence_score
    }


# ---------------------------------------------------
# Sentiment Analysis
# ---------------------------------------------------
def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "positive"
    elif analysis.sentiment.polarity < 0:
        return "negative"
    else:
        return "neutral"


# ---------------------------------------------------
# Query Contextualization
# ---------------------------------------------------
def update_preferences(query):
    global user_preferences
    if "I like" in query:
        preference = query.split("I like")[-1].strip()
        user_preferences["likes"] = preference


def contextualize_response(response):
    global user_preferences
    if "likes" in user_preferences:
        response += f" By the way, I remember you like {user_preferences['likes']}."
    return response


# ---------------------------------------------------
# Dashboard
# ---------------------------------------------------
def chatbot_dashboard():
    st.title("Chatbot Analytics")
    st.metric("Total Queries", len(user_memory))

    positive_queries = sum(1 for m in user_memory if m["sentiment"] == "positive")
    negative_queries = sum(1 for m in user_memory if m["sentiment"] == "negative")

    st.metric("Positive Sentiment Queries", positive_queries)
    st.metric("Negative Sentiment Queries", negative_queries)


# ---------------------------------------------------
# Intent Detection
# ---------------------------------------------------
def detect_intent(query):
    """
    Detects intent using zero-shot classification from Hugging Face transformers.
    Returns intent and confidence score.
    """
    candidate_labels = list(rule_based_responses.keys())
    result = classifier(query, candidate_labels)
    intent = result["labels"][0]
    confidence = result["scores"][0]

    if confidence > 0.85:
        return intent, confidence
    return None, None


# ---------------------------------------------------
# Semantic Search
# ---------------------------------------------------
def semantic_search(query):
    """
    Performs semantic similarity search on FAQs using cosine similarity.

    Returns:
    {
        "answer": str,
        "score": float,
        "matched_faq": str
    }
    or None
    """
    query_embedding = embedder.encode(query).reshape(1, -1)
    faq_embeddings_array = np.array(faq_embeddings)

    similarities = cosine_similarity(query_embedding, faq_embeddings_array)[0]
    best_match_idx = np.argmax(similarities)
    best_similarity = similarities[best_match_idx]

    if best_similarity > 0.70:
        matched_faq = faq_keys[best_match_idx]
        return {
            "answer": faq_data[matched_faq],
            "score": float(best_similarity),
            "matched_faq": matched_faq
        }

    return None


# ---------------------------------------------------
# LLM Calls
# ---------------------------------------------------
def call_llm(query, memory):
    """
    Sends a query to OpenAI GPT and returns its response.
    """
    try:
        context = "\n".join(
            [f"User: {m['user']}\nAssistant: {m['bot']}" for m in memory[-5:]]
        ) if memory else ""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a customer service assistant for AcmeCorp. "
                    "Provide accurate and specific information about our products, "
                    "services, and policies to help customers with their inquiries."
                )
            },
            {
                "role": "user",
                "content": query
            }
        ]

        if context:
            messages.insert(1, {"role": "assistant", "content": context})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150
        )

        return response["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Error: {str(e)}"


def call_llm_with_context(query, memory, system_prompt):
    """
    Sends a query to OpenAI GPT with provided context and returns its response.
    """
    try:
        context = "\n".join(
            [f"User: {m['user']}\nAssistant: {m['bot']}" for m in memory[-5:]]
        ) if memory else ""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        if context:
            messages.insert(1, {"role": "assistant", "content": context})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150
        )

        return response["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Error: {str(e)}"


# ---------------------------------------------------
# Wikipedia Fact-Checking
# ---------------------------------------------------
def fact_check(query, response):
    """
    Performs fact-checking using Wikipedia summaries.
    Returns response and similarity score.
    """
    page = wiki_wiki.page(query)

    if page.exists():
        wiki_summary = page.summary[:500]

        response_embedding = embedder.encode(response).reshape(1, -1)
        wiki_embedding = embedder.encode(wiki_summary).reshape(1, -1)

        similarity = cosine_similarity(response_embedding, wiki_embedding)[0][0]

        if similarity < 0.5:
            warning = (
                f"Warning: The response may not be accurate. "
                f"For reference, Wikipedia says: {wiki_summary}"
            )
            return warning, similarity

        return response, similarity

    return response, None


# ---------------------------------------------------
# Confidence Heuristic
# ---------------------------------------------------
def evaluate_confidence(response):
    """
    Evaluates confidence of GPT responses based on token count and heuristics.
    Returns confidence score.
    """
    token_count = len(response.split())

    if token_count < 10:
        confidence = 0.5
    elif token_count < 20:
        confidence = 0.7
    else:
        confidence = 0.9

    return confidence


# ---------------------------------------------------
# Traditional Rule-Based Chatbot
# ---------------------------------------------------
def traditional_chatbot(query):
    """
    Uses only the rule-based chatbot logic to generate responses.
    Returns structured output for logging.
    """
    intent, intent_confidence = detect_intent(query)

    if intent in rule_based_responses and intent_confidence:
        response = rule_based_responses[intent]
        return format_chatbot_output(
            response=response,
            component_used="rule_based",
            matched_intent_or_faq=intent,
            confidence_score=round(intent_confidence * 100, 2)
        )

    return format_chatbot_output(
        response="I’m sorry, I didn’t understand that.",
        component_used="none",
        matched_intent_or_faq="",
        confidence_score=0
    )


# ---------------------------------------------------
# LLM-Only Chatbot
# ---------------------------------------------------
def llm_chatbot(query):
    """
    Uses only the LLM to generate responses.
    Returns structured output for logging.
    """
    system_prompt = (
        "You are a customer service assistant for AcmeCorp. "
        "Provide accurate and specific information about our products, services, "
        "and policies to help customers with their inquiries."
    )

    llm_response = call_llm_with_context(query, user_memory, system_prompt)
    llm_confidence = evaluate_confidence(llm_response) * 100

    fact_checked_response, fact_similarity = fact_check(query, llm_response)
    if fact_similarity is not None:
        llm_confidence = fact_similarity * 100

    return format_chatbot_output(
        response=fact_checked_response,
        component_used="llm",
        matched_intent_or_faq="",
        confidence_score=round(llm_confidence, 2)
    )


# ---------------------------------------------------
# Hybrid Chatbot
# ---------------------------------------------------
def hybrid_chatbot(query, target_language="en"):
    """
    Combines rule-based, semantic search, LLM responses, fact-checking,
    sentiment analysis, contextual memory and multilingual support.

    Returns structured output for dissertation logging.
    """

    # Translate query to English if needed
    original_query = query
    if target_language != "en":
        query = translator.translate(query, dest="en").text

    # Sentiment + preference update
    sentiment = analyze_sentiment(query)
    update_preferences(query)

    # -----------------------------
    # Step 1: Rule-based layer
    # -----------------------------
    intent, intent_confidence = detect_intent(query)

    if intent in rule_based_responses and intent_confidence:
        doc = nlp(query)
        person_entities = [ent.text.lower() for ent in doc.ents if ent.label_ == "PERSON"]
        assistant_names = ["you", "your name", "assistant", "chatbot"]

        if not person_entities or any(name in query.lower() for name in assistant_names):
            response = rule_based_responses[intent]
            response = contextualize_response(response)

            if target_language != "en":
                response = translator.translate(response, dest=target_language).text

            return format_chatbot_output(
                response=response,
                component_used="rule_based",
                matched_intent_or_faq=intent,
                confidence_score=round(intent_confidence * 100, 2)
            )

    # -----------------------------
    # Step 2: Semantic retrieval
    # -----------------------------
    semantic_result = semantic_search(query)

    if semantic_result:
        response = contextualize_response(semantic_result["answer"])

        if target_language != "en":
            response = translator.translate(response, dest=target_language).text

        return format_chatbot_output(
            response=response,
            component_used="semantic_search",
            matched_intent_or_faq=semantic_result["matched_faq"],
            confidence_score=round(semantic_result["score"] * 100, 2)
        )

    # -----------------------------
    # Step 3: LLM fallback
    # -----------------------------
    top_n = 3
    similarities = []

    query_embedding = embedder.encode(query).reshape(1, -1)
    for faq in faq_keys:
        faq_embedding = embedder.encode(faq).reshape(1, -1)
        similarity = cosine_similarity(query_embedding, faq_embedding)[0][0]
        similarities.append(similarity)

    top_indices = np.argsort(similarities)[-top_n:]
    context_faqs = [list(faq_data.values())[i] for i in reversed(top_indices)]

    system_prompt = (
        "You are a customer service assistant for AcmeCorp. "
        "Here are some company policies and information:\n"
        f"- {chr(10).join(context_faqs)}\n"
        "Provide accurate and specific information about our products, services, "
        "and policies to help customers with their inquiries."
    )

    llm_response = call_llm_with_context(query, user_memory, system_prompt)
    llm_response = contextualize_response(llm_response)

    llm_confidence = evaluate_confidence(llm_response) * 100

    fact_checked_response, fact_similarity = fact_check(query, llm_response)
    if fact_similarity is not None:
        llm_confidence = fact_similarity * 100

    if target_language != "en":
        fact_checked_response = translator.translate(
            fact_checked_response,
            dest=target_language
        ).text

    # Save conversation memory
    user_memory.append({
        "user": original_query,
        "bot": fact_checked_response,
        "sentiment": sentiment
    })

    return format_chatbot_output(
        response=fact_checked_response,
        component_used="llm_fallback",
        matched_intent_or_faq="",
        confidence_score=round(llm_confidence, 2)
    )


# ---------------------------------------------------
# Multilingual chatbot wrapper
# ---------------------------------------------------
def multilingual_chatbot(query, target_language="en"):
    """
    Wrapper for multilingual use.
    Returns translated final response and confidence.
    """
    result = hybrid_chatbot(query, target_language=target_language)

    return result["response"], result["confidence_score"]


# ---------------------------------------------------
# Evaluation Functions
# ---------------------------------------------------
def compute_cosine_similarity_score(response1, response2):
    embedding1 = embedder.encode(response1).reshape(1, -1)
    embedding2 = embedder.encode(response2).reshape(1, -1)
    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    return similarity


def compute_bleu_score(reference, candidate):
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    score = sentence_bleu([reference_tokens], candidate_tokens)
    return score


def compute_rouge_score(reference, candidate):
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)

    rouge_1_f = scores[0]["rouge-1"]["f"]
    rouge_2_f = scores[0]["rouge-2"]["f"]
    rouge_l_f = scores[0]["rouge-l"]["f"]

    return (rouge_1_f + rouge_2_f + rouge_l_f) / 3


def compute_fuzzy_score(reference, candidate):
    score = fuzz.token_set_ratio(reference, candidate) / 100.0
    return score


def check_response(actual_response, expected_response, method="cosine"):
    """
    Checks whether the actual response matches the expected response
    using different evaluation methods.
    Returns True if acceptable, False otherwise.
    """
    if method == "cosine":
        return compute_cosine_similarity_score(actual_response, expected_response) >= 0.75
    elif method == "bleu":
        return compute_bleu_score(expected_response, actual_response) >= 0.30
    elif method == "rouge":
        return compute_rouge_score(expected_response, actual_response) >= 0.30
    elif method == "fuzzy":
        return compute_fuzzy_score(expected_response, actual_response) >= 0.75
    else:
        raise ValueError("Unsupported evaluation method. Choose from: cosine, bleu, rouge, fuzzy.")