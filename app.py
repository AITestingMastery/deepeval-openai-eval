"""Streamlit app that evaluates LLM answers with DeepEval.

    streamlit run app.py

Two modes:
  💬 Chat                     question -> OpenAI -> answer -> 10 answer metrics
  📄 Chat with documents (RAG) question -> retrieve chunks from your documents ->
                              OpenAI answers from them -> answer metrics + retrieval metrics
"""

import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import streamlit as st
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
from openai import OpenAI

import rag

try:  # newer DeepEval
    from deepeval.test_case import SingleTurnParams as P
except ImportError:  # older DeepEval
    from deepeval.test_case import LLMTestCaseParams as P

load_dotenv()

st.set_page_config(page_title="LLM Evaluation", page_icon="🎯", layout="wide")

CHAT_MODE = "💬 Chat"
RAG_MODE = "📄 Chat with documents (RAG)"

ANSWER_METRICS = [
    "Answer Relevancy",
    "Correctness",
    "Coverage",
    "Faithfulness",
    "Structure Quality",
    "Prompt Adherence",
    "Traceability",
    "Redundancy",
    "Helpfulness",
    "Clarity",
]
RETRIEVAL_METRICS = ["Contextual Relevancy", "Contextual Precision", "Contextual Recall"]

METRIC_HELP = {
    "Answer Relevancy": "Does the answer address the question?",
    "Correctness": "Is the answer factually correct (vs expected answer if given)?",
    "Coverage": "Does it cover every part of the question / every key point expected?",
    "Faithfulness": "Is every claim supported by the context (no made-up facts)?",
    "Structure Quality": "Is it well organised and formatted?",
    "Prompt Adherence": "Does it follow the system prompt and the user's instructions?",
    "Traceability": "Can each claim be traced to a specific part of the context?",
    "Redundancy": "Is it free of repetition and filler? (higher = less redundant)",
    "Helpfulness": "Is it useful and actionable for the user?",
    "Clarity": "Is it easy to read and understand?",
    "Contextual Relevancy": "Retrieval: are the retrieved chunks relevant to the question?",
    "Contextual Precision": "Retrieval: are the most useful chunks ranked at the top? (needs expected answer)",
    "Contextual Recall": "Retrieval: did we retrieve everything needed for the expected answer? (needs expected answer)",
}

DEFAULT_CHAT_PROMPT = os.getenv(
    "SYSTEM_PROMPT", "You are a helpful assistant. Answer clearly and concisely."
)

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is missing. Add it to your .env file and restart the app.")
    st.stop()

client = OpenAI()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", [CHAT_MODE, RAG_MODE])

    chat_model = st.text_input("Chat model", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    judge_model = st.text_input("Scoring model", os.getenv("DEEPEVAL_JUDGE_MODEL", "gpt-4o-mini"))
    threshold = st.slider("Pass threshold", 0.0, 1.0, float(os.getenv("PASS_THRESHOLD", "0.7")), 0.05)

    index = None
    if mode == RAG_MODE:
        st.divider()
        st.subheader("Documents")
        doc_source = st.radio("Source", ["Sample documents (BrightPath HR)", "Upload my own"])
        uploads = []
        if doc_source == "Upload my own":
            uploads = st.file_uploader(
                "PDF, Word, TXT or Markdown", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True
            )
        top_k = st.slider("Chunks to retrieve (top-k)", 1, 8, 3)
        chunk_size = st.slider("Chunk size (characters)", 300, 2000, 800, 100)

        if doc_source == "Upload my own":
            docs = [(f.name, rag.read_file(f.name, f.getvalue())) for f in (uploads or [])]
        else:
            docs = rag.load_sample_docs()

        if docs:
            key = hashlib.md5(
                (str(chunk_size) + "".join(name + text for name, text in docs)).encode()
            ).hexdigest()
            if st.session_state.get("index_key") != key:
                with st.spinner("Reading, chunking and embedding documents..."):
                    st.session_state.index = rag.build_index(client, docs, chunk_size=chunk_size)
                    st.session_state.index_key = key
            index = st.session_state.index
            st.success(f"Indexed {len(index.chunks)} chunks from {len(docs)} file(s)")
            with st.expander("Files in the index"):
                for name, text in docs:
                    st.markdown(f"- **{name}** ({len(text):,} characters)")
        else:
            st.info("Upload at least one document to start.")

    st.divider()
    system_prompt = st.text_area(
        "System prompt",
        rag.RAG_SYSTEM_PROMPT if mode == RAG_MODE else DEFAULT_CHAT_PROMPT,
        height=120,
        key=f"system_prompt_{mode}",
    )
    metric_options = ANSWER_METRICS + (RETRIEVAL_METRICS if mode == RAG_MODE else [])
    selected_metrics = st.multiselect(
        "Metrics to run", metric_options, default=metric_options, key=f"metrics_{mode}"
    )
    st.divider()
    st.caption("Optional, used for your next question:")
    expected = st.text_area(
        "Expected answer",
        placeholder="Makes Correctness and Coverage compare against this. "
        "In RAG mode it also enables Contextual Precision and Recall.",
        height=90,
    )
    source_context = ""
    if mode == CHAT_MODE:
        source_context = st.text_area(
            "Source context",
            placeholder="Paste text the answer should be based on. "
            "Used by Faithfulness and Traceability.",
            height=120,
        )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


# ---------------- LLM call (chat mode) ----------------
def ask_llm(question: str, context: str) -> str:
    user_content = question
    if context:
        user_content = f"Use this context to answer.\n\nContext:\n{context}\n\nQuestion: {question}"
    response = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
    )
    return (response.choices[0].message.content or "").strip()


# ---------------- Metrics ----------------
def geval(name: str, criteria: str, params: list) -> GEval:
    return GEval(
        name=name, criteria=criteria, evaluation_params=params,
        threshold=threshold, model=judge_model, async_mode=False,
    )


def build_metric(name: str, has_expected: bool, has_context: bool):
    io = [P.INPUT, P.ACTUAL_OUTPUT]
    io_exp = io + [P.EXPECTED_OUTPUT]
    io_ctx = io + [P.RETRIEVAL_CONTEXT]
    common = dict(threshold=threshold, model=judge_model, include_reason=True, async_mode=False)

    # ----- retrieval metrics (RAG) -----
    if name == "Contextual Relevancy":
        return ContextualRelevancyMetric(**common)
    if name == "Contextual Precision":
        return ContextualPrecisionMetric(**common) if has_expected else None
    if name == "Contextual Recall":
        return ContextualRecallMetric(**common) if has_expected else None

    # ----- answer metrics -----
    if name == "Answer Relevancy":
        return AnswerRelevancyMetric(**common)
    if name == "Correctness":
        if has_expected:
            return geval(name, "Determine whether the actual output is factually correct and "
                         "consistent with the expected output. Minor wording differences are fine.", io_exp)
        return geval(name, "Determine whether the actual output is factually accurate based on "
                     "well-established knowledge. Penalize any false or misleading statements.", io)
    if name == "Coverage":
        if has_expected:
            return geval(name, "Check whether the actual output covers all the key points present in "
                         "the expected output. Penalize each missing key point.", io_exp)
        return geval(name, "Check whether the actual output addresses every part and sub-question of "
                     "the input. Penalize any part of the question left unanswered.", io)
    if name == "Faithfulness":
        if has_context:
            return FaithfulnessMetric(**common)
        return geval(name, "Check whether the actual output avoids hallucinations: no invented facts, "
                     "fake numbers, fake sources or unsupported claims. Penalize fabricated content.", io)
    if name == "Structure Quality":
        return geval(name, "Evaluate how well the actual output is organised: logical order, sensible "
                     "paragraphs, and lists, headings or code blocks where they help. Penalize walls of "
                     "text or messy formatting.", io)
    if name == "Prompt Adherence":
        return geval(name, "The input contains a system prompt and a user question. Check whether the "
                     "actual output follows every instruction in both: required format, length, tone, "
                     "language and constraints. Penalize each instruction that is ignored.", io)
    if name == "Traceability":
        if has_context:
            return geval(name, "Check whether each claim in the actual output can be traced back to a "
                         "specific part of the retrieval context. Penalize claims that cannot be located "
                         "in the context.", io_ctx)
        return geval(name, "Check whether the claims in the actual output are specific and verifiable, "
                     "and whether the reasoning behind conclusions is shown. Penalize vague or "
                     "unverifiable claims.", io)
    if name == "Redundancy":
        return geval(name, "Score how free the actual output is from redundancy. A high score means no "
                     "repeated points, no restating the question, and no filler. Penalize repetition.", io)
    if name == "Helpfulness":
        return geval(name, "Judge how useful and actionable the actual output is for the person who "
                     "asked the input.", io)
    if name == "Clarity":
        return geval(name, "Judge how clear and easy to understand the actual output is: simple "
                     "language, well-defined terms, and no ambiguity.", io)
    raise ValueError(name)


def run_one(name: str, question: str, answer: str, expected_answer: str, contexts: list[str]) -> dict:
    # Prompt Adherence needs to see the system prompt as part of the input
    input_text = (
        f"System prompt: {system_prompt}\n\nUser question: {question}"
        if name == "Prompt Adherence"
        else question
    )
    test_case = LLMTestCase(
        input=input_text,
        actual_output=answer,
        expected_output=expected_answer or None,
        retrieval_context=contexts or None,
    )
    group = "retrieval" if name in RETRIEVAL_METRICS else "answer"
    try:
        metric = build_metric(name, bool(expected_answer), bool(contexts))
        if metric is None:
            return {"metric": name, "group": group, "score": None, "passed": None,
                    "reason": "Skipped: fill 'Expected answer' in the sidebar to run this metric."}
        metric.measure(test_case)
        score = round(metric.score or 0.0, 2)
        return {"metric": name, "group": group, "score": score,
                "passed": score >= threshold, "reason": metric.reason or ""}
    except Exception as e:  # keep the chat working if one metric fails
        return {"metric": name, "group": group, "score": None, "passed": False, "reason": f"Error: {e}"}


def score_all(question: str, answer: str, expected_answer: str, contexts: list[str]) -> list[dict]:
    """Run all selected metrics in parallel."""
    with ThreadPoolExecutor(max_workers=max(1, len(selected_metrics))) as pool:
        futures = [
            pool.submit(run_one, name, question, answer, expected_answer, contexts)
            for name in selected_metrics
        ]
        return [f.result() for f in futures]


# ---------------- Display ----------------
def metric_grid(scores: list[dict]) -> None:
    per_row = 5
    for start in range(0, len(scores), per_row):
        cols = st.columns(per_row)
        for col, s in zip(cols, scores[start:start + per_row]):
            if s["passed"] is None:
                col.metric(s["metric"], "—", "Skipped", delta_color="off", help=METRIC_HELP.get(s["metric"]))
                continue
            col.metric(
                s["metric"],
                "—" if s["score"] is None else f"{s['score']:.2f}",
                "✅ Pass" if s["passed"] else "❌ Fail",
                delta_color="normal" if s["passed"] else "inverse",
                help=METRIC_HELP.get(s["metric"]),
            )


def show_scores(scores: list[dict]) -> None:
    ran = [s for s in scores if s["score"] is not None]
    if ran:
        passed = sum(1 for s in ran if s["passed"])
        overall = sum(s["score"] for s in ran) / len(ran)
        st.markdown(f"**Overall: {overall:.2f}** · {passed}/{len(ran)} metrics passed")

    answer_scores = [s for s in scores if s["group"] == "answer"]
    retrieval_scores = [s for s in scores if s["group"] == "retrieval"]
    if retrieval_scores:
        st.markdown("🔎 **Retrieval quality**: did we find the right document chunks?")
        metric_grid(retrieval_scores)
        st.markdown("✍️ **Answer quality**: is the answer good?")
    metric_grid(answer_scores)

    with st.expander("Why these scores?"):
        for s in scores:
            st.markdown(f"**{s['metric']}:** {s['reason']}")


def show_retrieved(retrieved: list[dict]) -> None:
    with st.expander(f"📄 Retrieved chunks ({len(retrieved)})"):
        for i, r in enumerate(retrieved, start=1):
            st.markdown(f"**#{i} · {r['source']} · chunk {r['chunk_id']} · similarity {r['score']:.2f}**")
            st.text(r["text"])


def show_timings(t: dict) -> None:
    parts = [f"{k}: {v:.1f}s" for k, v in t.items()]
    st.caption("⏱️ " + " · ".join(parts))


# ---------------- Header ----------------
st.title("🎯 LLM Evaluation")
if mode == RAG_MODE:
    st.caption("Ask a question about your documents. The app retrieves the most relevant chunks, "
               "OpenAI answers from them, and DeepEval scores both the retrieval and the answer.")
    with st.expander("How does RAG work here?"):
        st.markdown(
            "1. **Load**: documents are read (PDF, Word, TXT, Markdown).\n"
            "2. **Chunk**: each document is split into small overlapping pieces.\n"
            "3. **Embed**: each chunk is turned into a vector with an OpenAI embedding model.\n"
            "4. **Retrieve**: your question is embedded too, and the top-k most similar chunks are found.\n"
            "5. **Generate**: OpenAI answers using only those chunks.\n"
            "6. **Evaluate**: DeepEval checks whether the right chunks were found (retrieval metrics) "
            "and whether the answer is good and faithful to them (answer metrics)."
        )
    if index is not None:
        st.caption("Try: *How many casual leaves do I get?* · *What is the hotel limit in Hyderabad?* · "
                   "*Can I work from Goa for a month?* · *What is the company's dress code?*")
else:
    st.caption("Ask anything. OpenAI answers, and DeepEval scores the answer on up to 10 metrics.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("expected"):
            st.caption(f"Expected: {msg['expected']}")
        if msg.get("context"):
            st.caption("Source context provided")
        if msg.get("retrieved"):
            show_retrieved(msg["retrieved"])
        if msg.get("scores"):
            show_scores(msg["scores"])
        if msg.get("timings"):
            show_timings(msg["timings"])

# ---------------- Chat input ----------------
chat_disabled = mode == RAG_MODE and index is None
if question := st.chat_input("Ask a question...", disabled=chat_disabled):
    expected_answer = (expected or "").strip()
    pasted_context = (source_context or "").strip()
    st.session_state.messages.append(
        {"role": "user", "content": question, "expected": expected_answer, "context": bool(pasted_context)}
    )
    with st.chat_message("user"):
        st.markdown(question)
        if expected_answer:
            st.caption(f"Expected: {expected_answer}")
        if pasted_context:
            st.caption("Source context provided")

    with st.chat_message("assistant"):
        timings, retrieved_view = {}, []
        try:
            if mode == RAG_MODE:
                t0 = time.perf_counter()
                with st.spinner("Searching documents..."):
                    retrieved = index.search(question, top_k=top_k)
                timings["Retrieval"] = time.perf_counter() - t0
                retrieved_view = [
                    {"source": r.chunk.source, "chunk_id": r.chunk.chunk_id,
                     "score": r.score, "text": r.chunk.text}
                    for r in retrieved
                ]
                contexts = [r.chunk.text for r in retrieved]
                t0 = time.perf_counter()
                with st.spinner("Generating answer from documents..."):
                    answer = rag.answer_with_rag(client, chat_model, question, retrieved, system_prompt)
                timings["Generation"] = time.perf_counter() - t0
            else:
                contexts = [pasted_context] if pasted_context else []
                t0 = time.perf_counter()
                with st.spinner("Getting answer..."):
                    answer = ask_llm(question, pasted_context)
                timings["Generation"] = time.perf_counter() - t0
        except Exception as e:
            st.error(f"OpenAI error: {e}")
            st.stop()

        st.markdown(answer)
        if retrieved_view:
            show_retrieved(retrieved_view)

        scores = []
        if selected_metrics:
            t0 = time.perf_counter()
            with st.spinner(f"Scoring on {len(selected_metrics)} metrics..."):
                scores = score_all(question, answer, expected_answer, contexts)
            timings["Scoring"] = time.perf_counter() - t0
            show_scores(scores)
        show_timings(timings)

    st.session_state.messages.append({
        "role": "assistant", "content": answer, "scores": scores,
        "retrieved": retrieved_view, "timings": timings, "mode": mode,
    })

# ---------------- History table ----------------
rows = []
msgs = st.session_state.messages
for i in range(0, len(msgs) - 1, 2):
    q, a = msgs[i], msgs[i + 1]
    row = {
        "Mode": "RAG" if a.get("mode") == RAG_MODE else "Chat",
        "Question": q["content"],
        "Answer": a["content"],
    }
    if a.get("retrieved"):
        row["Sources"] = ", ".join(sorted({r["source"] for r in a["retrieved"]}))
    for s in a.get("scores", []):
        row[s["metric"]] = s["score"]
    rows.append(row)

if rows:
    st.divider()
    st.subheader("Score history")
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True)
    st.download_button("Download as CSV", df.to_csv(index=False), "scores.csv", "text/csv")