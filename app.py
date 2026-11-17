"""Streamlit chat UI for the HR/IT FAQ chatbot.

Run with:  streamlit run app.py

Streamlit re-runs this whole script from top to bottom on every interaction
(each message, each button click). st.session_state is what survives between
those re-runs, so it holds the chat history and the "already synced" flag.
"""

import openai
import streamlit as st

from rag.config import get_api_key
from rag.embeddings import MissingAPIKeyError
from rag.generation import CONTACT_SUGGESTION, cited_numbers, friendly_error, is_declined
from rag.index import EmptyDocumentsError, SettingsChangedError, index_status, sync_index
from rag.pipeline import answer_stream

st.set_page_config(page_title="HR & IT FAQ Assistant", page_icon="💬")
st.title("💬 HR & IT FAQ Assistant")
st.caption(
    "Answers come only from the company documents and cite their sources. Each question is "
    "answered on its own, so include all the details in every question. Answers are "
    "informational: HR or IT is the authority on any individual case."
)

# ------------------------------------------------------------------ API key check
if get_api_key() is None:
    st.error(
        "**No OpenAI API key found.** Copy `.env.example` to `.env`, paste your key after "
        "`OPENAI_API_KEY=`, then restart the app. README.md explains how to get a key."
    )
    st.stop()  # nothing below works without a key, so stop rendering here


# ------------------------------------------------------------------ index sync
def run_sync() -> None:
    """Syncs the index with documents/ and stores the outcome in session_state."""
    st.session_state.sync_error = None
    st.session_state.chat_blocked = False
    # st.status shows a collapsible progress box while the block inside runs.
    with st.status("Syncing documents...", expanded=False) as status:
        try:
            # The progress callback shows "Indexing hr/leave-policy.md (2/8)" in the status box.
            summary = sync_index(progress=lambda m: status.update(label=m))
            st.session_state.sync_summary = summary.describe()
            status.update(label="Documents up to date", state="complete")
        except (EmptyDocumentsError, SettingsChangedError) as exc:
            # No documents, or an index built with different settings: answering would be wrong.
            st.session_state.sync_error = str(exc)
            st.session_state.chat_blocked = True
            status.update(label="Index not ready", state="error")
        except (MissingAPIKeyError, openai.OpenAIError) as exc:
            # The sync failed (e.g. network). Keep using the existing index, if there is one.
            st.session_state.sync_error = "Re-indexing failed, using the existing index. " + friendly_error(exc)
            st.session_state.chat_blocked = index_status()["chunks"] == 0
            status.update(label="Sync failed", state="error")


# The flag makes the sync run once per browser session (on open or refresh), not on
# every message. A refresh starts a new session, so it re-syncs.
if "synced" not in st.session_state:
    run_sync()
    st.session_state.synced = True

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Index")
    stats = index_status()
    st.metric("Documents", stats["documents"])
    st.metric("Chunks", stats["chunks"])
    st.caption(f"Last indexed: {stats['last_indexed'] or 'never'} (UTC)")
    if st.button("Re-index documents", help="Use this if files changed while the page was open"):
        run_sync()
        st.rerun()  # redraw the page so the numbers above update
    if st.session_state.get("sync_summary"):
        with st.expander("Last sync"):
            st.text(st.session_state.sync_summary)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if st.session_state.get("sync_error"):
    st.warning(st.session_state.sync_error)


# ------------------------------------------------------------------ chat
def show_sources(chunks: list[dict], cited: set[int]) -> None:
    if not chunks:
        return
    with st.expander(f"Sources ({len(chunks)} passages retrieved)"):
        for n, c in enumerate(chunks, start=1):
            mark = "✅ cited" if n in cited else "retrieved"
            st.markdown(f"**[{n}] {c['title']} > {c['section']}**  \n`{c['source']}` · similarity {c['score']:.2f} · {mark}")
            st.text(c["text"])


if "messages" not in st.session_state:
    st.session_state.messages = []  # chat history: list of {role, content, sources, cited}

# Re-draw the conversation so far (Streamlit cleared the screen on this re-run).
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            show_sources(msg.get("sources", []), msg.get("cited", set()))

# The input box is disabled while the index isn't usable, so no question reads a bad index.
question = st.chat_input("Ask an HR or IT question...", disabled=st.session_state.chat_blocked)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the documents..."):
                result = answer_stream(question)  # retrieval happens here
            # write_stream prints text as it arrives and returns the full answer at the end.
            text = st.write_stream(result.stream)
            chunks = [c.__dict__ for c in result.chunks]  # plain dicts are safe to keep in session_state
        except (MissingAPIKeyError, openai.OpenAIError) as exc:
            text, chunks = f"**Error:** {friendly_error(exc)}", []
            st.markdown(text)

        if is_declined(text) and CONTACT_SUGGESTION not in text:
            st.info(CONTACT_SUGGESTION)
        cited = cited_numbers(text)
        show_sources(chunks, cited)

    st.session_state.messages.append(
        {"role": "assistant", "content": text, "sources": chunks, "cited": cited}
    )
