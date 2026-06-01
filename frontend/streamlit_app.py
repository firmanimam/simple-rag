"""Streamlit chat UI for the RAG pipeline."""
from __future__ import annotations

import httpx
import streamlit as st

import api_client as api

st.set_page_config(page_title="RAG Chat", page_icon="📚", layout="wide")

TYPE_BADGE = {"text": "📄 text", "ocr": "🔍 ocr", "caption": "🖼️ caption"}


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, sources?}
if "top_k" not in st.session_state:
    st.session_state.top_k = 20
if "top_n" not in st.session_state:
    st.session_state.top_n = 5


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📎 Sources ({len(sources)})", expanded=False):
        for i, s in enumerate(sources, start=1):
            badge = TYPE_BADGE.get(s.get("type", "text"), s.get("type", "text"))
            page = s.get("page")
            score = s.get("score")
            score_str = f" · score {score:.3f}" if isinstance(score, (int, float)) else ""
            page_str = f" · p.{page}" if page else ""
            st.markdown(
                f"**[{i}] {s.get('filename', 'unknown')}**{page_str} · {badge}{score_str}"
            )
            st.caption(s.get("snippet", ""))


def stream_tokens(question: str, sources_holder: list):
    """Generator for st.write_stream; collects sources as a side effect."""
    try:
        for event in api.query_stream(
            question, top_k=st.session_state.top_k, top_n=st.session_state.top_n
        ):
            etype = event.get("type")
            if etype == "token":
                yield event.get("content", "")
            elif etype == "sources":
                sources_holder.extend(event.get("sources", []))
            elif etype == "error":
                yield f"\n\n⚠️ **Error:** {event.get('message')}"
    except httpx.HTTPError as exc:
        yield f"\n\n⚠️ **Connection error:** {exc}"


# --------------------------------------------------------------------------- #
# Sidebar: upload, documents, settings
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("📚 RAG Pipeline")

    # Backend health
    try:
        h = api.health()
        st.success(f"Backend online · {h.get('doc_count', 0)} chunks indexed")
    except Exception:
        st.error("Backend offline — start the FastAPI server.")

    st.divider()
    st.subheader("Upload")
    uploads = st.file_uploader(
        "Drag & drop PDFs or images",
        type=["pdf", "png", "jpg", "jpeg", "bmp", "tiff", "webp", "gif"],
        accept_multiple_files=True,
    )
    if st.button("Ingest", type="primary", use_container_width=True, disabled=not uploads):
        files = [
            (f.name, f.getvalue(), f.type or "application/octet-stream") for f in uploads
        ]
        with st.spinner(f"Ingesting {len(files)} file(s)… (vision/OCR can take a while)"):
            try:
                res = api.ingest(files)
                st.toast(
                    f"Indexed {res['chunks_indexed']} chunks from "
                    f"{len(res['doc_ids'])} file(s).",
                    icon="✅",
                )
            except httpx.HTTPStatusError as exc:
                st.toast(f"Ingest failed: {exc.response.text}", icon="⚠️")
            except Exception as exc:
                st.toast(f"Ingest failed: {exc}", icon="⚠️")
        st.rerun()

    st.divider()
    st.subheader("Documents")
    try:
        docs = api.list_documents()
    except Exception:
        docs = []
    if not docs:
        st.caption("No documents yet. Upload something to get started.")
    for d in docs:
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.markdown(f"**{d['filename']}**")
            st.caption(f"{d.get('pages') or '?'} pages · {d['chunks']} chunks")
        with col2:
            if st.button("🗑️", key=f"del_{d['id']}", help="Delete document"):
                try:
                    api.delete_document(d["id"])
                    st.toast(f"Deleted {d['filename']}", icon="🗑️")
                except Exception as exc:
                    st.toast(f"Delete failed: {exc}", icon="⚠️")
                st.rerun()

    st.divider()
    with st.expander("⚙️ Settings"):
        st.session_state.top_k = st.slider("Candidates (top_k)", 5, 50, st.session_state.top_k)
        st.session_state.top_n = st.slider("Final chunks (top_n)", 1, 15, st.session_state.top_n)

    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------------------------------- #
# Main panel: chat
# --------------------------------------------------------------------------- #
st.title("💬 Ask your documents")

if not st.session_state.messages:
    st.info("Upload PDFs or images in the sidebar, then ask a question below.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []))

if prompt := st.chat_input("Ask a question about your documents…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        sources_holder: list = []
        with st.spinner("Retrieving & reranking…"):
            answer_text = st.write_stream(stream_tokens(prompt, sources_holder))
        render_sources(sources_holder)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer_text, "sources": sources_holder}
    )
