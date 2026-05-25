import sys

import pytest

sys.argv = sys.argv[:1]

from lightrag.api.routers.document_routes import (  # noqa: E402
    DocStatusResponse,
    normalize_file_path,
    pipeline_index_texts,
)
from lightrag import operate  # noqa: E402
from lightrag.base import DocStatus  # noqa: E402
from lightrag.constants import PROCESS_OPTION_CHUNK_FIXED  # noqa: E402
from lightrag.pipeline import _PipelineMixin  # noqa: E402
from lightrag.utils_pipeline import build_chunks_dict_from_chunking_result  # noqa: E402


class DummyRAG:
    def __init__(self):
        self.enqueued_calls = []
        self.processed = False

    async def apipeline_enqueue_documents(
        self,
        input,
        file_paths=None,
        track_id=None,
        process_options=None,
        context_chunk_headers=None,
    ):
        self.enqueued_calls.append(
            {
                "input": input,
                "file_paths": file_paths,
                "track_id": track_id,
                "process_options": process_options,
                "context_chunk_headers": context_chunk_headers,
            }
        )

    async def apipeline_process_enqueue_documents(self):
        self.processed = True


class CaptureDocStatus:
    def __init__(self):
        self.upserts = []

    async def upsert(self, data):
        self.upserts.append(data)


class DummyPipeline(_PipelineMixin):
    def __init__(self):
        self.doc_status = CaptureDocStatus()


class CaptureKV:
    def __init__(self):
        self.upserts = []

    async def filter_keys(self, keys):
        return set(keys)

    async def upsert(self, data):
        self.upserts.append(data)


@pytest.mark.asyncio
async def test_pipeline_index_texts_rejects_missing_file_sources():
    rag = DummyRAG()

    with pytest.raises(ValueError, match="valid file source"):
        await pipeline_index_texts(
            rag,
            texts=["alpha"],
            file_sources=[None],
            track_id="track-1",
        )

    assert rag.enqueued_calls == []
    assert rag.processed is False


@pytest.mark.asyncio
async def test_pipeline_index_texts_normalizes_file_sources_to_basename():
    rag = DummyRAG()

    await pipeline_index_texts(
        rag,
        texts=["alpha"],
        file_sources=["/tmp/source/alpha.txt"],
        track_id="track-1",
    )

    assert rag.enqueued_calls == [
        {
            "input": ["alpha"],
            "file_paths": ["alpha.txt"],
            "track_id": "track-1",
            "process_options": PROCESS_OPTION_CHUNK_FIXED,
            "context_chunk_headers": None,
        }
    ]
    assert rag.processed is True


@pytest.mark.asyncio
async def test_pipeline_index_texts_forwards_context_chunk_headers():
    rag = DummyRAG()

    await pipeline_index_texts(
        rag,
        texts=["alpha", "beta"],
        file_sources=["alpha.txt", "beta.txt"],
        track_id="track-headers",
        context_chunk_headers=["Title: Alpha", "Title: Beta"],
    )

    assert rag.enqueued_calls == [
        {
            "input": ["alpha", "beta"],
            "file_paths": ["alpha.txt", "beta.txt"],
            "track_id": "track-headers",
            "process_options": PROCESS_OPTION_CHUNK_FIXED,
            "context_chunk_headers": ["Title: Alpha", "Title: Beta"],
        }
    ]
    assert rag.processed is True


@pytest.mark.asyncio
async def test_ainsert_forwards_context_chunk_headers_to_pipeline():
    from lightrag import LightRAG

    rag = LightRAG.__new__(LightRAG)
    rag._addon_params = {}
    captured = {}

    async def fake_enqueue(*args, **kwargs):
        captured["enqueue_args"] = args
        captured["enqueue_kwargs"] = kwargs

    async def fake_process():
        captured["processed"] = True

    rag.apipeline_enqueue_documents = fake_enqueue
    rag.apipeline_process_enqueue_documents = fake_process

    track_id = await rag.ainsert(
        ["alpha", "beta"],
        ids=["doc-alpha", "doc-beta"],
        file_paths=["alpha.txt", "beta.txt"],
        track_id="track-headers",
        context_chunk_headers=["Title: Alpha", "Title: Beta"],
    )

    assert track_id == "track-headers"
    assert captured["enqueue_args"] == (
        ["alpha", "beta"],
        ["doc-alpha", "doc-beta"],
        ["alpha.txt", "beta.txt"],
        "track-headers",
    )
    assert captured["enqueue_kwargs"]["context_chunk_headers"] == [
        "Title: Alpha",
        "Title: Beta",
    ]
    assert captured["processed"] is True


def test_doc_status_response_uses_non_null_unknown_source():
    response = DocStatusResponse(
        id="doc-1",
        content_summary="summary",
        content_length=5,
        status=DocStatus.PENDING,
        created_at="2026-03-19T00:00:00+00:00",
        updated_at="2026-03-19T00:00:00+00:00",
        file_path=normalize_file_path(None),
    )

    assert response.file_path == "unknown_source"


@pytest.mark.asyncio
async def test_error_document_enqueue_canonicalizes_file_path_before_upsert():
    rag = DummyPipeline()

    await rag.apipeline_enqueue_error_documents(
        [
            {
                "file_path": "/tmp/uploads/report.[native-Fi].pdf",
                "error_description": "bad file",
                "original_error": "parse failed",
            }
        ],
        track_id="track-1",
    )

    saved = next(iter(rag.doc_status.upserts[0].values()))
    assert saved["file_path"] == "report.pdf"


@pytest.mark.asyncio
async def test_custom_chunks_use_canonical_unknown_source_before_upsert():
    from lightrag import LightRAG

    rag = LightRAG.__new__(LightRAG)
    rag.full_docs = CaptureKV()
    rag.text_chunks = CaptureKV()
    rag.chunks_vdb = CaptureKV()
    rag.tokenizer = type("Tokenizer", (), {"encode": lambda self, text: [text]})()

    async def _process_extract_entities(chunks):
        return []

    async def _insert_done():
        return None

    rag._process_extract_entities = _process_extract_entities
    rag._insert_done = _insert_done

    await rag.ainsert_custom_chunks("full text", ["chunk text"], doc_id="doc-1")

    assert rag.full_docs.upserts[0]["doc-1"]["file_path"] == "unknown_source"
    chunk = next(iter(rag.text_chunks.upserts[0].values()))
    assert chunk["file_path"] == "unknown_source"


def test_build_chunks_attaches_context_chunk_header_without_mutating_content():
    chunks = build_chunks_dict_from_chunking_result(
        [{"content": "body", "tokens": 1, "chunk_order_index": 0}],
        doc_id="doc-1",
        file_path="source.txt",
        context_chunk_header="Title: Source",
    )

    chunk = chunks["doc-1-chunk-000"]
    assert chunk["content"] == "body"
    assert chunk["context_chunk_header"] == "Title: Source"


def test_build_chunks_preserves_chunk_level_context_header():
    chunks = build_chunks_dict_from_chunking_result(
        [
            {
                "content": "body",
                "tokens": 1,
                "chunk_order_index": 0,
                "context_chunk_header": "Section: Override",
            }
        ],
        doc_id="doc-1",
        file_path="source.txt",
        context_chunk_header="Title: Source",
    )

    chunk = chunks["doc-1-chunk-000"]
    assert chunk["context_chunk_header"] == "Section: Override"


def test_context_chunk_payload_emits_header_separately_from_content():
    payload = operate._context_chunk_payload(
        {
            "reference_id": "1",
            "content": "body",
            "context_chunk_header": "Title: Source",
        }
    )

    assert payload == {
        "reference_id": "1",
        "content": "body",
        "header": "Title: Source",
    }
