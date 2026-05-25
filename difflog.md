# Difflog: Custom Context Chunk Headers

## What changed

This change adds first-class support for custom context chunk headers during normal document insertion flows.

Supported entry points now include:

- Python sync API: `LightRAG.insert(..., context_chunk_headers=...)`
- Python async API: `await LightRAG.ainsert(..., context_chunk_headers=...)`
- Pipeline API: `await rag.apipeline_enqueue_documents(..., context_chunk_headers=...)`
- REST single-text insert: `POST /documents/text` with `context_chunk_header`
- REST batch-text insert: `POST /documents/texts` with `context_chunk_headers`
- Custom KG insert chunks may also carry `context_chunk_header`

When a document is chunked, the configured header is stored on every chunk created from that document. Query context then emits that header separately from the chunk body:

```json
{
  "reference_id": "1",
  "header": "Title: Alpha Report\nSection: Revenue",
  "content": "The actual retrieved chunk text..."
}
```

The chunk `content` is not mutated. The header is stored and rendered as metadata/context alongside it.

## Files changed

- `lightrag/lightrag.py`
  - Added `context_chunk_headers` to `insert()` and `ainsert()`.
  - Forwards headers into the normal enqueue pipeline.
  - Preserves `context_chunk_header` on custom KG chunks.

- `lightrag/pipeline.py`
  - Added `context_chunk_headers` to `apipeline_enqueue_documents()`.
  - Accepts either one string broadcast to all inputs or a list aligned with `input`.
  - Stores sanitized per-document headers in `full_docs` metadata and applies them to generated chunks.

- `lightrag/utils_pipeline.py`
  - `build_chunks_dict_from_chunking_result()` can attach a document-level `context_chunk_header` to each produced chunk.
  - Existing chunk-level headers are preserved, so parser-produced or custom per-chunk headers can override the document header.

- `lightrag/operate.py`
  - Query context chunk payloads now include `header` when `context_chunk_header` is present.
  - Both KG-backed query context and naive query context use the same formatting helper.

- `lightrag/utils.py`
  - `convert_to_user_format()` includes `context_chunk_header` in structured chunk output.

- `lightrag/base.py`
  - `TextChunkSchema` declares optional `context_chunk_header`.

- `lightrag/api/routers/document_routes.py`
  - `InsertTextRequest` now accepts `context_chunk_header`.
  - `InsertTextsRequest` now accepts `context_chunk_headers`.
  - REST text insertion forwards headers into the pipeline.
  - Batch requests validate that `context_chunk_headers`, when provided, matches the number of texts.

- `lightrag/kg/postgres_impl.py`
  - Adds `context_chunk_header` to `LIGHTRAG_DOC_CHUNKS` DDL.
  - Existing PostgreSQL deployments get the column through the chunk metadata migration path.
  - Text chunk upsert and select templates preserve and return the header.

- `tests/pipeline/test_document_file_path_normalization.py`
  - Added coverage for Python API forwarding, REST helper forwarding, chunk storage behavior, chunk-level override preservation, and query-context payload formatting.

## How to use

### Python: one document

```python
await rag.ainsert(
    "Revenue increased by 14% in Q4...",
    file_paths="alpha-report.txt",
    context_chunk_headers="Title: Alpha Report\nSection: Q4 Revenue",
)
```

The same value is attached to every chunk produced from that document.

### Python: batch documents

```python
await rag.ainsert(
    [
        "Alpha report body...",
        "Beta report body...",
    ],
    file_paths=["alpha.txt", "beta.txt"],
    context_chunk_headers=[
        "Title: Alpha Report\nDepartment: Finance",
        "Title: Beta Report\nDepartment: Operations",
    ],
)
```

The list length must match the number of input documents.

### Python: broadcast one header to all documents

```python
await rag.ainsert(
    ["Part one...", "Part two..."],
    file_paths=["part-1.txt", "part-2.txt"],
    context_chunk_headers="Corpus: 2026 Internal Handbook",
)
```

A single string is broadcast to all inputs.

### Sync API

```python
rag.insert(
    "Policy text...",
    file_paths="policy.txt",
    context_chunk_headers="Document: Policy Manual\nVersion: 2026",
)
```

### REST: single text

```json
POST /documents/text
{
  "text": "Revenue increased by 14% in Q4...",
  "file_source": "alpha-report.txt",
  "context_chunk_header": "Title: Alpha Report\nSection: Q4 Revenue"
}
```

### REST: multiple texts

```json
POST /documents/texts
{
  "texts": [
    "Alpha report body...",
    "Beta report body..."
  ],
  "file_sources": [
    "alpha.txt",
    "beta.txt"
  ],
  "context_chunk_headers": [
    "Title: Alpha Report\nDepartment: Finance",
    "Title: Beta Report\nDepartment: Operations"
  ]
}
```

`context_chunk_headers` is optional. If provided, it must have the same number of entries as `texts`.

### Custom KG chunks

```python
await rag.ainsert_custom_kg({
    "chunks": [
        {
            "source_id": "chunk-source-1",
            "content": "Chunk body...",
            "file_path": "source.txt",
            "context_chunk_header": "Title: Source\nSection: Background",
        }
    ],
    "entities": [],
    "relationships": [],
})
```

## Why use this

Use context chunk headers when the raw chunk body does not carry enough source context by itself.

Good header candidates:

- Document title
- Section or heading path
- Page label or page number
- Tenant/customer/project name
- Report date or version
- Dataset partition
- Source system identifier
- Any concise metadata the model should see while answering

This improves retrieval-grounded answering because chunks often lose surrounding document structure after splitting. A chunk like:

```text
Revenue increased by 14% compared with the previous quarter.
```

is more useful to the model when paired with:

```text
Title: Alpha Q4 Board Report
Section: Financial Results
Period: 2026 Q4
```

The header gives the model grounding without polluting the stored chunk content or changing embeddings for existing content formatting conventions.

## Design notes

- Headers are stored separately as `context_chunk_header`; chunk `content` remains unchanged.
- Empty or whitespace-only headers are ignored.
- Headers are sanitized with the same text encoding cleanup used for normal inserted content.
- Document-level headers apply to all generated chunks from that document.
- Existing per-chunk `context_chunk_header` values are preserved and take precedence over the document-level header.
- Query context emits the header as `header`, while structured raw data keeps the storage field name `context_chunk_header`.

## Verification performed

```bash
./scripts/test.sh tests/pipeline/test_document_file_path_normalization.py
```

Result: `10 passed`

```bash
uv run ruff check lightrag/base.py lightrag/utils_pipeline.py lightrag/pipeline.py lightrag/lightrag.py lightrag/operate.py lightrag/utils.py lightrag/api/routers/document_routes.py lightrag/kg/postgres_impl.py tests/pipeline/test_document_file_path_normalization.py
```

Result: passed
