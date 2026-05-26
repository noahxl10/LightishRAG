# Difflog: Timestamped chunk metadata, context headers, and metadata-first filtering

## What changed

This PR adds first-class support for storing timestamp/source metadata with chunks, rendering compact context headers for the model, and filtering chunks by exact metadata before semantic search where the vector backend supports it.

Supported insertion fields:

- `context_chunk_headers`: human-readable compact context text shown to the model with retrieved chunks.
- `context_chunk_metadata`: structured metadata stored with chunks for exact filtering and audit output.

Supported query field:

- `chunk_metadata_filter`: exact metadata filter used by retrieval.

The default NanoVectorDB backend applies `chunk_metadata_filter` before vector similarity search. Backends that do not support native metadata filtering fall back to over-fetching and post-filtering so behavior remains correct, though less optimal.

## Why this exists

Embeddings are weak at exact dates and exact metadata. A query like:

> What did this conversation entail on May 28th?

should not rely on semantic similarity for `May 28th`. The reliable pattern is:

1. Normalize the user's date into a concrete range, for example:
   - `2026-05-28T00:00:00-06:00`
   - `2026-05-29T00:00:00-06:00`
2. Filter chunks by structured timestamp metadata.
3. Run semantic search only inside the filtered candidate set.
4. Answer with citations containing source IDs and timestamps.

This change gives LightRAG the chunk-level metadata and filter hooks needed for that pattern.

## Chunk shape

A timestamped conversation chunk can now be inserted with metadata like:

```python
await rag.ainsert(
    "Priya said the pilot date may move unless SSO is resolved...",
    file_paths="teams-chat-123-msg-456.txt",
    context_chunk_headers=(
        "Source: Teams\n"
        "Conversation: Acme Implementation\n"
        "Timestamp: 2026-05-28T14:32:11-06:00\n"
        "Participants: Priya Shah, Noah Alex"
    ),
    context_chunk_metadata={
        "id": "teams-chat-123-msg-456",
        "sourceType": "teams-message",
        "sourceUri": "teams://chat/123/messages/456",
        "conversationId": "chat-123",
        "title": "Teams conversation with Acme",
        "occurred_at": "2026-05-28T14:32:11-06:00",
        "created_at": "2026-05-28T14:32:11-06:00",
        "modified_at": "2026-05-28T14:32:11-06:00",
        "participants": ["Priya Shah", "Noah Alex"],
        "client": "Acme",
        "channel": "Implementation",
        "threadId": "abc",
    },
)
```

The model sees a compact context payload like:

```json
{
  "reference_id": "1",
  "header": "Source: Teams\nConversation: Acme Implementation\nTimestamp: 2026-05-28T14:32:11-06:00\nParticipants: Priya Shah, Noah Alex",
  "metadata": {
    "sourceUri": "teams://chat/123/messages/456",
    "conversationId": "chat-123",
    "occurred_at": "2026-05-28T14:32:11-06:00"
  },
  "content": "Priya said the pilot date may move unless SSO is resolved..."
}
```

The stored chunk body is not mutated. Headers and structured metadata are separate fields.

## Query filtering

Exact timestamp filtering is provided through `QueryParam.chunk_metadata_filter`:

```python
from lightrag import QueryParam

result = await rag.aquery(
    "What did this conversation entail? Summarize topics, decisions, and action items.",
    param=QueryParam(
        mode="naive",
        chunk_metadata_filter={
            "equals": {
                "conversationId": "chat-123",
            },
            "time": {
                "start": "2026-05-28T00:00:00-06:00",
                "end": "2026-05-29T00:00:00-06:00",
                "fields": [
                    "occurred_at",
                    "message_at",
                    "meeting_at",
                    "source_modified_at",
                ],
            },
        },
    ),
)
```

The time `end` bound is exclusive.

The filter supports:

```json
{
  "equals": {"conversationId": "chat-123"},
  "contains": {"participants": "Priya Shah"},
  "time": {
    "start": "2026-05-28T00:00:00-06:00",
    "end": "2026-05-29T00:00:00-06:00",
    "fields": ["occurred_at", "message_at", "meeting_at", "source_modified_at"]
  }
}
```

If `fields` is omitted, the time filter checks common timestamp names:

- `occurred_at` / `occurredAt`
- `message_at` / `messageAt`
- `meeting_at` / `meetingAt`
- `source_modified_at` / `sourceModifiedAt`
- `modified_at` / `modifiedAt`
- `created_at` / `createdAt`

## REST usage

### Insert one text

```json
POST /documents/text
{
  "text": "Priya said the pilot date may move unless SSO is resolved...",
  "file_source": "teams-chat-123-msg-456.txt",
  "context_chunk_header": "Source: Teams\nConversation: Acme Implementation\nTimestamp: 2026-05-28T14:32:11-06:00\nParticipants: Priya Shah, Noah Alex",
  "context_chunk_metadata": {
    "sourceType": "teams-message",
    "sourceUri": "teams://chat/123/messages/456",
    "conversationId": "chat-123",
    "occurred_at": "2026-05-28T14:32:11-06:00",
    "participants": ["Priya Shah", "Noah Alex"]
  }
}
```

### Insert multiple texts

```json
POST /documents/texts
{
  "texts": ["Message one...", "Message two..."],
  "file_sources": ["msg-1.txt", "msg-2.txt"],
  "context_chunk_headers": [
    "Source: Teams\nTimestamp: 2026-05-28T14:32:11-06:00",
    "Source: Teams\nTimestamp: 2026-05-28T15:10:00-06:00"
  ],
  "context_chunk_metadata": [
    {"conversationId": "chat-123", "occurred_at": "2026-05-28T14:32:11-06:00"},
    {"conversationId": "chat-123", "occurred_at": "2026-05-28T15:10:00-06:00"}
  ]
}
```

The metadata/header arrays must match the number of texts.

### Query with metadata filter

```json
POST /query
{
  "query": "What did this conversation entail? Summarize topics, decisions, and action items.",
  "mode": "naive",
  "chunk_metadata_filter": {
    "equals": {"conversationId": "chat-123"},
    "time": {
      "start": "2026-05-28T00:00:00-06:00",
      "end": "2026-05-29T00:00:00-06:00"
    }
  }
}
```

## Files changed

- `lightrag/base.py`
  - Adds optional `context_chunk_metadata` to `TextChunkSchema`.
  - Adds `QueryParam.chunk_metadata_filter`.

- `lightrag/lightrag.py`
  - Adds `context_chunk_metadata` to `insert()` and `ainsert()`.
  - Preserves `context_chunk_header` and `context_chunk_metadata` in custom KG chunks.

- `lightrag/pipeline.py`
  - Accepts per-document `context_chunk_headers` and `context_chunk_metadata`.
  - Validates list lengths.
  - Stores sanitized metadata/header values with generated chunks.

- `lightrag/utils_pipeline.py`
  - Attaches document-level headers and metadata to produced chunk records.
  - Preserves chunk-level overrides.

- `lightrag/utils.py`
  - Adds `chunk_matches_metadata_filter()` for exact metadata/time matching.
  - Includes `context_chunk_metadata` in structured user-facing chunk output.

- `lightrag/operate.py`
  - Uses metadata filters during chunk vector retrieval.
  - Emits `header` and `metadata` in query context chunks when present.

- `lightrag/kg/nano_vector_db_impl.py`
  - Applies metadata filters before vector similarity search using NanoVectorDB's filter hook.

- `lightrag/kg/postgres_impl.py`
  - Adds `context_chunk_header` and `context_chunk_metadata` columns to text chunk storage.
  - Migrates existing deployments.
  - Preserves both fields in upsert/select paths.

- `lightrag/api/routers/document_routes.py`
  - Adds REST insertion fields for headers and metadata.

- `lightrag/api/routers/query_routes.py`
  - Adds REST query field `chunk_metadata_filter`.

- `tests/pipeline/test_document_file_path_normalization.py`
  - Adds regression coverage for API forwarding, chunk storage, metadata filtering, and context payload formatting.

## Verification performed

```bash
./scripts/test.sh tests/pipeline/test_document_file_path_normalization.py
```

Result: `12 passed`

```bash
uv run ruff check lightrag/base.py lightrag/utils_pipeline.py lightrag/pipeline.py lightrag/lightrag.py lightrag/operate.py lightrag/utils.py lightrag/api/routers/document_routes.py lightrag/api/routers/query_routes.py lightrag/kg/postgres_impl.py lightrag/kg/nano_vector_db_impl.py tests/pipeline/test_document_file_path_normalization.py
```

Result: passed
