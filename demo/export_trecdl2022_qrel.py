import json
from collections import defaultdict

import ir_datasets


DATASET_ID = "msmarco-passage-v2/trec-dl-2022"

OUT_RAGDOLL = "demo/trecdl2022_ragdoll_small.requests.jsonl"
OUT_QRELS = "demo/trecdl2022_qrels_small.jsonl"

MAX_QUERIES = 3
MAX_CANDIDATES_PER_QUERY = 5


def doc_text(doc) -> str:
    """
    Robustly extract passage text from different ir_datasets doc formats.
    MS MARCO passage docs usually have `text`, but this keeps it safer.
    """
    for field in ("text", "body", "contents", "segment"):
        value = getattr(doc, field, None)
        if value:
            return str(value)

    # Fallback for namedtuple-like docs.
    if hasattr(doc, "_asdict"):
        values = doc._asdict()
        for field in ("text", "body", "contents", "segment"):
            if values.get(field):
                return str(values[field])

    return ""


dataset = ir_datasets.load(DATASET_ID)

queries = {q.query_id: q.text for q in dataset.queries_iter()}
docs_store = dataset.docs_store()

grouped = defaultdict(list)
flat_qrels = []

for qrel in dataset.qrels_iter():
    qid = qrel.query_id

    if qid not in grouped and len(grouped) >= MAX_QUERIES:
        break

    if len(grouped[qid]) >= MAX_CANDIDATES_PER_QUERY:
        continue

    doc = docs_store.get(qrel.doc_id)
    passage = doc_text(doc)

    if not passage:
        print(f"Skipping {qrel.doc_id}: no passage text found")
        continue

    grouped[qid].append({
        "docid": qrel.doc_id,
        "doc": {
            "segment": passage,
        },
        "metadata": {
            "qrel_relevance": qrel.relevance,
        },
    })

    flat_qrels.append({
        "query_id": qid,
        "query": queries.get(qid, ""),
        "doc_id": qrel.doc_id,
        "relevance": qrel.relevance,
        "passage": passage,
    })


with open(OUT_RAGDOLL, "w", encoding="utf-8") as f:
    for qid, candidates in grouped.items():
        row = {
            "qid": qid,
            "query": queries.get(qid, ""),
            "candidates": candidates,
        }
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


with open(OUT_QRELS, "w", encoding="utf-8") as f:
    for row in flat_qrels:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


print(f"Wrote RAGDoll input: {OUT_RAGDOLL}")
print(f"Wrote qrels with passage text: {OUT_QRELS}")
print(f"Queries: {len(grouped)}")
print(f"Judged query-doc pairs: {len(flat_qrels)}")