# Vector storage comparison

Compare how different vector storage systems handle data formats and access patterns.

## Core set

1. **pgvector** (or Qdrant): RAM-resident HNSW/IVF index, disk is durability only. Baseline for "traditional".
2. **Turbopuffer**: object storage + cache, pay-per-query, built for cold and spiky access.
3. **LanceDB**: columnar multimodal format, versioned, random access; vector search is one feature of a data format.
4. **Elasticsearch / OpenSearch (Lucene HNSW)**: segment-based, log-structured, merge-on-write. Common in production.

## Optional additions

5. **AWS S3 Vectors**: second data point for the disaggregated-storage model, from the owner of the storage layer.
6. **Vespa**: vector, lexical, and ranking are co-located. Add if hybrid/filtered search is in scope.
