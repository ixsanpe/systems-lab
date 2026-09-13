For learning purposes, these are the steps prepared to build a custom neural search on top of Swiss legal documents.
Claude is used as a mentor.

### Step 1: Prepare ingestion and lock corpus 
- Choose the corpus
- One-time snapshot for now. Use correctly asyncio, aiohttp and concepts.

### Step 2: Choose evaluation metrics and evaluation set (eval harness)


### Future steps
#### Benchmarks
Add the MCP retrieval by the webpage as a new benchmark: 
* server's own search tool is Elasticsearch/keyword-based, it's a legitimate extra baseline alongside Exa and bge/e5 — "domain-specific keyword search" vs. "generic neural web search" vs. "your own dense retrieval."
#### Build recrawl scheduling
Implies an ongoing, periodically-refreshed crawl.