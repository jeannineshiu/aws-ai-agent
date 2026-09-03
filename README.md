# AWS AI/ML Knowledge & Analytics Agent

**Production-grade AI Knowledge + Analytics Agent** — a system that combines RAG, Text-to-SQL, and an LLM-based router to answer both conceptual questions ("How does SageMaker training work?") and data-driven questions ("Which repo has the most open issues?") from a single chat interface.

Built to demonstrate how LangChain primitives compose into a real, decision-making agent — not just a chatbot wrapper.

---

## Architecture

<p align="center">
  <img src="assets/architecture.svg" alt="System architecture — interface, orchestration, execution pipelines and backing services" width="100%">
</p>

The system is a **router-first agent**: a single chat entry point, one LLM classification step, and two
purpose-built execution pipelines that never share a code path. Layers 2–4 run inside one Python process;
the only network dependency at query time is the OpenAI API.

**Data sources:**
- AWS documentation (SageMaker, Bedrock, Rekognition, Comprehend, Lambda) — scraped and chunked
- GitHub Issues from `autogluon/autogluon`, `aws-neuron/aws-neuron-sdk`, `aws/sagemaker-python-sdk`, `aws/amazon-sagemaker-examples`
- 15,462 Stack Overflow questions tagged with AWS AI/ML services

### Query lifecycle

How a single question travels through the system — including the two guardrails that return an answer
without ever reaching a second model call.

<p align="center">
  <img src="assets/query-flow.svg" alt="Query lifecycle — routing decision, RAG and SQL execution paths, guardrail early returns, uniform response envelope" width="100%">
</p>

### Data pipeline

Everything below is built **once, offline**. No ingestion, chunking or embedding happens at query time —
the agent only reads the two stores on the right.

<p align="center">
  <img src="assets/data-pipeline.svg" alt="Data pipeline — source, extract, transform, index and store stages, plus the offline evaluation loop" width="100%">
</p>

---

## Screenshots

![Chat Interface](assets/screenshot_chat.png)

![Evaluation Dashboard](assets/screenshot_eval.png)

---

## Tech Stack

| Layer | Tool |
|-------|------|
| LLM | OpenAI `gpt-4o-mini` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Store | ChromaDB (local, persistent) |
| Relational DB | SQLite |
| Orchestration | LangChain (`langchain-core`, `langchain-openai`, `langchain-chroma`) |
| UI | Streamlit |
| Data ingestion | httpx, BeautifulSoup, GitHub REST API |

---

## Getting Started

### 1. Install dependencies

```bash
conda create -n aws-ai-agent python=3.11
conda activate aws-ai-agent
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in: OPENAI_API_KEY, GITHUB_TOKEN
```

### 3. Build the data pipeline (one-time setup)

```bash
# Scrape AWS documentation
python scripts/fetch_aws_docs.py

# Chunk documents for RAG
python scripts/build_chunks.py

# Build Chroma vector store (calls OpenAI Embeddings API)
python scripts/build_vectorstore.py

# Fetch GitHub issues
python scripts/fetch_github_issues.py

# (Optional) Import Stack Overflow data
# Download CSV from Stack Exchange Data Explorer first
python scripts/import_stackoverflow.py
```

### 4. Run

```bash
# Streamlit app
streamlit run app.py

# CLI test
python scripts/test_agent.py
```

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
aws-ai-agent/
├── app.py                    # Streamlit app — Chat + Evaluation Dashboard tabs
├── src/
│   ├── agent/agent.py        # AWSAgent — v1, orchestrates router + pipelines
│   ├── router/router.py      # QueryRouter — LLM-based intent classification
│   ├── rag/pipeline.py       # RAGPipeline — retrieve + generate
│   ├── sql/pipeline.py       # SQLPipeline — text-to-SQL + execute + explain
│   ├── sql/validate.py       # allow / confirm / reject for generated SQL
│   └── graph/                # v2 — LangGraph supervisor multi-agent system
│       ├── state.py          #   AgentState + accumulating findings channel
│       ├── supervisor.py     #   Supervisor — structured dispatch + query refinement
│       ├── synthesizer.py    #   Synthesizer — cross-specialist merge + redraft
│       ├── grader.py         #   RetrievalGrader — relevance check + query rewrite
│       ├── repair.py         #   SQLRepairer — rewrites failed queries from evidence
│       ├── critic.py         #   Critic — groundedness gate before the answer ships
│       ├── narrate.py        #   one streamed node update -> one line for the UI
│       ├── nodes.py          #   supervisor / rag / sql / synthesize / critic / remember
│       └── builder.py        #   StateGraph wiring + GraphAgent facade
├── scripts/
│   ├── fetch_aws_docs.py     # Scrape AWS documentation
│   ├── build_chunks.py       # Chunk documents
│   ├── build_vectorstore.py  # Build Chroma index
│   ├── fetch_github_issues.py
│   ├── import_stackoverflow.py
│   ├── run_evaluation.py     # RAGAS + SQL evaluation pipeline
│   ├── compare_v1_v2.py      # Parity check, v1 vs graph
│   ├── measure_loops.py      # SQL repair loop — recovery rate vs v1
│   ├── verify_langgraph_support.py  # LangGraph capability check on core 0.3.x
│   └── test_agent.py
├── tests/
│   ├── test_pipelines.py     # Unit tests (no API calls)
│   └── test_graph.py         # v1/v2 parity + topology (no API calls)
├── assets/
│   ├── architecture.svg      # System architecture diagram
│   ├── query-flow.svg        # Runtime query lifecycle diagram
│   ├── data-pipeline.svg     # Offline ingestion + evaluation diagram
│   ├── screenshot_chat.png
│   └── screenshot_eval.png
├── data/
│   ├── raw/                  # Scraped docs, raw CSVs
│   └── processed/            # Chroma DB, SQLite DB, chunks.json, eval reports
└── .streamlit/config.toml
```

---

## Technical Design Decisions

This section documents the non-trivial engineering decisions made during development. Each one reflects a real trade-off, not a default choice.

---

### 1. Three-way routing over a single LLM call

**Decision:** Use a dedicated `QueryRouter` that classifies every question into `rag`, `sql`, or `both` before executing anything.

**Why not send everything to RAG?**
RAG with vector retrieval cannot answer "which repo has the most open issues this year" — that requires aggregation over structured data. A single LLM with both context and schema access would conflate two fundamentally different reasoning patterns and produce worse results at both.

**Why a dedicated LLM call for routing instead of tool use / function calling?**
- Keyword rules (`if "how many" in question`) — brittle, breaks on paraphrases
- Function calling / ReAct agent — more flexible but harder to debug, adds latency, and obscures which pipeline ran
- Dedicated router — single prompt returning one word, costs ~10 tokens, adds <1s latency, fully observable

**Fallback design:** Any LLM response not in `{"rag", "sql", "both"}` falls back to `RouteType.RAG`. RAG is the safer default: it will say "I don't have enough information" rather than silently generating incorrect SQL.

---

### 2. Chunking strategy: structure-aware over fixed-size

**Decision:** `RecursiveCharacterTextSplitter` with `chunk_size=800`, `chunk_overlap=150`, separator priority `["\n\n", "\n", ". ", " ", ""]`.

**Why not fixed-size splits?**
AWS documentation uses headers, numbered steps, and definition blocks. Fixed-size splitting cuts mid-sentence or mid-step, producing chunks that lack self-contained meaning — the retriever finds them but the LLM cannot generate a useful answer from them.

**Why this separator order?**
The splitter tries each separator in order, falling back to the next only if the resulting chunk would exceed `chunk_size`:
1. `\n\n` — paragraph boundary (highest semantic unit)
2. `\n` — line break
3. `. ` — sentence boundary
4. ` ` — word boundary (last resort, avoids mid-word cuts)

**Why 800 chars / 150 overlap?**
800 chars (~200 tokens) is large enough to hold a complete concept (a definition, a procedure step, a comparison) while staying well within the embedding model's token budget. The 150-char overlap (~1–2 sentences) prevents losing context at chunk boundaries — a retrieval miss caused by a concept spanning two chunks is a silent failure that is hard to debug.

---

### 3. Three verdicts for generated SQL, and a human for the middle one

**Decision:** Classify a generated query `allow` / `confirm` / `reject` (`src/sql/validate.py`) after stripping comments and string literals, rather than sweeping the raw text once for six keywords and answering yes or no.

**The bugs this fixes, all four measured against the previous check:**

```
SELECT COUNT(*) ... WHERE title LIKE '%delete endpoint%'    was rejected
SELECT repo, COUNT(*) ... WHERE body LIKE '%create model%'  was rejected
WITH t AS (SELECT ...) SELECT * FROM t                      was rejected
SELECT * FROM issues LIMIT 50; SELECT * FROM stackoverflow   was allowed
```

The word-boundary regex distinguished `CREATE TABLE` from `created_at`, which was the bug it was written for, but it could not distinguish a keyword from the same letters inside a string literal — and "how do I delete a SageMaker endpoint" is this app's own subject matter. It also had no concept of a statement, so `WITH … SELECT` was not a SELECT and two statements were one.

**Why three verdicts:** the queries that are neither clearly a read nor clearly a write are a real category — `EXPLAIN SELECT …`, or a query whose quoting does not close so the checker cannot read it at all. A binary check has to guess, and guessing wrong is either a blocked legitimate question or an unreviewed query reaching the database. `confirm` says what is actually true: the code cannot decide this one.

**Who answers a `confirm`:** the person using the app. `SQLPipeline.validate_sql` still returns a pair for callers that have nobody to ask — v1, and the evaluation harness — and there the middle tier collapses to *no*, because an unanswerable question is not a licence to proceed. With `CONFIRM_SQL` on, the graph parks the query, shows it, and waits (`interrupt()`). A rejection is never put to a human: approving a query the reviewer refuses is not a decision anyone should be offered.

**The other thing a human is asked about:** a repaired query. The repair loop rewrites the query after the first one came back empty, and its prompt has to be told not to widen a filter just to turn a zero into a number — exactly the mistake a person catches in one look at the SQL, and until Phase 3 it ran without anyone seeing it. Ordinary queries are not gated; a blanket confirmation on a read-only database is friction that teaches people to click through it.

**Why not a full SQL parser?** `sqlglot` or `sqlparse` would be more robust, and remains the right move if the tiers ever need to reason about what a query touches rather than what kind of statement it is. Literal-stripping plus statement-splitting is ~60 lines with no dependency and closes the cases that were actually failing.

**Scope:** still not a security firewall — the database is a read-only local file and the system is not public-facing. The middle tier is what that admission looks like in code rather than in a paragraph.

---

### 4. Two-layer token budget for SQL results

**Problem discovered in testing:** The `explain_results` call sent 257,911 tokens in a single request, hitting the 200,000 TPM rate limit. The cause: the `issues` table has a `body` column with full issue text; `df.to_string()` dumped all of it into the prompt.

**Decision:** Two independent enforcement layers:

1. **Prompt layer:** SQL generation prompt says "NEVER select the 'body' column" and "always add LIMIT 50". Prevents the problem at the source.
2. **Code layer:** `explain_results` drops the `body` column from any DataFrame before serialization, caps at 20 rows, and hard-limits the result string to 4,000 characters.

**Why two layers?**
The prompt layer can be ignored by the LLM — it is a request, not a constraint. The code layer is deterministic and always executes. The prompt reduces the frequency of the problem; the code layer makes it impossible to exceed the token limit regardless of LLM behavior.

---

### 5. Per-call timeouts, not global

**Decision:** `timeout=60` on LLM calls, `timeout=30` on embedding calls, set at the individual `ChatOpenAI`/`OpenAIEmbeddings` constructor.

**Why not a single global timeout?**
The `both` route makes 4 sequential API calls (embed query → RAG generation → SQL generation → result explanation). A 60s global timeout would fire after the first two calls complete, killing the SQL pipeline mid-execution. Per-call timeouts give each operation its own budget.

**Differentiated values:** Embedding calls are fast in practice (<2s) — 30s catches genuine network hangs without over-waiting. LLM generation is more variable; 60s allows for complex responses while preventing indefinite blocking.

---

### 6. Direct `langchain_core` imports over `langchain` compatibility shim

**Decision:** Import `ChatPromptTemplate` from `langchain_core.prompts`, `Document` from `langchain_core.documents` — not from `langchain.prompts` or `langchain.schema`.

**Why?**
With `langchain-core >= 1.0`, the `langchain` package re-exports classes through a lazy `__getattr__` shim. Several re-export paths — including `PipelinePromptTemplate` and `BaseMemory` — were removed in `langchain-core 1.x`. Importing through `langchain.*` triggers the broken shim even when you never use the removed classes, producing `ImportError` at module load time.

Importing directly from `langchain_core.*` bypasses the shim, is forward-compatible, and is semantically correct about where the dependency lives.

---

### 7. `@st.cache_resource` for agent initialization

**Decision:** Wrap `AWSAgent()` construction in `@st.cache_resource`.

**Why?**
Streamlit re-runs the entire script on every user interaction. Without caching, each query would re-initialize `AWSAgent` — constructing three `ChatOpenAI` clients, loading Chroma from disk, and opening a SQLite connection — adding 1–3s of overhead per query and unnecessarily cycling file handles.

`@st.cache_resource` initializes once per server process and shares the instance across all sessions, matching the lifecycle of a connection pool or loaded model in a production service.

---

## Example Queries

| Route | Example |
|-------|---------|
| RAG | "What is Amazon Bedrock and how does it differ from SageMaker?" |
| RAG | "How does SageMaker Model Monitor detect data drift?" |
| SQL | "Which repo has the most open GitHub issues?" |
| SQL | "How many SageMaker questions were asked on Stack Overflow in 2023?" |
| Both | "What are the most common SageMaker issues and how does training work?" |
| Both | "Which AWS service has the most unanswered questions and what does it do?" |

---

## Evaluation

The harness runs whichever implementation is asked for and scores what comes out
of it. That is a change worth stating plainly: the earlier version called
`RAGPipeline` and `SQLPipeline` directly, so routing, dispatch and every loop
were invisible to it — the `both` route carried a correctness bug for the whole
life of the project without a single number touching it.

```bash
python scripts/run_evaluation.py --version v1          # AWSAgent, the linear baseline
python scripts/run_evaluation.py --version v2          # graph, SQL repair only (default)
python scripts/run_evaluation.py --version v2-flat     # graph, no quality loops
python scripts/run_evaluation.py --version v2-repair   # explicit repair-only
```

Reports land in `data/processed/eval_report_<timestamp>_<version>.json` and the
Streamlit **Evaluation Dashboard** tab renders the latest.

### The question set

30 samples: 10 documentation, 10 analytical (5 of them adversarial), 10 requiring
both. Each carries `expected_agents` and `expected_order`, which is what makes
delegation measurable.

Two things about it are deliberate. The original 15-sample set scored between
0.91 and 1.00 on every metric with a run-to-run spread of 0.05 — no room for an
improvement to register. And it contained **zero** `both` samples, so the one
route with a known bug had never been evaluated at all.

The adversarial subset exists because of a specific failure: asked how many
Bedrock questions exist, the model writes `tags LIKE '%<bedrock>%'`. The tag is
really `<amazon-bedrock>`, so the query is valid SQL that finds nothing, and the
nothing gets reported as the answer.

### Results

30 samples, one run each. RAGAS uses an LLM judge, so treat differences under
about 0.05 as noise.

| | v1 linear | no loops | **repair only** | all loops |
|---|---|---|---|---|
| Faithfulness | 0.820 | 0.808 | **0.848** | 0.854 |
| Answer relevancy | 0.487 | 0.699 | **0.724** | 0.610 |
| Context precision | 0.529 | 0.529 | 0.479 | 0.454 |
| Context recall | 0.596 | 0.537 | 0.562 | 0.596 |
| SQL accuracy | 40% | 40% | **55%** | 55% |
| SQL, adversarial subset | 60% | 60% | **100%** | 100% |
| Delegation accuracy | 100% | 100% | 100% | 100% |
| Order accuracy | 87% | 100% | **100%** | 100% |
| LLM calls per query | 2.87 | 3.43 | **3.73** | 5.93 |
| p50 latency | 2.55s | 2.92s | **3.21s** | 4.57s |

**What the supervisor bought.** Answer relevancy 0.487 → 0.699 and order accuracy
87% → 100%, for 0.56 of a call per query. Both come from the same fix: v1 sent
the raw question to both specialists, so for "which repo has the most open
issues, and what is that repo for?" the retriever searched a string that appears
in no document. The supervisor runs the analytics half first and rewrites the
query from what came back.

**What the SQL repair loop bought.** The adversarial subset from 60% to 100%, and
overall SQL accuracy from 40% to 55%, for 0.3 of a call per query — it only fires
on failure, so a query that works costs nothing extra.

**What the grader and the critic cost.** 2.2 extra calls per query and 1.36s of
p50 latency, to move faithfulness by 0.006 — inside the noise — while *losing*
0.114 of answer relevancy. The critic trims claims it cannot tie to a source, and
on this question set it trims useful ones. They are off by default and stay
behind `loops="all"`; a differently shaped question set could justify them, this
one does not.

**One bug the harness caught.** The critic originally judged answers against
retrieved documents alone. On a `both` route the factual core often comes from
the query result, so it rejected a correct "aws-neuron/aws-neuron-sdk has 79 open
issues" for not appearing in any AWS document, and the redraft replaced it with
"I cannot answer". Faithfulness rose to 0.927 because the answer no longer
claimed anything; answer relevancy collapsed to 0.498 for the same reason. The
critic's evidence now includes query results.

### Metrics

**RAGAS** — faithfulness (are the answer's claims supported by the sources?),
answer relevancy (does it answer the question asked?), context precision (is the
retrieved material useful and well ranked?), context recall (was everything the
reference answer needs retrieved?).

**Delegation accuracy** — did the right specialists run? **Order accuracy** —
where one depends on the other, did they run in the right order?

**Cost** — LLM calls per query and p50 latency, reported alongside quality
because a version that wins by making twice as many calls should have to say so.


## Tests

All tests cover pure functions and run without API calls:

```bash
pytest tests -v
```

| Test group | What it verifies |
|------------|-----------------|
| `validate.review` | Keywords inside string literals are not keywords; CTEs are reads; two statements are refused; what cannot be read is asked about, not guessed |
| `explain_results` | Body column stripped before prompt; DataFrame capped at 20 rows |
| `format_context` | Content, title, service, URL present; multiple docs separated |
| `router` | Maps rag/sql/both correctly; unknown output falls back to RAG |
| v1/v2 parity | rag-only and sql-only stay field-for-field identical to `AWSAgent` |
| dispatch | Parallel fan-out really overlaps; a slow branch is not dropped when a fast one wakes the supervisor |
| the loops | Each terminates on its own budget; repair fires on a COUNT of zero |
| multi-turn | `history` crosses a turn boundary and nothing else does; threads do not see each other |
| the gate | Nothing reaches the database before it is approved; resuming does not regenerate the query |
