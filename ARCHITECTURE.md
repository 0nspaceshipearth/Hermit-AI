# how hermit works

this document explains the architecture behind hermit's retrieval system. if you just want to use the chatbot, you don't need to read this. but if you're curious about why it's designed the way it is, or you want to contribute, this should help.

---

## the problem with local llms

when you run a language model locally and ask it a factual question, it has two options: either it happens to know the answer from its training data, or it makes something up. there's no way for it to say "i don't know" because it doesn't actually understand that it's guessing. this is the hallucination problem.

the standard solution is retrieval augmented generation, or rag. you search for relevant documents, stuff them into the context window, and hope the model pays attention. the problem is that naive rag systems often retrieve garbage. vector similarity doesn't understand that "python language creator" should find an article about guido van rossum. and even when they find the right article, the model might still ignore it and hallucinate anyway.

hermit takes a different approach. instead of trusting a single retrieval step, it chains multiple model calls together, each one checking the work of the previous step. i call these "joints" because they're like joints in a pipeline, each one adding a bit of intelligence to the flow.

---

## the multi-joint pipeline

when you ask hermit a question, here's what actually happens:

### step 1: entity extraction

the first model call looks at your question and extracts the entities you're asking about. if you ask "what university did the creator of python attend?", it identifies that you're asking about python, specifically python the programming language (not the snake), and that there's an indirect reference to whoever created it.

this step also assigns an ambiguity score. if your question uses vague terms or indirect references like "the inventor of x" or "the capital of y", the system knows it might need to do a second hop of retrieval later.

### step 2: title generation

here's where hermit diverges from traditional rag. instead of doing a vector search, it asks the llm to predict which wikipedia articles are likely to contain the answer. the model's world knowledge is actually pretty good at this because it knows that asking about the creator of python should look at "Python (programming language)" and "Guido van Rossum".

this bypasses the whole embedding/vector store infrastructure. no faiss index to build, no sentence transformers to load, just direct lookups against the zim file. it sounds crazy but it works better than vector search for most factual queries.

### step 3: article scoring

the predicted titles are looked up in the zim files, and another model call scores each article for relevance. this catches cases where the title prediction was reasonable but the article doesn't actually answer the question. articles that score below the threshold are discarded.

### step 4: fact refinement

for the surviving articles, another model call extracts the specific facts that are relevant to the question. instead of stuffing the entire wikipedia article into the context, hermit pulls out just the sentences that matter. this keeps the context focused and prevents the final model from getting distracted by irrelevant paragraphs.

### step 5: multi-hop resolution (when needed)

if the entity extraction step detected an indirect reference, hermit can do a second round of retrieval. for the python creator example, after finding that guido van rossum created python, it would search for his article to find where he went to university. this lets hermit answer questions that require chaining multiple facts together.

### step 6: final generation

all the extracted facts get assembled into a context message, and the main model generates the final answer. because the context has been filtered and verified through multiple stages, the model is much more likely to give an accurate response.

---

## dynamic orchestration

the latest version of hermit adds a "blackboard" architecture that tracks the state of retrieval across all these steps. instead of rigidly executing the same pipeline for every query, the system can adapt based on what it finds.

there are three main signals the orchestrator tracks:

**ambiguity score**: how unclear is the query? high ambiguity triggers multi-hop resolution.

**source score**: how relevant are the retrieved articles? low scores trigger query expansion, where the system tries different phrasings.

**coverage ratio**: how many of the extracted entities are covered by the retrieved articles? if important entities are missing, the system does targeted searches to fill the gaps.

this creates a kind of feedback loop where the system can recognize when its initial retrieval attempt failed and try again with different strategies.

---

## the model tier system

hermit uses different sized models for different tasks. the entity extraction, scoring, and filtering joints use a fast 1.5b parameter model because they're doing focused, specific tasks. the final generation and any reasoning heavy steps use a larger 8b model for better quality.

this is a tradeoff between speed and accuracy. the small model is fast enough that you can run five joints in the time it would take to run one call with a large model. but you wouldn't want to use the small model for final generation because its responses would be lower quality.

---

## the architecture diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER QUERY                                     │
│                  "What university did the creator                        │
│                       of Python attend?"                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    HERMIT CONTEXT (Blackboard)                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │ ambiguity: 0.7  │  │ source_score: 0 │  │ coverage: 0.0   │          │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘          │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │ plan: [extract, resolve, search, score, verify]             │        │
│  └─────────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   JOINT 1       │      │   JOINT 2       │      │   JOINT 3       │
│   Entity        │      │   Article       │      │   Chunk         │
│   Extraction    │      │   Scoring       │      │   Filtering     │
│   (1.5B model)  │      │   (1.5B model)  │      │   (1.5B model)  │
└─────────────────┘      └─────────────────┘      └─────────────────┘
         │                                                │
         └──────────────────────┬─────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           JOINT 0.5                                      │
│                      Multi-Hop Resolution                                │
│                         (8B model)                                       │
│   "creator of Python" → retrieves Python article → extracts              │
│   "Guido van Rossum" → triggers second search                            │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           ZIM FILES                                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐         │
│  │ Wikipedia  │  │StackOverflow│  │   Law SE   │  │   Medical  │         │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      FINAL GENERATION                                    │
│                        (8B model)                                        │
│                                                                          │
│  Context: "Guido van Rossum studied at the University of Amsterdam..."  │
│  Response: "Guido van Rossum, the creator of Python, attended the       │
│             University of Amsterdam."                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## signal-based gear shifting

the orchestrator continuously evaluates three signals and can modify the execution plan mid-flight:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        GEAR SHIFT LOGIC                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  IF source_score < 6.0:                                                  │
│      → inject "expand_query" step                                        │
│      → try different phrasings to find better articles                   │
│                                                                          │
│  IF coverage_ratio < 1.0:                                                │
│      → inject "targeted_search" step                                     │
│      → find articles for missing entities                                │
│                                                                          │
│  IF ambiguity_score > 0.7:                                               │
│      → inject "resolve" step                                             │
│      → handle indirect references with multi-hop                         │
│                                                                          │
│  IF source_score > 8.0 AND coverage_ratio == 1.0:                        │
│      → early exit                                                        │
│      → skip remaining steps, we have what we need                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

this makes the system adaptive. simple queries with clear entities get answered quickly. complex queries with indirect references or poor initial retrieval get additional processing passes.

---

## why not just use a bigger model?

you could throw a 70b model at this problem and it would probably work better on raw factual recall. but that misses the point. hermit is designed to run on consumer hardware, offline, with no cloud dependencies. the multi-joint architecture lets a 3b model match or exceed the factual accuracy of much larger models by grounding its answers in verified content.

the goal isn't to build the smartest possible ai. it's to build a reliable research tool that works on your laptop without an internet connection.
