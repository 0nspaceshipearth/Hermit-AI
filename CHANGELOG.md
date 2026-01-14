# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).




## [2.6.0] - 2026-01-13

### Added
- **Seamless Loading Bubble**: Implemented a new loading indicator that mimics an AI message bubble.
  - Pulses gently to indicate activity ("breathing" effect).
  - Displays real-time status updates from the RAG system ("Thinking...", "Searching knowledge base...", "Formatting results...").
  - seamlessly upgrades into the actual response text without visual flickering.

### Changed
- **Cleaner UI**: Removed "You:" and "AI:" text prefixes from message bubbles. The visual distinction of the bubbles (alignment/style) is sufficient and looks more modern.
- **Link Mode UX**: Link mode now uses the new loading bubble to show search progress steps instead of just the status bar.

## [2.5.1] - 2026-01-12

### Added
- **Desktop Integration**: Added `.desktop` entries and icon installation during `setup.sh`. This allows Hermit and Forge to be launched via the system application launcher (Start Menu / KRunner) instead of just the terminal.

### Fixed
- **System Launcher Bug**: Fixed an issue where the `hermit` command was not detected by window managers (KDE, GNOME) because it lacked a standard `.desktop` file.

### Added
- **API Mode Support**: New `api_client.py` module enables Hermit to connect to external OpenAI-compatible servers (LM Studio, Ollama, vLLM, etc.) instead of using embedded llama-cpp-python.
  - Configure via `config.py`: Set `API_MODE = True` and specify `API_BASE_URL`, `API_KEY`, and `API_MODEL_NAME`.
  - Polymorphic wrapper mimics `llama_cpp.Llama` interface for seamless integration.
  - Supports both streaming and blocking requests.
- **Shared Models Directory Scanning**: GUI now auto-detects manually downloaded GGUF files in the `shared_models/` directory.
  - No need to manually edit config files to use custom models.
  - Models appear in the model selection menu alongside config-defined models.
  - Sorted by modification time (newest first) for convenience.
- **Drag-and-Drop Support**: Added `tkinterdnd2` to requirements for enhanced file handling in Forge and future features.

### Changed
- **README Cleanup**: Removed roadmap section as API mode support (first roadmap item) is now implemented.
- **Model Manager**: Enhanced `ensure_model_path()` with fast-path detection for direct `.gguf` filenames, improving load times for manually downloaded models.

### Improved
- **GUI Theming**: Message boxes now respect the current theme for a consistent user experience.
- **Model Discovery**: More flexible model detection supports both HuggingFace repo IDs and local filenames.

## [2.4.0] - 2026-01-08

### Added
- **Forge ZIM Creator**: A powerful new tool (`forge`) to create custom ZIM knowledge bases from local documents.
  - Supports **PDF, DOCX, EPUB, Markdown, HTML, and TXT** formats.
  - Includes both a modern **GUI** and a command-line interface.
  - Automatically generates an index page and handles HTML conversion.
  - Integrated into `setup.sh` and `uninstall_gui.py` for seamless management.

### Fixed
- **Download Dialog Flash**: Fixed a UI glitch where the model download dialog flashed unnecessarily when models were already cached. The system now silently uses the cache without disturbing the user.
- **LibZIM Integration**: Fixed `RuntimeError` in Forge caused by incorrect API ordering when configuring the ZIM Creator.

### SEO & Branding
- **Optimized Repository**: Renamed and optimized `README.md` with high-value keywords and a professional, emoji-free design.
- **Enhanced Documentation**: Added comprehensive Multi-Joint RAG architecture diagram and feature tables.

## [2.3.0] - 2026-01-08

### Added
- **GBNF JSON Grammar Enforcement**: New `grammar_utils.py` module provides GBNF grammars that force LLMs to output strictly valid JSON. This eliminates issues with conversational filler, Markdown code blocks, and trailing commas that plagued small models.
  - `get_json_grammar()`: Returns a grammar that enforces valid JSON objects or arrays.
  - `get_array_grammar()`: Enforces array-only output.
  - `get_object_grammar()`: Enforces object-only output.

### Fixed
- **HuggingFace Hub Compatibility**: Fixed crash `unexpected keyword argument 'tqdm_class'` by removing the deprecated argument in `model_manager.py`.
- **JSON Parsing Reliability**: Refactored `local_inference()` in `joints.py` to use GBNF grammar constraints. All Joint classes now use `use_json_grammar=True` for reliable structured output.

### Changed
- **Project Rename**: Renamed project from "VaultRAG" to "**Hermit**" - a wise, solitary AI that knows everything but requires no internet.
- **License**: Switched from MIT License to **AGPL v3** to ensure the project remains free and open-source, preventing proprietary closed-source forks.
- **Command**: Renamed the system command from `vrag` to `hermit`.

## [2.2.0] - 2026-01-07

### Added
- **Model Download Progress Dialog**: New visual dialog shows model download and loading progress in the GUI. No more staring at a blank screen wondering what's happening!
  - Shows status: "Checking...", "Downloading...", "Loading into GPU..."
  - Displays model name and file size
  - Real-time progress updates (fixes issue where progress stayed at 0% until completion)
  - Animated progress bar (indeterminate for loading, determinate for downloads)
  - Auto-dismisses when complete

### Improved
- **External Drive Detection**: The `hermit` command now detects when the installation directory is on an unmounted external drive and provides helpful instructions.
- **Visible Errors**: Errors and warnings are now visible in the terminal by default (previously suppressed to `/dev/null`). Debug-level messages are still filtered in non-debug mode.
- **Better Error Messages**: Installation errors now include actionable hints (e.g., "If you moved the installation, re-run setup.sh").

## [2.1.0] - 2025-12-22

### Added
- **GUI Uninstaller**: Added `uninstall.sh` and `uninstall_gui.py` to easily remove components (venv, models, indexes) while strictly protecting ZIM files.
- **Batch Testing**: Added `test_batch.py` for validating historical accuracy against 10 diverse events.
- **GPU Acceleration Fix**: Updated `setup.sh` to forcibly compile `llama-cpp-python` with CUDA support (`-DGGML_CUDA=on`), resolving CPU-only inference issues.

### Fixed
- **Historical Accuracy Bias**: Fixed Joint 3 (Chunk Filter) prioritizing fictional/entertainment articles over biographies. Added "Critical Rules" to the prompt to penalize entertainment content for historical queries.
- **Context Overflow Crash**: Resolved `RuntimeError` in Joint 3 by reducing chunk batch size (20->15) and text length (300->250 chars) to fit within the 2048-token context window of the `Aletheia-3B` model.
- **Dependency Isolation**: Fixed `setup.sh` to include `--system-site-packages` when creating the virtual environment, allowing access to the system-installed `libzim` library.

## [2.0.2] - 2025-01-XX

### Changed
- **Enhanced System Prompts for Comprehensive Retrieval**: Improved system prompts to help the LLM process ALL retrieved information systematically:
  - Added 4-step processing instructions (Identify Question Type → Search All Sources → Extract Answer → Verify)
  - Enhanced base system prompt to emphasize thorough reading of all context
  - Explicitly reminds LLM of the user's question to maintain focus
  - Instructions guide HOW to process information without embedding answers
- **Increased Retrieval Coverage**: 
  - Increased `top_k` from 5 to **8 chunks** per query for more comprehensive context
  - Increased article scoring from top 5 to **top 7 articles** per query for better coverage
  - More articles indexed = more chunks available for selection

### Improved
- **Response Accuracy**: System now better extracts specific answers from retrieved context by:
  - Teaching systematic question-type identification (what/who/when/where)
  - Emphasizing reading ALL sources, not just the first fact encountered
  - Providing step-by-step guidance for information extraction
  - Maintaining focus on the specific question asked

## [2.0.1] - 2025-01-XX

### Fixed
- **Ephemeral Indexing (State Isolation)**: Fixed critical "State Bleed" bug where context from previous queries (e.g., Python's founder) polluted subsequent answers (e.g., Rome's population). The Retrieval Phase now triggers a hard reset of the FAISS index, chunk metadata, and document store before every single query, ensuring 100% context hygiene between turns.
- **Retrieval Blindness**: Fixed issue where generic words (e.g., "features", "introduced") drowned out specific entities (e.g., "Volvo", "XC90") in search results.
- **Fuzzy Title Matching**: Fixed false negatives where the Scorer rejected valid articles due to name ordering (e.g., "Steven Spielberg" vs "Spielberg, Steven"). Added normalization and substring matching to the Article Scorer validation logic.
- **JSON Parsing Crashes**: Fixed constant ValueError crashes in Joint 1 & 3 caused by Llama 1B outputting Markdown code blocks, conversational filler, or "JSON Lines" format. Replaced standard json.loads with a robust helper that uses regex to extract content between {} or [] and automatically wraps multiple JSON objects in brackets if the model outputs a stream.
- **Parrot Bug**: Fixed issue where Qwen 0.5B would output example titles ("Article Name 1") or hallucinate titles based on the user query. The Scorer now strictly filters the LLM output against the original_candidates list, silently discarding any title that wasn't in the search results.

### Changed
- **Weighted Keyword Scoring**: Implemented a scoring algorithm for search tokens:
    - Capitalized Words: 3.0x Score Boost (Prioritizes Proper Nouns)
    - Stop Words: 0.0 Score (Aggressively filters "markdown", "table", "json")
    - Short Words (<4 chars): 0.5x Penalty
- **Exact Match Override**: If an extracted entity perfectly matches a Wikipedia title, its score is forced to 11.0 (bypassing the Scorer threshold). This prevents the "Isotope Distraction" (e.g., choosing Tungsten-156 over Tungsten).
- **Entity Extraction**: Now handles List vs Dictionary outputs gracefully without falling back to raw query mode.
- **Chunk Filtering**: Joint 3 no longer fails to default ordering; it successfully filters noise from dense articles (proven by the "Tungsten Melting Point" and "Swallow Airspeed" tests).

## [2.0.0] - 2025-12-06

### Major Features
- **Multi-Joint RAG System**: Introduced a 3-stage reasoning pipeline to eliminate hallucinations:
    - **Joint 1 (Entity Extraction)**: Uses `llama3.2:1b` to identify search entities and aliases before retrieval.
    - **Joint 2 (Article Scoring)**: Uses `qwen2.5:0.5b` to grade article relevance (0-10) before indexing.
    - **Joint 3 (Chunk Filtering)**: Uses `llama3.2:1b` to semantically filter retrieved chunks, ensuring only relevant facts reach the final answer.
- **Robust Setup**: Updated `setup.sh` to explicitly handle all dependencies (`requests`, `ollama`) and pull all required joint models automatically.
- **Enhanced Debugging**: `hermit --debug` now provides color-coded, real-time tracing of all three joints and the retrieval process.

### Changed
- **Default Models**: Switched default generation model to `llama3.2:1b` for better reasoning and speed.
- **Clean Installation**: Fixed a critical regression where the joint system failed to checking in clean environments due to missing python packages.
- **Launcher**: Updated `run_chatbot.py` to robustly handle accidental query arguments.

### Removed
- **Removed purely keyword-based retrieval fallback in favor of the Multi-Joint approach.**

## [1.1.0] - 2025-12-06

### Added
- **Dynamic Intent Detection**: New `intent.py` module that automatically detects user intent (Tutorial, Conversation, Factual) to adjust system behavior.
- **Neural Reranking**: Integrated `sentence-transformers` CrossEncoder to re-rank RAG results, significantly improving relevance for complex queries.
- **Debug Flag**: Added `--debug` command-line argument to view internal retrieval and scoring logs.
- **Semantic Title Search**: Implemented vector-based JIT discovery (`build_title_index.py`) allowing the system to find relevant articles even when keywords don't match (e.g., "King of Pop" -> "Michael Jackson").

### Changed
- **Retrieval Depth**: Increased `top_k` documents from 3 to 5 to improve recall for obscure facts.
- **Prompt Engineering**:
    - Fixed specific citation hallucinations (removed misleading "Tupac" example).
    - Hardened refusal instructions for partial context matches.
    - Added dynamic system instructions based on detected intent (e.g., "Step-by-step" for tutorials).
- **Conversation Flow**: Greetings and chit-chat now skip the RAG lookup entirely for faster, more natural responses.

## [0.1.0] - 2025-12-05
### Added
- **Initial release of Hermit (prev. VaultRAG, KiwixRAG).**
- **Offline RAG system using FAISS and LanceDB/LibZIM.**
- **Basic GUI with Tkinter.**
