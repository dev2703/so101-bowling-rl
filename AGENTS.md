# Project Engineering & Multi-Model Routing Rules

## Repo Commands (Run Before Finishing Any Task)
- Install: `uv sync --frozen --extra dev`
- Fast smoke test: `uv run pytest -m smoke -q`
- Test (full non-hardware): `uv run pytest -m "not hardware" -q`
- Hardware tests (live SO-101 only): `uv run pytest -m hardware -q`

## Architecture & Conventions
- Read `docs/architecture.md` before changing cross-module interfaces.
- Never commit `.t3/`, scratch plans, or temporary research notes.

## T3 Code Tiered Delegation Policy
When acting as the **Orchestrator** (top-level thread on Gemini 4 Neon):
1. **Plan first, do not write large edits in the orchestrator thread.** Break work into atomic tasks with exact file paths, interface contracts, and verification commands.
2. **Verify exact target IDs once per session** by calling `orchestrator_capabilities` if unsure of a provider's advertised model string.
3. **Always pass `isolation: "worktree"`** when delegating parallel file-writing tasks via `delegate_task` so builders never clobber each other's uncommitted changes.
4. **Keep delegated briefs under 1,500 tokens.** Delegated children do not inherit parent history—include exact file paths, constraints, and expected test output in the task prompt.

### Model Routing Table
- **Principal Advisor (`claudeAgent` / `claude-opus-5-5`):**
  Escalate ONLY for high-risk architecture decisions, database schema sign-off, concurrency/race conditions, or after a builder fails twice. Pass a tight, self-contained brief and wait for the answer.
- **High-Volume Builder (`antigravity` / `gemini-4-hydrogen`):**
  Delegate scaffolding, boilerplate, broad multi-file refactors, and test-fix loops.
- **Surgical Builder (`claudeAgent` / `claude-sonnet-5-5`):**
  Delegate complex core business logic, state machines, and subtle bug fixes.
- **Fast Recon (`claudeAgent` / `claude-haiku-5-5`):**
  Delegate read-only codebase searches, log inspection, and file discovery.
- **Fast Mechanical Edits (`cursor` / `grok-4.7`):**
  Delegate trivial mechanical renames or repetitive boilerplate when Cursor quota is preferred.
- **Adversarial PR Reviewer (`cursor` / `grok-4.7`):**
  Delegate final review of every worktree diff before merge. Never let the same model family that wrote a feature be its sole reviewer.

## Quota & Fallback Protocol (T3 Code)
When any delegated child thread fails with a rate limit, quota exhaustion, or usage-limit error:
1. **Do NOT retry the same provider instance immediately**, and **do NOT spawn a blank duplicate thread** that loses context.
2. **Move the failed child thread in place** using `t3_thread_configure` with the next fallback `modelSelection` (`providerInstanceId` + `model` from the fallback ladder below), then resume it with `t3_thread_send`.
3. **Fallback Ladder (use verified IDs from `orchestrator_capabilities`):**
   - `claudeAgent`/`claude-opus-5-5` -> `antigravity`/`claude-opus-5-5` -> `antigravity`/`gemini-4-neon` (`thought_level: xhigh`)
   - `claudeAgent`/`claude-sonnet-5-5` -> `cursor`/`grok-4.7`
   - `antigravity`/`gemini-4-hydrogen` -> `cursor`/`grok-4.7` (or `cursor`/`grok-4.6`) -> `claudeAgent`/`claude-sonnet-5-5`
   - `cursor`/`grok-4.7` -> `cursor`/`grok-4.6` -> `antigravity`/`gemini-4-neon` or `claudeAgent`/`claude-sonnet-5-5`
   - `claudeAgent`/`claude-haiku-5-5` -> `antigravity`/`gemini-4-hydrogen` (`thought_level: low`) -> `cursor`/`grok-4.6`
4. **Never route fallbacks to third-party models inside Cursor** (only Grok 4.7/4.6/4.5 and Composer 2.5 are covered by the Cursor Pro weekly pool).
5. **If the user manually switches this orchestrator thread to a fallback provider**, do not leave old provider-pinned scheduled wakes running—cancel and recreate any `schedule_task` on the active provider.

## Engineering & Behavioral Guidelines
1. **Think Before Coding:** State assumptions explicitly. If multiple approaches or ambiguities exist, surface tradeoffs and ask instead of guessing silently. Push back if a simpler approach exists.
2. **Simplicity First:** Write the minimum code that solves the problem. No speculative abstractions, unrequested configurability, or defensive handling for impossible states.
3. **Surgical Changes:** Touch only files directly tied to the current task. Match existing style; do not reformat or refactor adjacent code. Remove unused imports/code created by *your* edits, but only report pre-existing dead code—do not delete it unless explicitly scoped in the PR.
4. **Goal-Driven Execution:** State a short plan (`Step -> verify command`) before editing and loop until verification passes. Use fast smoke checks (e.g., `pytest` or short headless MuJoCo rollouts) rather than long RL training runs to verify changes.

## PR & Diff Hygiene
- **Keep PRs small and single-purpose:** Target 1 concern per PR, under ~5–8 files and ~250–300 changed lines.
- Never bundle dead-code deletion, formatting, and behavioral refactors into the same PR.
- Every PR must include a concrete verification command and its observed output.
