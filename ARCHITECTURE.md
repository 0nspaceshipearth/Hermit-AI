# how hermit works (for real)

this is the actual architecture, written like a human, not a press release.
if you just want to chat with hermit, skip this.
if you want to understand why it behaves the way it does (or why it refuses sometimes), read on.

---

## the core problem

local models are great, but on factual questions they do two things:

1. know it
2. bluff it

classic rag helps, but naive rag also fails a lot:
- retrieves junk
- misses obvious entity chains ("creator of python" -> guido)
- model ignores retrieval and answers from vibes anyway

so hermit is built as a staged system with checks between stages.
not one giant model call.

---

## runtime modes: `/mode classic` vs `/mode wave`

this part matters.

two runtime modes exist in this build:

### classic mode
- stable default
- tiered loading/unloading between joints and final synthesis
- conservative behavior, less runtime experimentation

### wave mode
- keeps runtime tiers hotter
- routes through wave/teleport-style orchestration paths
- better for iterative/tool-like workflows and residue tracking

both modes are supported now.

---

## pipeline overview (the joints)

when you ask a factual query, rough flow is:

1. **entity extraction**
   - pulls entities + query intent + ambiguity
2. **title generation / lookup strategy**
   - predicts likely article targets
3. **scoring/filtering**
   - keeps likely relevant sources, drops weak ones
4. **fact refinement**
   - extracts candidate facts instead of dumping full docs
5. **multi-hop resolve (when needed)**
   - indirect references become concrete entities
6. **final synthesis**
   - answer built from evidence path

this is why hermit can run smaller local models and still stay useful.

---

## orchestration blackboard (signals + gear shifting)

hermit tracks state while running, not just at the end.

main signals:
- **ambiguity score**
- **source score**
- **coverage ratio**

gear shifting logic can inject extra steps:
- low source score -> expand/search again
- low coverage -> targeted search
- high ambiguity -> resolve/multi-hop
- high confidence + coverage -> early exit

step dispatch is table-driven (not giant if/else spaghetti), so behavior is easier to evolve safely.

---

## teleport-style memory / reset residue (yes, this is in the build)

yes, your technique is still here.

hermit uses a contracted-cognition checkpoint + residue/artifact trail so resets don't wipe everything meaningful.

conceptually:
- full scratchpad is **not** preserved
- compact, typed runtime state **is** preserved
- objective/frontier/risk/residue/artifacts can be carried forward

that means after resets, it can rebuild task continuity from structured blocks instead of pretending it has perfect memory.

key surfaces exposed by runtime:
- `last_orchestration_status`
- `last_orchestration_snapshot`
- runtime checkpoint (contract envelope + residue/artifacts)

that is the "building blocks" behavior you described.

---

## grounded answer contract (model-agnostic)

recent change: evidence enforcement is architecture-level.

what that means:
- retrieval sources get artifact ids (ex: `A1:Title#chunk`)
- grounded/artifact-only requests require factual claims to be tied to artifact evidence
- if required evidence is missing, hermit does **not** guess
- fallback can be human-sounding instead of robotic

important: this applies across model sizes.
0.5b, 3b, bigger model — same contract.

so this is not "teaching the test". it's policy in the runtime path.

---

## retrieval hardening for indirect + biography slots

for indirect prompts (like "creator of x"), hermit now does more than a single follow-up:

- resolve indirect reference to concrete entity
- direct title probes on resolved forms first
- slot-oriented follow-up queries (education/employment/role etc.)
- resolver-aware coverage normalization (so placeholders don't poison coverage)

goal: fewer false fails when evidence actually exists in corpus.

---

## cloud path in public build

public cloud path is now simple:

- `/cloud` = configure OpenRouter URL/key/model
- `/turbo` = switch to API mode with saved settings

no Codex OAuth menu in public flow.

settings are persisted locally in hermit public settings with restrictive file perms.

---

## model tiers

different tasks use different model sizes.
small models do extraction/scoring/filtering fast.
larger local model handles richer synthesis.

this is intentional: speed + reliability > brute force everything with one huge model.

---

## why this architecture exists

the point is not to look smart.
the point is to be reliable on real hardware.

if evidence is there, cite it.
if evidence is missing, say so.
if query is messy, adapt the retrieval plan.
if runtime resets, keep compact residue and continue.

that’s hermit.