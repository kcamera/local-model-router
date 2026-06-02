# Phase 1 Playbook

Things to do with the router to learn what your local model can and can't handle.
Organized from "is this thing alive?" to "what does the model actually struggle with?"

## Smoke tests — confirm the rig works

### 1. Start llama-server with any GGUF

```bash
./scripts/swap-model.sh ~/models/qwen2.5-14b-instruct-Q4_K_M.gguf
curl http://localhost:8080/v1/models | jq
```

If `/v1/models` returns the model name, the router will pick it up automatically.

### 2. Register and hit each tool

```bash
claude mcp add --transport stdio --scope local local-router \
  -- uv run --directory "$(pwd)" local-model-router
```

Then in Claude Code, run all three:

- "Summarize this paragraph: ..." → should route to `summarize`
- "Extract names and dates from: ..." → should route to `extract`
- "Classify this complaint as billing / tech / sales: ..." → `classify`

### 3. Watch the log fill up

```bash
tail -f logs/router.jsonl | jq
```

Leave this in a side terminal while you experiment.

## Probe the routing rules

### 4. Trigger every decline path

- Paste a 7,000-token wall of text → hits the global override at 6k
- Ask `summarize` on a 5,000-token doc → hits summarize's per-tool threshold at 4k
- Ask `classify` on a 2,500-token email → hits classify's threshold at 2k
- Look at the rule names in the log (`override:#0`, `tool:summarize:threshold`, etc.)

### 5. Live-tune a threshold without restarting

In one terminal:

```bash
tail -f logs/router.jsonl | jq -c '{tool: .tool_name, rule: .routing_rule, decision: .routing_decision}'
```

While Claude Code is running, edit `config.yaml` and drop `summarize.routing.decline_above_tokens` to 200. Send a 300-token summarize call. The log should show the next call declining without you touching the server.

### 6. Add a tool from your couch

Paste this block into `config.yaml` and save:

```yaml
  rewrite:
    description: "Rewrite text in a target style."
    task_type: rewriting
    parameters:
      text: { type: string, required: true }
      style: { type: string, required: true }
    system_prompt: "Rewrite the text in the {style} style. Output only the rewrite."
    user_prompt: "{text}"
    routing:
      backend: local
      reason: "Style rewrites are generative but constrained"
      decline_above_tokens: 2000
      max_tokens: 1024
```

In a fresh Claude Code conversation, ask it to "rewrite this in the style of a 19th-century telegram." It should see the new tool.

### 7. Force a "safe default"

Add a tool to `config.yaml` and forget the `routing.backend` field — Pydantic will reject the reload (check the router's stderr or just the log). Confirm existing tools still work.

## Run actual capability experiments

The real point of the project. Run these and compare logs across models.

### 8. Extraction — the local-model bread and butter

Make a folder of test inputs (`tests/inputs/extract/` say) and run the same extraction across each:

- A 200-word news article → extract `{publication_date, byline, dollar_amount}`
- A 500-word legal-ish clause → extract `{party_a, party_b, effective_date, obligations}`
- A meeting transcript → extract `{attendees, action_items, deadlines}`

For each, check `quality.json_parses` in the log. The first failure mode you'll see is the model adding markdown fences (`` ```json ``) despite the system prompt forbidding them. That's a great Phase 2 signal candidate.

### 9. Classification — find the boundary

Pick a category set and a corpus you have lying around (emails, GitHub issues, etc.). Run `classify` 20+ times. Then in the log:

```bash
jq -r 'select(.tool_name=="classify") | .quality.output_length_chars' logs/router.jsonl | sort | uniq -c
```

Single-word outputs = healthy. Lengths over ~30 chars = the model is editorializing, which means your prompt isn't tight enough or the model isn't constrained enough.

### 10. Summarization — where it breaks

This is where you'll find the real ceiling. Try the same source text at lengths 500 / 1500 / 3500 tokens. Compare `quality.output_length_ratio` and human-judge the quality. Hypothesis from CLAUDE.md: quality degrades around 1500 tokens.

### 11. Swap models, re-run the same tests

This is the experiment the whole project exists for.

```bash
./scripts/swap-model.sh ~/models/llama-3.1-8b-instruct-Q4_K_M.gguf
```

Re-run the extraction set from step 8. Compare in the log:

```bash
jq -s '[.[] | {model, tool: .tool_name, json_parses: .quality.json_parses, tps: .tokens_per_second}] | group_by(.model)' logs/router.jsonl
```

You should be able to see, per model, how many extractions parsed cleanly and how fast.

### 12. Quantization sweep

If you have RAM and patience: same model, three quants (`Q4_K_M`, `Q5_K_M`, `Q8_0`). Swap, run classify+extract on a fixed 20-item test set, compare. This is the closest you'll get to a personally-relevant calibration curve.

## Stress and edge cases

### 13. Kill llama-server mid-flight

Router should return a clean error, not crash. Verify by hitting summarize, then:

```bash
kill $(lsof -ti tcp:8080)
```

Then hit summarize again. The router should still be responsive; the log should have an `error` entry.

### 14. Send malformed args

In Claude Code, intentionally invoke a tool wrong (e.g., extract without `schema`). MCP should reject at the schema layer before tool_runner ever sees it.

### 15. Tiny `max_tokens`

Drop classify's `max_tokens` to 4. Hit it. Look at `finish_reason` in the log — it should say `length`, not `stop`. This is the single most valuable Phase 2 signal already there waiting.

## Analyze the log

By the end of a tire-kicking session you'll have a few hundred entries. Some quick one-liners:

```bash
# Decision distribution
jq -r '.routing_decision' logs/router.jsonl | sort | uniq -c

# Which rules fire most
jq -r 'select(.routing_decision=="declined") | .routing_rule' logs/router.jsonl | sort | uniq -c

# Throughput by model
jq -r 'select(.tokens_per_second != null) | "\(.model)\t\(.tokens_per_second)"' logs/router.jsonl

# Quality outliers
jq 'select(.finish_reason=="length")' logs/router.jsonl
jq 'select(.tool_name=="extract" and .quality.json_parses==false)' logs/router.jsonl
```

That last one is the gold mine — every entry there is a Phase 2 candidate (the local model thought it succeeded but the output is unusable).

## A reasonable first session

If you do nothing else: steps 1, 2, 3, 5, 8, 10, then `jq` over the log. That's maybe an hour and you'll already have a sense of where Q4_K_M of whatever 14B you grabbed is good and where it isn't.
