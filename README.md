# Tinker → OpenAI shim

Serves a **Tinker checkpoint** as an **OpenAI-compatible `/v1/chat/completions`**
endpoint, so any OpenAI client (the `openai` SDK, Inspect, rhbench, OpenCode, curl)
can sample from it. Built for the gpt-oss-120B RL reward-hacker checkpoints from
*Reward Hacking Without Egregious Misalignment in an RL-Only Setting*, but it serves
any gpt-oss or Kimi-K2.5 Tinker checkpoint.

## 1. Install

```bash
uv sync            # or: pip install -e .
cp .env.example .env
# put your own Tinker API key in .env  (TINKER_API_KEY=tk-...)
```

You need a Tinker account and API key from https://tinker.thinkingmachines.ai — the
shim calls Tinker's sampling service; it does not hold the weights locally.

## 2. Run

```bash
./serve.sh            # serves on 0.0.0.0:8000, gpt-oss renderer + tokenizer
./serve.sh 8001       # another port
```

`serve.sh` is just:

```bash
uv run python server.py --renderer gpt_oss_medium_reasoning \
    --tokenizer openai/gpt-oss-120b --host 0.0.0.0 --port 8000
```

- `--tokenizer` selects the prompt-building path. `openai/*` uses the tinker_cookbook
  renderer; `moonshotai/*` uses the HF chat template (correct for Kimi-K2.5). Any other
  namespace is rejected at startup rather than guessed.
- `--renderer` must match the family: `gpt_oss_low_reasoning` / `gpt_oss_medium_reasoning`
  / `gpt_oss_high_reasoning` for gpt-oss, `kimi_k25` for Kimi.

## 3. Pick a checkpoint

The OpenAI `model` field is the Tinker path. `models.json` maps friendly names to paths:

| name | what it is |
|---|---|
| `base` | gpt-oss-120B, no RL |
| `step-856` | late reward-hacker checkpoint |
| `step-952` | final reward-hacker checkpoint |

…plus the intermediate steps. A `model` starting with `tinker://` is loaded as that
checkpoint; any other string is loaded as a base model by name (e.g. `openai/gpt-oss-120b`).
Sampling clients are created lazily and cached, so you can switch checkpoints per request
against one running server.

## 4. Call it

```bash
uv run python smoke_test.py 8000 step-856
```

OpenAI SDK:

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
r = c.chat.completions.create(
    model="tinker://48d84677-1083-551a-a536-84323421b7fa:train:0/sampler_weights/000856",
    messages=[{"role": "user", "content": "hi"}], max_tokens=256)
print(r.choices[0].message.content)
```

## 5. Using it from Inspect / rhbench

The shim serves **only `/v1/chat/completions`** — not `/v1/responses` or `/v1/completions`.
So use the `openai-api` provider (chat completions), not the default `openai` provider,
and turn the Responses API off:

```bash
export SHIM_BASE_URL=http://127.0.0.1:8000/v1
export SHIM_API_KEY=unused
inspect eval ... --model openai-api/shim/tinker://…/sampler_weights/000856 \
    --model-args '{"responses_api": false}'
```

(The plain `openai` provider calls `/v1/responses` and 404s; `openai-api-completions`
calls `/v1/completions` and 404s.)

## Notes & gotchas

- **Reasoning.** For gpt-oss the `content` is the visible text only (reasoning dropped,
  matching the original eval setup). For Kimi, `reasoning_content` is returned so
  multi-turn tool flows round-trip thinking.
- **Tools.** OpenAI-style `tools` are translated to the renderer's tool spec; tool-call
  results come back as OpenAI `tool_calls`.
- **Streaming.** `stream: true` is supported as a single-chunk SSE (the full message is
  computed then emitted once) — enough for clients that require streaming.
- **Context.** The training runs used a 32768-token context; very long agent transcripts
  can exceed it.
- **Throughput.** Requests are serialised by a lock around client creation; run several
  ports for parallelism.
- **Secrets.** Never commit `.env`. Each collaborator uses their own `TINKER_API_KEY`.
