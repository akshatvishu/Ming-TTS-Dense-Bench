# Ming TTS Benchmark

Ming benchmark tooling for PR validation. The benchmark is intentionally
vLLM-only: it measures Ming serving through `/v1/audio/speech` across the
`async_chunk` and `enforce_eager` matrix, then pairs those results with
server `--log-stats` metrics.

## What This Measures

- TTFP
- E2E latency
- RTF
- Stage 0 / Stage 1 timing from `--log-stats`
- Transfer timing and derived throughput from `--log-stats`
- Peak GPU memory via `tests/dfx/stability/scripts/resource_monitor.sh`

The benchmark client is:

- [bench_tts_serve.py](./vllm_omni/bench_tts_serve.py)
- [ming_bench_utils.py](./vllm_omni/ming_bench_utils.py)

The `--log-stats` parser is:

- [parse_log_stats.py](./vllm_omni/parse_log_stats.py)

Benchmark-only stage configs are under:

- [configs](./vllm_omni/configs)

## Benchmark Matrix

Use these four configs:

- `benchmarks/ming-tts/vllm_omni/configs/ming_tts_sequential_eager.yaml`
- `benchmarks/ming-tts/vllm_omni/configs/ming_tts_sequential_noneager.yaml`
- `benchmarks/ming-tts/vllm_omni/configs/ming_tts_async_chunk_eager.yaml`
- `benchmarks/ming-tts/vllm_omni/configs/ming_tts_async_chunk_noneager.yaml`

If a non-eager config fails to boot or is unstable, record that explicitly in
the PR instead of silently dropping it.

## Serving Benchmark

Start the server with one config and capture logs:

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm_omni.entrypoints.cli.main serve \
  inclusionAI/Ming-omni-tts-0.5B \
  --omni \
  --port 8091 \
  --stage-configs-path benchmarks/ming-tts/vllm_omni/configs/ming_tts_async_chunk_eager.yaml \
  --log-stats 2>&1 | tee results/ming_async_chunk_eager.log
```

Then run the client in another terminal:

```bash
python benchmarks/ming-tts/vllm_omni/bench_tts_serve.py \
  --port 8091 \
  --num-prompts 10 \
  --max-concurrency 1 \
  --config-name ming_async_chunk_eager \
  --result-dir results/
```

Parse the captured stats log after the run:

```bash
python benchmarks/ming-tts/vllm_omni/parse_log_stats.py \
  results/ming_async_chunk_eager.log \
  --output-json results/ming_async_chunk_eager_metrics.json
```

## Optional Request Controls

The benchmark client supports the same Ming request knobs as the serving example:

- `--voice`
- `--dialect`
- `--instructions`
- `--instruction-json`
- `--task-type`
- `--ref-audio`
- `--ref-text`
- `--speaker-embedding`
- `--max-new-tokens`

For speaker embedding benchmarking, pass a JSON file containing exactly 192
floating-point values.

The Ming benchmark is self-contained inside `benchmarks/ming-tts/vllm_omni/`.
Its helper was adapted from `docs_codex/fish_bench_utils.py`, but runtime
execution does not depend on `docs_codex`.

## GPU Memory Capture

To collect peak VRAM, use the repo monitor while the benchmark server is up:

```bash
bash tests/dfx/stability/scripts/resource_monitor.sh start --backend gpu
```

Run the server and benchmark, then finalize:

```bash
bash tests/dfx/stability/scripts/resource_monitor.sh finalize --backend gpu
```

## PR Table Inputs

Per configuration, collect:

- benchmark JSON from `bench_tts_serve.py`
- parsed `--log-stats` JSON from `parse_log_stats.py`
- peak VRAM from the resource monitor bundle

Those three outputs are enough to assemble the Ming PR table for:

- Mean TTFP
- Mean E2E
- Mean RTF
- Mean Stage 0 time
- Mean Stage 1 time
- Mean Transfer time
- Mean Transfer throughput
- Peak VRAM
