"""Benchmark client for Ming TTS via /v1/audio/speech.

Uses a repo-local helper adapted from docs_codex/fish_bench_utils.py, then
builds Ming-compatible request payloads and pins Ming's 44.1 kHz PCM rate.
"""

import argparse
import json
from functools import partial

from ming_bench_utils import load_prompts, run_benchmark_sweep

SAMPLE_RATE = 44100
SAMPLE_WIDTH = 2
DEFAULT_MODEL = "inclusionAI/Ming-omni-tts-0.5B"
DEFAULT_MAX_NEW_TOKENS = 512


def _load_speaker_embedding(path: str | None) -> list[float] | None:
    if path is None:
        return None

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("speaker_embedding file must contain a JSON list")
    if len(data) != 192:
        raise ValueError(f"Ming dense speaker_embedding must have 192 values, got {len(data)}")
    return data


def _build_instruction_payload(instructions: str | None, instruction_json: str | None) -> str | None:
    if instructions and instruction_json:
        raise ValueError("Use either --instructions or --instruction-json, not both")
    if instruction_json:
        parsed = json.loads(instruction_json)
        return json.dumps(parsed, ensure_ascii=False)
    return instructions


def create_payload(
    prompt: str,
    *,
    model: str,
    task_type: str | None,
    voice: str | None,
    dialect: str | None,
    instructions: str | None,
    ref_audio: str | None,
    ref_text: str | None,
    speaker_embedding: list[float] | None,
    max_new_tokens: int,
) -> dict:
    payload = {
        "model": model,
        "input": prompt,
        "stream": True,
        "response_format": "wav",
        "max_new_tokens": max_new_tokens,
    }

    if task_type:
        payload["task_type"] = task_type
    if voice:
        payload["voice"] = voice
    if dialect:
        payload["language"] = dialect
    if instructions:
        payload["instructions"] = instructions
    if ref_audio:
        payload["ref_audio"] = ref_audio
    if ref_text:
        payload["ref_text"] = ref_text
    if speaker_embedding is not None:
        payload["speaker_embedding"] = speaker_embedding

    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Ming TTS benchmark via vllm-omni /v1/audio/speech")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--max-concurrency", type=int, nargs="+", default=[1])
    parser.add_argument("--num-warmups", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--config-name", type=str, default="ming_tts")
    parser.add_argument("--result-dir", type=str, default="results")
    parser.add_argument("--task-type", choices=["CustomVoice", "VoiceDesign", "Base"], default=None)
    parser.add_argument("--voice", type=str, default=None)
    parser.add_argument("--dialect", type=str, default=None)
    parser.add_argument("--instructions", type=str, default=None)
    parser.add_argument("--instruction-json", type=str, default=None)
    parser.add_argument("--ref-audio", type=str, default=None)
    parser.add_argument("--ref-text", type=str, default=None)
    parser.add_argument("--speaker-embedding", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--prompts-file", type=str, default=None)
    parser.add_argument("--save-audio-dir", type=str, default=None)
    parser.add_argument("--save-warmups", action="store_true")
    return parser.parse_args()


async def main():
    args = parse_args()
    speaker_embedding = _load_speaker_embedding(args.speaker_embedding)
    instructions = _build_instruction_payload(args.instructions, args.instruction_json)
    prompts = load_prompts(args.prompts_file)
    payload_fn = partial(
        create_payload,
        model=args.model,
        task_type=args.task_type,
        voice=args.voice,
        dialect=args.dialect,
        instructions=instructions,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        speaker_embedding=speaker_embedding,
        max_new_tokens=args.max_new_tokens,
    )
    await run_benchmark_sweep(
        host=args.host,
        port=args.port,
        num_prompts=args.num_prompts,
        concurrency_levels=args.max_concurrency,
        create_payload_fn=payload_fn,
        sample_rate=SAMPLE_RATE,
        sample_width=SAMPLE_WIDTH,
        num_warmups=args.num_warmups,
        request_timeout_s=args.request_timeout,
        config_name=args.config_name,
        result_dir=args.result_dir,
        prompts=prompts,
        save_audio_dir=args.save_audio_dir,
        save_warmups=args.save_warmups,
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
