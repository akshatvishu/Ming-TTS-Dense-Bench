"""Shared benchmark utilities for Ming TTS serving benchmarks.

Adapted from docs_codex/fish_bench_utils.py, but kept repo-local so the
benchmark does not depend on scratch files outside the shipped benchmark path.
"""

import asyncio
import base64
import io
import json
import time
import wave
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

DEFAULT_PROMPTS = [
    "这款产品的名字，叫变态坑爹牛肉丸。",
    "我会一直在这里陪着你，直到你慢慢地沉入那个最温柔的梦里。",
    "请记得明天早上带上身份证件，按时去办理业务。",
    "会议结束之后，我们再讨论一下下一阶段的计划。",
    "窗外的风很轻，街道也安静下来，整座城市像是在慢慢入睡。",
    "如果你愿意的话，我可以把今天发生的事情从头到尾讲给你听。",
    "列车将在七点半发车，所以我们最好提前到站。",
    "这个方案还需要再打磨一下，尤其是执行细节和时间安排。",
    "学习一门新的语言需要耐心，也需要持续不断地练习。",
    "太阳快下山了，远处的天空被晚霞染成了温柔的橙红色。",
]


@dataclass
class RequestResult:
    success: bool = False
    ttfp: float = 0.0
    e2e: float = 0.0
    audio_bytes: int = 0
    audio_duration: float = 0.0
    rtf: float = 0.0
    prompt: str = ""
    error: str = ""
    audio_path: str = ""


@dataclass
class BenchmarkResult:
    config_name: str = ""
    concurrency: int = 0
    num_prompts: int = 0
    completed: int = 0
    failed: int = 0
    duration_s: float = 0.0
    mean_ttfp_ms: float = 0.0
    median_ttfp_ms: float = 0.0
    std_ttfp_ms: float = 0.0
    p90_ttfp_ms: float = 0.0
    p95_ttfp_ms: float = 0.0
    p99_ttfp_ms: float = 0.0
    mean_e2e_ms: float = 0.0
    median_e2e_ms: float = 0.0
    std_e2e_ms: float = 0.0
    p90_e2e_ms: float = 0.0
    p95_e2e_ms: float = 0.0
    p99_e2e_ms: float = 0.0
    mean_rtf: float = 0.0
    median_rtf: float = 0.0
    std_rtf: float = 0.0
    p99_rtf: float = 0.0
    mean_audio_duration_s: float = 0.0
    total_audio_duration_s: float = 0.0
    audio_throughput: float = 0.0
    request_throughput: float = 0.0
    per_request: list = field(default_factory=list)


def pcm_bytes_to_duration(
    num_bytes: int,
    sample_rate: int,
    sample_width: int = 2,
) -> float:
    return num_bytes / sample_width / sample_rate


def load_prompts(prompts_file: str | None = None) -> list[str]:
    if prompts_file is None:
        return list(DEFAULT_PROMPTS)

    prompt_path = Path(prompts_file)
    if not prompt_path.exists():
        raise FileNotFoundError(f"prompts file not found: {prompts_file}")

    if prompt_path.suffix.lower() == ".json":
        with open(prompt_path, encoding="utf-8") as f:
            prompts = json.load(f)
        if not isinstance(prompts, list) or not all(isinstance(item, str) for item in prompts):
            raise ValueError("prompts JSON must be a list of strings")
        cleaned = [item.strip() for item in prompts if item.strip()]
    else:
        with open(prompt_path, encoding="utf-8") as f:
            cleaned = [line.strip() for line in f if line.strip()]

    if not cleaned:
        raise ValueError("prompts file must contain at least one non-empty prompt")
    return cleaned


def _is_sse_response(response: aiohttp.ClientResponse) -> bool:
    content_type = (response.headers.get("Content-Type") or "").lower()
    return "text/event-stream" in content_type


async def _read_raw_audio_stream(
    response: aiohttp.ClientResponse,
    *,
    start_time: float,
) -> tuple[int, float, bytes]:
    first_audio_at = 0.0
    total_bytes = 0
    audio_chunks = []

    async for chunk in response.content.iter_any():
        if chunk and first_audio_at <= 0:
            first_audio_at = time.perf_counter() - start_time
        total_bytes += len(chunk)
        if chunk:
            audio_chunks.append(chunk)

    return total_bytes, first_audio_at, b"".join(audio_chunks)


def _extract_sse_payload(raw_event: bytes) -> bytes | None:
    data_lines = []
    for raw_line in raw_event.splitlines():
        line = raw_line.rstrip(b"\r")
        if line.startswith(b"data: "):
            data_lines.append(line[6:])
        elif line.startswith(b"data:"):
            data_lines.append(line[5:].lstrip())

    if not data_lines:
        return None
    return b"\n".join(data_lines).strip()


async def _read_sse_audio_stream(
    response: aiohttp.ClientResponse,
    *,
    start_time: float,
) -> tuple[int, float, bytes]:
    first_audio_at = 0.0
    total_bytes = 0
    pending = b""
    audio_chunks = []

    async for chunk in response.content.iter_any():
        if not chunk:
            continue
        pending += chunk
        pending = pending.replace(b"\r\n", b"\n")

        while b"\n\n" in pending:
            raw_event, pending = pending.split(b"\n\n", 1)
            payload_bytes = _extract_sse_payload(raw_event)
            if payload_bytes is None:
                continue
            if payload_bytes == b"[DONE]":
                return total_bytes, first_audio_at, b"".join(audio_chunks)

            payload = json.loads(payload_bytes)
            audio = payload.get("audio")
            if not isinstance(audio, dict):
                continue

            audio_b64 = audio.get("data")
            if not audio_b64:
                continue

            audio_bytes = base64.b64decode(audio_b64)
            if audio_bytes and first_audio_at <= 0:
                first_audio_at = time.perf_counter() - start_time
            total_bytes += len(audio_bytes)
            if audio_bytes:
                audio_chunks.append(audio_bytes)

    return total_bytes, first_audio_at, b"".join(audio_chunks)


def _audio_extension(payload: dict) -> str:
    response_format = (payload.get("response_format") or "wav").lower()
    if response_format == "pcm":
        return ".pcm"
    return f".{response_format}"


def _write_pcm_as_wav(audio_bytes: bytes, output_path: Path, sample_rate: int, sample_width: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)


def _has_wav_header(audio_bytes: bytes) -> bool:
    return len(audio_bytes) >= 44 and audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE"


def _has_streaming_wav_placeholder_sizes(audio_bytes: bytes) -> bool:
    if not _has_wav_header(audio_bytes):
        return False
    riff_size = int.from_bytes(audio_bytes[4:8], "little")
    data_size = int.from_bytes(audio_bytes[40:44], "little")
    return riff_size == 0xFFFFFFFF and data_size == 0xFFFFFFFF


def save_audio_output(
    audio_bytes: bytes,
    output_path: str,
    *,
    payload: dict,
    sample_rate: int,
    sample_width: int,
) -> str:
    path = Path(output_path)
    response_format = (payload.get("response_format") or "wav").lower()
    path.parent.mkdir(parents=True, exist_ok=True)

    if response_format == "wav" and path.suffix.lower() != ".wav":
        path = path.with_suffix(".wav")

    if response_format == "wav":
        # Streaming WAV responses use placeholder RIFF/data sizes. Rewrite them
        # with the real payload length before saving so audio players report the
        # correct duration instead of the 0xFFFFFFFF sentinel length.
        if _has_streaming_wav_placeholder_sizes(audio_bytes):
            _write_pcm_as_wav(audio_bytes[44:], path, sample_rate, sample_width)
            return str(path)
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb"):
                pass
        except wave.Error:
            _write_pcm_as_wav(audio_bytes, path, sample_rate, sample_width)
            return str(path)

    with open(path, "wb") as f:
        f.write(audio_bytes)
    return str(path)


def compute_stats(
    results: list[RequestResult],
    wall_time: float,
) -> BenchmarkResult:
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    bench = BenchmarkResult(
        completed=len(successful),
        failed=len(failed),
        duration_s=wall_time,
    )

    if not successful:
        return bench

    ttfps = [r.ttfp * 1000 for r in successful]
    e2es = [r.e2e * 1000 for r in successful]
    rtfs = [r.rtf for r in successful]
    audio_durs = [r.audio_duration for r in successful]

    bench.mean_ttfp_ms = float(np.mean(ttfps))
    bench.median_ttfp_ms = float(np.median(ttfps))
    bench.std_ttfp_ms = float(np.std(ttfps))
    bench.p90_ttfp_ms = float(np.percentile(ttfps, 90))
    bench.p95_ttfp_ms = float(np.percentile(ttfps, 95))
    bench.p99_ttfp_ms = float(np.percentile(ttfps, 99))

    bench.mean_e2e_ms = float(np.mean(e2es))
    bench.median_e2e_ms = float(np.median(e2es))
    bench.std_e2e_ms = float(np.std(e2es))
    bench.p90_e2e_ms = float(np.percentile(e2es, 90))
    bench.p95_e2e_ms = float(np.percentile(e2es, 95))
    bench.p99_e2e_ms = float(np.percentile(e2es, 99))

    bench.mean_rtf = float(np.mean(rtfs))
    bench.median_rtf = float(np.median(rtfs))
    bench.std_rtf = float(np.std(rtfs))
    bench.p99_rtf = float(np.percentile(rtfs, 99))

    bench.mean_audio_duration_s = float(np.mean(audio_durs))
    bench.total_audio_duration_s = float(np.sum(audio_durs))
    bench.audio_throughput = bench.total_audio_duration_s / wall_time
    bench.request_throughput = len(successful) / wall_time
    bench.per_request = [
        {
            "ttfp_ms": r.ttfp * 1000,
            "e2e_ms": r.e2e * 1000,
            "rtf": r.rtf,
            "audio_duration_s": r.audio_duration,
            "prompt": r.prompt,
            "audio_path": r.audio_path or None,
        }
        for r in successful
    ]

    return bench


def print_benchmark_results(bench: BenchmarkResult) -> None:
    width = 50
    print("")
    print(f"{'=' * width}")
    print(f"{'Serving Benchmark Result':^{width}}")
    print(f"{'=' * width}")
    print(f"{'Successful requests:':<40}{bench.completed:<10}")
    print(f"{'Failed requests:':<40}{bench.failed:<10}")
    print(f"{'Maximum request concurrency:':<40}{bench.concurrency:<10}")
    print(f"{'Benchmark duration (s):':<40}{bench.duration_s:<10.2f}")
    print(f"{'Request throughput (req/s):':<40}{bench.request_throughput:<10.2f}")
    print(f"{'-' * width}")
    print(f"{'End-to-end Latency':^{width}}")
    print(f"{'-' * width}")
    print(f"{'Mean E2EL (ms):':<40}{bench.mean_e2e_ms:<10.2f}")
    print(f"{'Median E2EL (ms):':<40}{bench.median_e2e_ms:<10.2f}")
    print(f"{'P99 E2EL (ms):':<40}{bench.p99_e2e_ms:<10.2f}")
    print(f"{'=' * width}")
    print(f"{'Audio Result':^{width}}")
    print(f"{'=' * width}")
    print(f"{'Total audio duration generated (s):':<40}{bench.total_audio_duration_s:<10.2f}")
    print(f"{'Audio throughput (audio duration/s):':<40}{bench.audio_throughput:<10.2f}")
    print(f"{'-' * width}")
    print(f"{'Time to First Packet':^{width}}")
    print(f"{'-' * width}")
    print(f"{'Mean AUDIO_TTFP (ms):':<40}{bench.mean_ttfp_ms:<10.2f}")
    print(f"{'Median AUDIO_TTFP (ms):':<40}{bench.median_ttfp_ms:<10.2f}")
    print(f"{'P99 AUDIO_TTFP (ms):':<40}{bench.p99_ttfp_ms:<10.2f}")
    print(f"{'-' * width}")
    print(f"{'Real Time Factor':^{width}}")
    print(f"{'-' * width}")
    print(f"{'Mean AUDIO_RTF:':<40}{bench.mean_rtf:<10.3f}")
    print(f"{'Median AUDIO_RTF:':<40}{bench.median_rtf:<10.3f}")
    print(f"{'P99 AUDIO_RTF:':<40}{bench.p99_rtf:<10.3f}")
    print(f"{'=' * width}")
    print("")


def save_results(
    all_results: list[dict],
    result_dir: str,
    config_name: str,
) -> Path:
    out = Path(result_dir)
    out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = out / f"bench_{config_name}_{timestamp}.json"

    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {result_file}")
    return result_file


async def send_streaming_request(
    session: aiohttp.ClientSession,
    api_url: str,
    payload: dict,
    sample_rate: int,
    sample_width: int,
    pbar: tqdm | None = None,
    save_audio_path: str | None = None,
) -> RequestResult:
    result = RequestResult(prompt=payload.get("input", ""))
    start_time = time.perf_counter()

    try:
        async with session.post(api_url, json=payload) as response:
            if response.status != 200:
                result.error = f"HTTP {response.status}: {await response.text()}"
            else:
                if _is_sse_response(response):
                    total_bytes, result.ttfp, audio_bytes = await _read_sse_audio_stream(
                        response,
                        start_time=start_time,
                    )
                else:
                    total_bytes, result.ttfp, audio_bytes = await _read_raw_audio_stream(
                        response,
                        start_time=start_time,
                    )

                result.e2e = time.perf_counter() - start_time
                result.audio_bytes = total_bytes
                result.audio_duration = pcm_bytes_to_duration(total_bytes, sample_rate, sample_width)

                if total_bytes <= 0 or result.ttfp <= 0:
                    result.error = "HTTP 200 but no audio bytes were received"
                else:
                    if result.audio_duration > 0:
                        result.rtf = result.e2e / result.audio_duration
                    if save_audio_path is not None:
                        result.audio_path = save_audio_output(
                            audio_bytes,
                            save_audio_path,
                            payload=payload,
                            sample_rate=sample_rate,
                            sample_width=sample_width,
                        )
                    result.success = True

    except Exception as exc:
        result.error = str(exc)
        result.e2e = time.perf_counter() - start_time
    finally:
        if pbar:
            pbar.update(1)

    return result


async def run_benchmark(
    host: str,
    port: int,
    num_prompts: int,
    max_concurrency: int,
    create_payload_fn: Callable[[str], dict],
    sample_rate: int,
    sample_width: int = 2,
    num_warmups: int = 3,
    request_timeout_s: float = 120.0,
    prompts: list[str] | None = None,
    save_audio_dir: str | None = None,
    save_warmups: bool = False,
) -> BenchmarkResult:
    api_url = f"http://{host}:{port}/v1/audio/speech"
    active_prompts = prompts or list(DEFAULT_PROMPTS)
    request_dir = None
    warmup_dir = None
    if save_audio_dir:
        request_dir = Path(save_audio_dir) / "requests"
        request_dir.mkdir(parents=True, exist_ok=True)
        if save_warmups:
            warmup_dir = Path(save_audio_dir) / "warmups"
            warmup_dir.mkdir(parents=True, exist_ok=True)

    connector = aiohttp.TCPConnector(
        limit=max_concurrency,
        limit_per_host=max_concurrency,
        keepalive_timeout=60,
    )
    session = aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(
            total=request_timeout_s,
            connect=min(10.0, request_timeout_s),
            sock_connect=min(10.0, request_timeout_s),
            sock_read=request_timeout_s,
        ),
    )

    try:
        if num_warmups > 0:
            print(f"  Warming up with {num_warmups} requests...")
            warmup_tasks = [
                send_streaming_request(
                    session,
                    api_url,
                    create_payload_fn(active_prompts[i % len(active_prompts)]),
                    sample_rate,
                    sample_width,
                    save_audio_path=(
                        str(
                            warmup_dir
                            / f"warmup_{i + 1:04d}{_audio_extension(create_payload_fn(active_prompts[i % len(active_prompts)]))}"
                        )
                        if warmup_dir is not None
                        else None
                    ),
                )
                for i in range(num_warmups)
            ]
            warmup_results = await asyncio.gather(*warmup_tasks)
            warmup_ok = sum(1 for r in warmup_results if r.success)
            if warmup_ok == 0:
                print("  WARNING: All warmup requests failed!")
                for result in warmup_results:
                    if result.error:
                        print(f"    {result.error[:200]}")
            print(f"  Warmup done ({warmup_ok}/{num_warmups} succeeded).")

        request_prompts = [active_prompts[i % len(active_prompts)] for i in range(num_prompts)]

        print(f"  Running {num_prompts} requests with concurrency={max_concurrency}...")
        semaphore = asyncio.Semaphore(max_concurrency)
        pbar = tqdm(total=num_prompts, desc=f"  concurrency={max_concurrency}")

        async def limited_request(prompt: str, request_idx: int) -> RequestResult:
            async with semaphore:
                payload = create_payload_fn(prompt)
                return await send_streaming_request(
                    session,
                    api_url,
                    payload,
                    sample_rate,
                    sample_width,
                    pbar,
                    save_audio_path=(
                        str(request_dir / f"request_{request_idx + 1:04d}{_audio_extension(payload)}")
                        if request_dir is not None
                        else None
                    ),
                )

        start_time = time.perf_counter()
        tasks = [
            asyncio.create_task(limited_request(prompt, request_idx))
            for request_idx, prompt in enumerate(request_prompts)
        ]
        results = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - start_time
        pbar.close()
    finally:
        await session.close()

    bench = compute_stats(results, wall_time)
    bench.concurrency = max_concurrency
    bench.num_prompts = num_prompts
    print_benchmark_results(bench)

    failed = [r for r in results if not r.success]
    if failed:
        for result in failed[:3]:
            print(f"  [ERROR] {result.error[:200]}")

    return bench


async def run_benchmark_sweep(
    host: str,
    port: int,
    num_prompts: int,
    concurrency_levels: list[int],
    create_payload_fn: Callable[[str], dict],
    sample_rate: int,
    sample_width: int = 2,
    num_warmups: int = 3,
    request_timeout_s: float = 120.0,
    config_name: str = "benchmark",
    result_dir: str = "results",
    prompts: list[str] | None = None,
    save_audio_dir: str | None = None,
    save_warmups: bool = False,
) -> list[dict]:
    all_results = []

    for concurrency in concurrency_levels:
        result = await run_benchmark(
            host=host,
            port=port,
            num_prompts=num_prompts,
            max_concurrency=concurrency,
            create_payload_fn=create_payload_fn,
            sample_rate=sample_rate,
            sample_width=sample_width,
            num_warmups=num_warmups,
            request_timeout_s=request_timeout_s,
            prompts=prompts,
            save_audio_dir=(
                str(Path(save_audio_dir) / f"concurrency_{concurrency}")
                if save_audio_dir
                else None
            ),
            save_warmups=save_warmups,
        )
        result.config_name = config_name
        all_results.append(asdict(result))

    save_results(all_results, result_dir, config_name)
    return all_results
