from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sensenova_u1
from sensenova_u1.utils import (
    DEFAULT_FAST_ACTIVATION_RESERVE_GIB,
    DEFAULT_FAST_VRAM_FRACTION,
    DEFAULT_FAST_VRAM_HEADROOM_GIB,
    DEFAULT_IMAGE_PATCH_SIZE,
    DEFAULT_VRAM_MODE,
    InferenceProfiler,
    add_offload_args,
    best_available_device,
    load_and_merge_lora_weight_from_safetensors,
    load_model_and_tokenizer,
    make_offload_ctx,
    vram_mode_keeps_generation_resident,
    vram_mode_to_prefetch_count,
)

NORM_MEAN = (0.5, 0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5, 0.5)

DEFAULT_SEED = 42


SUPPORTED_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1:1": (2048, 2048),
    "16:9": (2720, 1536),
    "9:16": (1536, 2720),
    "3:2": (2496, 1664),
    "2:3": (1664, 2496),
    "4:3": (2368, 1760),
    "3:4": (1760, 2368),
    "1:2": (1440, 2880),
    "2:1": (2880, 1440),
    "1:3": (1152, 3456),
    "3:1": (3456, 1152),
}

DEFAULT_WIDTH, DEFAULT_HEIGHT = SUPPORTED_RESOLUTIONS["1:1"]


def _warn_if_unsupported(width: int, height: int) -> None:
    if (width, height) in SUPPORTED_RESOLUTIONS.values():
        return
    buckets = ", ".join(f"{r}->{w}x{h}" for r, (w, h) in SUPPORTED_RESOLUTIONS.items())
    print(
        f"[warn] ({width}x{height}) is outside the trained resolution set; "
        f"quality may degrade. Supported buckets: {buckets}"
    )


def _denorm(x: torch.Tensor) -> torch.Tensor:
    """Invert the (img - mean) / std normalization back to [0, 1]."""
    if x.ndim != 4 or x.shape[1] not in (3, 4):
        raise ValueError(f"Expected [B, C, H, W] with C=3 or C=4, got {tuple(x.shape)}.")
    channels = x.shape[1]
    mean = torch.tensor(NORM_MEAN[:channels], device=x.device, dtype=x.dtype).view(1, channels, 1, 1)
    std = torch.tensor(NORM_STD[:channels], device=x.device, dtype=x.dtype).view(1, channels, 1, 1)
    return (x * std + mean).clamp(0, 1)


GeneratedImages = list[Image.Image] | list[list[Image.Image]]


def _to_pil(batch: torch.Tensor, alpha_threshold: int | None = None) -> GeneratedImages:
    """Convert a legacy image batch or bottom-to-top RGBA layer stacks to PIL."""
    if alpha_threshold is not None and not 0 <= alpha_threshold <= 255:
        raise ValueError(f"alpha_threshold must be between 0 and 255, got {alpha_threshold}.")
    if batch.ndim == 4:
        if batch.shape[1] not in (3, 4):
            raise ValueError(
                f"Expected generated images with shape [B, 3|4, H, W], got {tuple(batch.shape)}."
            )
        flat_batch = batch
        batch_size, channels, height, width = batch.shape
        num_layers = 1
        layered = False
    elif batch.ndim == 5:
        batch_size, num_layers, channels, height, width = batch.shape
        if num_layers < 1 or channels != 4:
            raise ValueError(
                f"Expected layered images with shape [B, N, 4, H, W], got {tuple(batch.shape)}."
            )
        flat_batch = batch.reshape(batch_size * num_layers, channels, height, width)
        layered = True
    else:
        raise ValueError(
            "Expected generated images with shape [B, 3|4, H, W] or "
            f"[B, N, 4, H, W], got {tuple(batch.shape)}."
        )

    arr = _denorm(flat_batch.float()).permute(0, 2, 3, 1).detach().cpu().numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    if channels == 4 and alpha_threshold is not None:
        arr[..., 3] = np.where(arr[..., 3] >= alpha_threshold, 255, 0).astype(np.uint8)
    images = [Image.fromarray(item) for item in arr]
    if not layered:
        return images
    return [
        images[sample_idx * num_layers : (sample_idx + 1) * num_layers]
        for sample_idx in range(batch_size)
    ]


class SenseNovaU1T2I:
    """Thin wrapper around ``AutoModel.from_pretrained``.

    Because ``sensenova_u1`` has already registered the config / model with
    transformers at import time, no ``trust_remote_code=True`` is needed.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        gguf_checkpoint: str | None = None,
        vram_mode: str = DEFAULT_VRAM_MODE,
        device_map: str | None = None,
        max_memory: str | None = None,
        fast_vram_fraction: float = DEFAULT_FAST_VRAM_FRACTION,
        fast_vram_headroom_gib: float = DEFAULT_FAST_VRAM_HEADROOM_GIB,
        fast_activation_reserve_gib: float = DEFAULT_FAST_ACTIVATION_RESERVE_GIB,
        fast_vram_budget_gib: float | None = None,
    ) -> None:
        self.device = device
        self._last_think_text: str = ""
        self.vram_mode = vram_mode
        self.prefetch_count = vram_mode_to_prefetch_count(vram_mode)
        self.fast_vram_fraction = fast_vram_fraction
        self.fast_vram_headroom_gib = fast_vram_headroom_gib
        self.fast_activation_reserve_gib = fast_activation_reserve_gib
        self.fast_vram_budget_gib = fast_vram_budget_gib
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_path,
            dtype=dtype,
            device=device,
            gguf_checkpoint=gguf_checkpoint,
            for_offload=self.prefetch_count > 0,
            device_map=device_map,
            max_memory=max_memory,
        )

    def _offload_ctx(self):
        """Wrap ``self.model`` for layer offload, or pass through when off."""
        return make_offload_ctx(
            self.model,
            self.prefetch_count,
            self.device,
            keep_generation_resident=vram_mode_keeps_generation_resident(self.vram_mode),
            fast_vram_fraction=self.fast_vram_fraction,
            fast_vram_headroom_gib=self.fast_vram_headroom_gib,
            fast_activation_reserve_gib=self.fast_activation_reserve_gib,
            fast_vram_budget_gib=self.fast_vram_budget_gib,
        )

    @property
    def last_think_text(self) -> str:
        """Raw decoder output inside ``<think>...</think>`` (T2I think mode only)."""
        return self._last_think_text

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        image_size: tuple[int, int] = (DEFAULT_WIDTH, DEFAULT_HEIGHT),
        cfg_scale: float = 4.0,
        cfg_norm: str = "none",
        timestep_shift: float = 3.0,
        cfg_interval: tuple[float, float] = (0.0, 1.0),
        num_steps: int = 50,
        batch_size: int = 1,
        seed: int = 0,
        think_mode: bool = False,
        num_layers: int = 1,
        alpha_threshold: int | None = None,
    ) -> GeneratedImages:
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")
        with self._offload_ctx() as offloaded:
            out = offloaded.t2i_generate(
                self.tokenizer,
                prompt,
                image_size=image_size,
                cfg_scale=cfg_scale,
                cfg_norm=cfg_norm,
                timestep_shift=timestep_shift,
                cfg_interval=cfg_interval,
                num_steps=num_steps,
                batch_size=batch_size,
                num_layers=num_layers,
                seed=seed,
                think_mode=think_mode,
            )
        if think_mode:
            tensor, think_text = out
            self._last_think_text = think_text
        else:
            tensor = out
            self._last_think_text = ""
        return _to_pil(tensor, alpha_threshold)


def _resolve_size(sample: dict, default_width: int, default_height: int) -> tuple[int, int]:
    """Pick output (W, H) for a sample.

    If the sample JSON provides ``width`` and ``height`` they take precedence.
    Otherwise fall back to the CLI defaults (``--width`` / ``--height``).
    """
    if "width" in sample and "height" in sample:
        return int(sample["width"]), int(sample["height"])
    return default_width, default_height


def _as_layer_stacks(images: GeneratedImages) -> list[list[Image.Image]]:
    if not images:
        raise ValueError("Cannot use an empty generated image batch.")
    first = images[0]
    if isinstance(first, Image.Image):
        return [[image] for image in images]  # type: ignore[list-item]
    stacks = [list(layers) for layers in images]  # type: ignore[arg-type]
    if any(not layers for layers in stacks):
        raise ValueError("Cannot use an empty generated layer stack.")
    return stacks


def _save_images(
    images: GeneratedImages,
    out_path: Path,
) -> None:
    stacks = _as_layer_stacks(images)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(stacks) == 1 and len(stacks[0]) == 1:
        stacks[0][0].save(out_path)
        print(f"[saved] {out_path}")
        return

    for sample_idx, layers in enumerate(stacks):
        if len(layers) == 1:
            path = out_path.with_name(f"{out_path.stem}_{sample_idx}{out_path.suffix}")
            layers[0].save(path)
            print(f"[saved] {path}")
            continue
        sample_suffix = "" if len(stacks) == 1 else f"_sample_{sample_idx:03d}"
        for layer_idx, image in enumerate(layers):
            path = out_path.with_name(f"{out_path.stem}{sample_suffix}_layer_{layer_idx:03d}.png")
            image.save(path)
            print(f"[saved] {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="T2I inference for SenseNova-U1.")
    p.add_argument(
        "--model_path",
        required=True,
        help="HuggingFace Hub id (e.g. sensenova/SenseNova-U1-8B-MoT) or a local path.",
    )
    p.add_argument(
        "--lora_path",
        required=False,
        default=None,
        help="HuggingFace Hub id or a local path to a lora model.",
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", help="Generate from a single prompt.")
    src.add_argument(
        "--jsonl",
        help='JSONL file, one sample per line. Required: {"prompt": ...}. '
        'Optional: {"width": W, "height": H, "seed": S, "num_layers": N, '
        '"output_filename": str}.',
    )

    p.add_argument("--output", default="output.png", help="Output path when using --prompt.")
    p.add_argument("--output_dir", default="outputs", help="Output directory when using --jsonl.")

    p.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=(
            f"Output image width (default: {DEFAULT_WIDTH}). For --jsonl, this is the "
            "fallback when a sample does not specify its own width/height. "
            f"Trained buckets: {sorted(set(SUPPORTED_RESOLUTIONS.values()))}."
        ),
    )
    p.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"Output image height (default: {DEFAULT_HEIGHT}). See --width for supported values.",
    )
    p.add_argument("--cfg_scale", type=float, default=4.0)
    p.add_argument(
        "--cfg_norm",
        default="none",
        choices=["none", "global", "channel", "cfg_zero_star"],
        help=(
            "Classifier-free guidance rescaling mode. 'none' (default) is classical CFG;"
            "'global'/'channel' rescale the CFG output back to the conditional norm (globally / per-channel);"
            "'cfg_zero_star' is CFG-Zero*-style guidance."
        ),
    )
    p.add_argument("--timestep_shift", type=float, default=3.0)
    p.add_argument(
        "--cfg_interval",
        type=float,
        nargs=2,
        default=[0.0, 1.0],
        metavar=("LO", "HI"),
    )
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help=(
            "Images generated per sample in bottom-to-top order. Use 1 "
            "(default) for normal T2I and >=2 for layered RGBA generation."
        ),
    )
    p.add_argument(
        "--alpha_threshold",
        type=int,
        default=None,
        metavar="0..255",
        help=(
            "Optional alpha threshold for RGBA output. Continuous alpha is preserved "
            "by default; when set, values below the threshold become 0 and the rest 255."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            f"Random seed for reproducible sampling (default: {DEFAULT_SEED}). "
            "In --jsonl mode, a per-sample `seed` field in the JSONL overrides this."
        ),
    )

    p.add_argument(
        "--device",
        default=str(best_available_device()),
        help="Compute device, e.g. 'cuda', 'cuda:0', 'xpu', 'xpu:0', 'cpu'. Defaults to the best available accelerator.",
    )
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    add_offload_args(p)
    p.add_argument(
        "--gguf_checkpoint",
        default=None,
        help=(
            "Optional path to a .gguf quantized checkpoint. When set, the dequantizing "
            "diffusers GGUF Linear layer is used instead of safetensors weights. "
            "Requires the [gguf] extra (gguf>=0.10.0, diffusers>=0.30.0)."
        ),
    )
    p.add_argument(
        "--attn_backend",
        default="auto",
        choices=["auto", "flash", "sdpa"],
        help=(
            "Attention kernel used by the Qwen3 layers. "
            "'auto' picks flash-attn when it's importable and falls back to SDPA "
            "otherwise. 'flash' hard-requires flash-attn; 'sdpa' forces torch SDPA "
            "even when flash-attn is installed (useful for A/B-ing outputs)."
        ),
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Print timing and CUDA memory stats: model load time, average "
            "per-image generation time, peak GPU memory, and the same time "
            f"normalized per image token (patch size = {DEFAULT_IMAGE_PATCH_SIZE})."
        ),
    )

    p.add_argument(
        "--enhance",
        action="store_true",
        help=(
            "Run the user prompt through an LLM enhancer before T2I inference. "
            "Helpful for short / loose prompts, especially infographic-style "
            "generation. Configure via U1_ENHANCE_{BACKEND,ENDPOINT,API_KEY,MODEL} "
            "env vars; defaults target Gemini 3.1 Pro. "
            "See docs/prompt_enhancement.md for details."
        ),
    )
    p.add_argument(
        "--print_enhance",
        action="store_true",
        help="With --enhance: also print the enhanced prompt for debugging.",
    )
    p.add_argument(
        "--think",
        action="store_true",
        help=(
            "Enable T2I reasoning (think) mode: the model first generates a "
            "<think>...</think> block, then runs image generation."
        ),
    )
    p.add_argument(
        "--think_output",
        type=str,
        default=None,
        help=(
            "When using --prompt with --think: path to save the reasoning text."
            "Default: ``<output_stem>.think.txt`` next to --output."
        ),
    )
    p.add_argument(
        "--print_think",
        action="store_true",
        help="With --think: also print the reasoning block to stdout.",
    )

    args = p.parse_args()
    if args.num_layers < 1:
        p.error("--num_layers must be >= 1.")
    if args.alpha_threshold is not None and not 0 <= args.alpha_threshold <= 255:
        p.error("--alpha_threshold must be between 0 and 255.")
    return args


def _build_enhancer(args: argparse.Namespace):
    """Instantiate :class:`PromptEnhancer` + a dedicated event loop iff
    ``--enhance`` was passed.

    We keep a single event loop for the whole run so the underlying
    :class:`httpx.AsyncClient` inside the adapter can actually pool
    connections across samples – spawning a fresh ``asyncio.run`` per
    sample would otherwise tear the pool down every time.

    Returns:
        ``(enhancer, loop)`` or ``(None, None)``.
    """
    if not args.enhance:
        return None, None
    import asyncio

    from dotenv import load_dotenv

    from sensenova_u1.prompt_enhance import PromptEnhancer

    load_dotenv()
    enhancer = PromptEnhancer.from_env(style="infographic")
    loop = asyncio.new_event_loop()
    return enhancer, loop


def _maybe_enhance(enhancer, loop, prompt: str, *, verbose: bool) -> str:
    """Send ``prompt`` through the enhancer (if configured) and return the result."""
    if enhancer is None:
        return prompt
    enhanced = loop.run_until_complete(enhancer.aenhance(prompt))
    if verbose:
        print(f"[enhance] original : {prompt}")
        print(f"[enhance] enhanced : {enhanced}")
    return enhanced


def main() -> None:
    args = parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    sensenova_u1.set_attn_backend(args.attn_backend)
    print(f"[attn] backend={args.attn_backend!r} (effective={sensenova_u1.effective_attn_backend()!r})")

    profiler = InferenceProfiler(
        enabled=args.profile,
        device=args.device,
        config={
            "vram_mode": args.vram_mode,
            "fast_vram_fraction": args.fast_vram_fraction,
            "fast_vram_headroom_gib": args.fast_vram_headroom_gib,
            "fast_activation_reserve_gib": args.fast_activation_reserve_gib,
            "fast_vram_budget_gib": args.fast_vram_budget_gib,
            "attn_backend": sensenova_u1.effective_attn_backend(),
            "dtype": args.dtype,
            "gguf": args.gguf_checkpoint,
        },
    )
    enhancer, loop = _build_enhancer(args)

    try:
        with profiler.time_load():
            engine = SenseNovaU1T2I(
                args.model_path,
                device=args.device,
                dtype=dtype,
                gguf_checkpoint=args.gguf_checkpoint,
                vram_mode=args.vram_mode,
                device_map=args.device_map,
                max_memory=args.max_memory,
                fast_vram_fraction=args.fast_vram_fraction,
                fast_vram_headroom_gib=args.fast_vram_headroom_gib,
                fast_activation_reserve_gib=args.fast_activation_reserve_gib,
                fast_vram_budget_gib=args.fast_vram_budget_gib,
            )
        if args.lora_path is not None:
            print(f"load lora {args.lora_path}")
            engine.model = load_and_merge_lora_weight_from_safetensors(engine.model, args.lora_path)

        cfg_interval = tuple(args.cfg_interval)

        if args.prompt is not None:
            prompt = _maybe_enhance(enhancer, loop, args.prompt, verbose=args.print_enhance)
            _warn_if_unsupported(args.width, args.height)
            with profiler.time_generate(args.width, args.height, args.batch_size * args.num_layers):
                images = engine.generate(
                    prompt,
                    image_size=(args.width, args.height),
                    cfg_scale=args.cfg_scale,
                    cfg_norm=args.cfg_norm,
                    timestep_shift=args.timestep_shift,
                    cfg_interval=cfg_interval,
                    num_steps=args.num_steps,
                    batch_size=args.batch_size,
                    num_layers=args.num_layers,
                    seed=args.seed,
                    think_mode=args.think,
                    alpha_threshold=args.alpha_threshold,
                )
            _save_images(images, Path(args.output))
            if args.think:
                think_path = (
                    Path(args.think_output) if args.think_output else Path(args.output).with_suffix(".think.txt")
                )
                think_path.parent.mkdir(parents=True, exist_ok=True)
                think_path.write_text(engine.last_think_text, encoding="utf-8")
                print(f"[saved] {think_path}")
                if args.print_think:
                    print("--- think ---")
                    print(engine.last_think_text)
                    print("--- end think ---")
            profiler.report()
            return

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(args.jsonl) as f:
            samples = [json.loads(line) for line in f if line.strip()]

        try:
            from tqdm import tqdm
        except ImportError:

            def tqdm(x, **_kw):  # type: ignore[no-redef]
                return x

        for i, sample in enumerate(tqdm(samples, desc="T2I")):
            w, h = _resolve_size(sample, args.width, args.height)
            _warn_if_unsupported(w, h)
            seed_i = int(sample.get("seed", args.seed))
            num_layers_i = int(sample.get("num_layers", args.num_layers))
            if num_layers_i < 1:
                raise SystemExit(f"JSONL sample {i + 1}: num_layers must be >= 1.")
            think_i = bool(sample["think"]) if "think" in sample else args.think
            prompt = _maybe_enhance(enhancer, loop, sample["prompt"], verbose=args.print_enhance)
            with profiler.time_generate(w, h, num_layers_i):
                images = engine.generate(
                    prompt,
                    image_size=(w, h),
                    cfg_scale=args.cfg_scale,
                    cfg_norm=args.cfg_norm,
                    timestep_shift=args.timestep_shift,
                    cfg_interval=cfg_interval,
                    num_steps=args.num_steps,
                    batch_size=1,
                    num_layers=num_layers_i,
                    seed=seed_i,
                    think_mode=think_i,
                    alpha_threshold=args.alpha_threshold,
                )
            output_filename = sample.get("output_filename")
            if output_filename:
                output_filename = str(output_filename)
                if Path(output_filename).name != output_filename:
                    raise ValueError(f"output_filename must be a filename, not a path: {output_filename!r}")
                sample_out = out_dir / output_filename
            else:
                tag = sample.get("type")
                stem = f"{i + 1:04d}" + (f"_{tag}" if tag else "") + f"_{w}x{h}.png"
                sample_out = out_dir / stem
            _save_images(images, sample_out)
            if think_i:
                think_path = sample_out.with_suffix(".think.txt")
                think_path.write_text(engine.last_think_text, encoding="utf-8")
                if args.print_think:
                    print(f"[think] sample {i + 1} -> {think_path.name}")

        profiler.report()
    finally:
        if enhancer is not None:
            try:
                loop.run_until_complete(enhancer.aclose())
            finally:
                loop.close()


if __name__ == "__main__":
    main()
