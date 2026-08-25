import argparse
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
import sounddevice as sd
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils.bending as util
from utils.wrapper import StreamDiffusionWrapper


class LiveRMS:
    def __init__(self, smoothing: float = 0.65) -> None:
        self.smoothing = smoothing
        self._rms = 0.0
        self._lock = threading.Lock()

    @property
    def rms(self) -> float:
        with self._lock:
            return self._rms

    def update(self, indata: np.ndarray) -> None:
        mono = indata.mean(axis=1) if indata.ndim > 1 else indata
        rms = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64)))
        with self._lock:
            self._rms = (self.smoothing * self._rms) + ((1.0 - self.smoothing) * rms)


def rms_to_bend_value(
    rms: float,
    min_db: float,
    max_db: float,
    bend_min: float,
    bend_max: float,
) -> float:
    db = 20.0 * math.log10(max(rms, 1e-8))
    normalized = (db - min_db) / max(max_db - min_db, 1e-8)
    normalized = float(np.clip(normalized, 0.0, 1.0))
    return bend_min + (normalized * (bend_max - bend_min))


def pil_to_bgr(image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def build_wrapper(
    model_id_or_path: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    acceleration: Literal["none", "xformers", "tensorrt"],
    warmup: int,
    seed: int,
    fixed_noise: bool,
) -> StreamDiffusionWrapper:
    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        t_index_list=[0, 16, 32, 45],
        frame_buffer_size=1,
        width=width,
        height=height,
        warmup=warmup,
        acceleration=acceleration,
        mode="txt2img",
        use_denoising_batch=False,
        cfg_type="none",
        seed=seed,
        do_add_noise=False,
    )
    stream.prepare(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=1.0,
    )
    if fixed_noise:
        stream.stream.input_noise = torch.randn(
            (1, 4, stream.stream.latent_height, stream.stream.latent_width),
            device=stream.stream.device,
            dtype=stream.stream.dtype,
        )
    return stream


def run(
    model_id_or_path: str = "CompVis/stable-diffusion-v1-4",
    prompt: str = "an abstract glowing orb, colorful light trails, high contrast",
    negative_prompt: str = "low quality, text, words, letters",
    width: int = 384,
    height: int = 384,
    acceleration: Literal["none", "xformers", "tensorrt"] = "none",
    warmup: int = 4,
    seed: int = 2,
    sample_rate: int = 48000,
    block_size: int = 1024,
    device: Optional[int] = None,
    min_db: float = -55.0,
    max_db: float = -15.0,
    bend_min: float = 0.0,
    bend_max: float = 2.0 * math.pi,
    rms_smoothing: float = 0.65,
    target_fps: float = 6.0,
    fixed_noise: bool = True,
) -> None:
    audio = LiveRMS(smoothing=rms_smoothing)

    def audio_callback(indata, frames, callback_time, status) -> None:
        if status:
            print(status, flush=True)
        audio.update(indata)

    stream = build_wrapper(
        model_id_or_path=model_id_or_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        acceleration=acceleration,
        warmup=warmup,
        seed=seed,
        fixed_noise=fixed_noise,
    )

    frame_interval = 1.0 / max(target_fps, 1e-6)
    last_frame_time = 0.0
    window_name = "live_rms - press q or Esc to quit"

    print("Starting live RMS visualizer. Press q or Esc in the preview window to quit.")
    print(f"RMS dB range: {min_db:.1f} to {max_db:.1f}; bend range: {bend_min:.3f} to {bend_max:.3f}")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            blocksize=block_size,
            channels=1,
            dtype="float32",
            device=device,
            callback=audio_callback,
        ):
            while True:
                elapsed = time.perf_counter() - last_frame_time
                if elapsed < frame_interval:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                    time.sleep(min(frame_interval - elapsed, 0.005))
                    continue

                last_frame_time = time.perf_counter()
                rms = audio.rms
                bend_value = rms_to_bend_value(
                    rms=rms,
                    min_db=min_db,
                    max_db=max_db,
                    bend_min=bend_min,
                    bend_max=bend_max,
                )

                stream.stream.bending_fn = util.rotate_x(bend_value)
                stream.stream.bending_layer = None

                with torch.no_grad():
                    image = stream()

                preview = pil_to_bgr(image)
                db = 20.0 * math.log10(max(rms, 1e-8))
                cv2.putText(
                    preview,
                    f"RMS {rms:.4f}  {db:6.1f} dB  bend {bend_value:.3f}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, preview)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live microphone RMS to StreamDiffusion bending preview.")
    parser.add_argument("--model-id-or-path", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--prompt", default="an abstract glowing orb, colorful light trails, high contrast")
    parser.add_argument("--negative-prompt", default="low quality, text, words, letters")
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--acceleration", choices=["none", "xformers", "tensorrt"], default="none")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--device", type=int, default=None, help="Optional sounddevice input device id.")
    parser.add_argument("--min-db", type=float, default=-55.0)
    parser.add_argument("--max-db", type=float, default=-15.0)
    parser.add_argument("--bend-min", type=float, default=0.0)
    parser.add_argument("--bend-max", type=float, default=2.0 * math.pi)
    parser.add_argument("--rms-smoothing", type=float, default=0.65)
    parser.add_argument("--target-fps", type=float, default=6.0)
    parser.add_argument(
        "--random-noise",
        action="store_false",
        dest="fixed_noise",
        help="Use fresh txt2img noise each frame instead of a fixed latent seed.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(**vars(parse_args()))