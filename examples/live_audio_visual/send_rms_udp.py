import argparse
import math
import socket
import threading
import time

import numpy as np
import sounddevice as sd


class LiveRMS:
    def __init__(self, smoothing: float = 0.65) -> None:
        self.smoothing = smoothing
        self._rms = 0.0
        self._lock = threading.Lock()

    def update(self, indata: np.ndarray) -> None:
        mono = indata.mean(axis=1) if indata.ndim > 1 else indata
        rms = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64)))
        with self._lock:
            self._rms = (self.smoothing * self._rms) + ((1.0 - self.smoothing) * rms)

    def get(self) -> float:
        with self._lock:
            return self._rms


def run(
    host: str,
    port: int = 5005,
    sample_rate: int = 48000,
    block_size: int = 1024,
    device: int | None = None,
    smoothing: float = 0.65,
    send_hz: float = 60.0,
) -> None:
    rms_state = LiveRMS(smoothing=smoothing)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def audio_callback(indata, frames, callback_time, status) -> None:
        if status:
            print(status, flush=True)
        rms_state.update(indata)

    print(f"Sending Mac microphone RMS to {host}:{port}")
    print("Press Ctrl+C to stop.")

    interval = 1.0 / max(send_hz, 1e-6)

    with sd.InputStream(
        samplerate=sample_rate,
        blocksize=block_size,
        channels=1,
        dtype="float32",
        device=device,
        callback=audio_callback,
    ):
        while True:
            rms = rms_state.get()
            db = 20.0 * math.log10(max(rms, 1e-8))
            message = f"{rms:.8f}".encode("utf-8")
            sock.sendto(message, (host, port))
            print(f"\rRMS {rms:.6f}  {db:6.1f} dB", end="", flush=True)
            time.sleep(interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send Mac microphone RMS to a remote visualizer over UDP.")
    parser.add_argument("--host", required=True, help="Windows receiver IP address.")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--smoothing", type=float, default=0.65)
    parser.add_argument("--send-hz", type=float, default=60.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(**vars(parse_args()))
