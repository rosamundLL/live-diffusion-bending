import argparse
import queue
import socket
import time

import librosa
import numpy as np
import sounddevice as sd


def compute_rms(audio):
    return float(np.sqrt(np.mean(audio * audio) + 1e-12))


def compute_centroid(audio, sr):
    window = np.hanning(len(audio))
    spectrum = np.abs(np.fft.rfft(audio * window))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    denom = np.sum(spectrum) + 1e-12
    return float(np.sum(freqs * spectrum) / denom)

def compute_chroma(audio, sr):
    try:
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=sr,
            n_fft=2048,
            hop_length=512,
        )

        chroma_vector = np.mean(chroma, axis=1)
        chroma_sum = float(np.sum(chroma_vector) + 1e-8)

        dominant_chroma = int(np.argmax(chroma_vector))
        chroma_strength = float(chroma_vector[dominant_chroma] / chroma_sum)

    except Exception:
        dominant_chroma = 0
        chroma_strength = 0.0
    return dominant_chroma, chroma_strength


class LiveOnsetDetector:
    def __init__(
        self,
        sr,
        buffer_seconds=1.0,
        hop_length=512,
        threshold=0.35,
        cooldown_blocks=20,
    ):
        self.sr = sr
        self.hop_length = hop_length
        self.threshold = threshold
        self.cooldown_blocks = cooldown_blocks
        self.cooldown = 0

        self.buffer = np.zeros(int(sr * buffer_seconds), dtype=np.float32)
        self.prev_strength = 0.0

    def update(self, audio_block):
        self.buffer = np.roll(self.buffer, -len(audio_block))
        self.buffer[-len(audio_block):] = audio_block

        onset_env = librosa.onset.onset_strength(
            y=self.buffer,
            sr=self.sr,
            hop_length=self.hop_length,
        )

        strength = float(np.max(onset_env[-4:])) if len(onset_env) else 0.0
        delta = max(0.0, strength - self.prev_strength)
        self.prev_strength = 0.85 * self.prev_strength + 0.15 * strength

        onset = 0
        if self.cooldown > 0:
            self.cooldown -= 1
        elif delta > self.threshold:
            onset = 1
            self.cooldown = self.cooldown_blocks

        return onset, delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=5006)
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--blocksize", type=int, default=2048)
    parser.add_argument("--device", type=int, default=None)

    parser.add_argument("--onset-threshold", type=float, default=2.0)
    parser.add_argument("--onset-cooldown-blocks", type=int, default=20)
    parser.add_argument("--onset-hold-blocks", type=int, default=4)

    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    audio_queue = queue.Queue()

    detector = LiveOnsetDetector(
        sr=args.sr,
        threshold=args.onset_threshold,
        cooldown_blocks=args.onset_cooldown_blocks,
    )

    onset_hold = 0

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        audio = indata[:, 0].astype(np.float32).copy()
        audio_queue.put(audio)

    print(f"Sending RMS / centroid / onset to {args.host}:{args.port}")
    print(
        f"onset threshold={args.onset_threshold}, "
        f"cooldown={args.onset_cooldown_blocks} blocks, "
        f"hold={args.onset_hold_blocks} blocks"
    )

    with sd.InputStream(
        samplerate=args.sr,
        blocksize=args.blocksize,
        channels=1,
        dtype="float32",
        device=args.device,
        callback=callback,
    ):
        while True:
            audio = audio_queue.get()

            rms = compute_rms(audio)
            centroid = compute_centroid(audio, args.sr)
            dominant_chroma, chroma_strength = compute_chroma(audio, args.sr)
            onset, onset_strength = detector.update(audio)

            if onset == 1:
                onset_hold = args.onset_hold_blocks
            elif onset_hold > 0:
                onset_hold -= 1

            send_onset = 1 if onset_hold > 0 else 0

            msg = (
                f"rms {rms:.8f} "
                f"centroid {centroid:.3f} "
                f"chroma {dominant_chroma:d} "
                f"chroma_strength {chroma_strength:.6f} "
                f"onset {send_onset:d} "
                f"onset_strength {onset_strength:.6f}"
            )

            sock.sendto(msg.encode("utf-8"), (args.host, args.port))

            print(msg, end="\r")
            time.sleep(0.001)
if __name__ == "__main__":
    main()
