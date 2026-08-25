import argparse
import os
import socket
import sys
import time
import math

import cv2
import numpy as np
import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.wrapper import StreamDiffusionWrapper
import utils.bending as bend_util


def rms_to_db(rms, floor_db=-80.0):
    rms = max(float(rms), 1e-8)
    db = 20.0 * np.log10(rms)
    return max(db, floor_db)


def scale_clamped(value, in_min, in_max, out_min, out_max):
    if in_max == in_min:
        return out_min
    value = min(max(value, in_min), in_max)
    t = (value - in_min) / (in_max - in_min)
    return out_min + t * (out_max - out_min)


def parse_feature_message(text, state):
    parts = text.replace(",", " ").split()

    if len(parts) == 1:
        try:
            state["rms"] = float(parts[0])
        except ValueError:
            pass
        return state

    i = 0
    while i < len(parts) - 1:
        key = parts[i].lower()
        try:
            value = float(parts[i + 1])
        except ValueError:
            i += 1
            continue

        if key in state:
            state[key] = value

        i += 2

    return state


def recv_latest_features(sock, state):
    while True:
        try:
            data, _ = sock.recvfrom(1024)
        except BlockingIOError:
            break

        text = data.decode("utf-8").strip()
        state = parse_feature_message(text, state)

    return state


def rotate_y_safe(amount):
    def fn(x):
        c = np.cos(amount)
        s = np.sin(amount)
        rotation_matrix = [
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-1 * s, 0, c, 0],
            [0, 0, 0, 1],
        ]
        op = torch.tensor(rotation_matrix, device=x.device, dtype=x.dtype)
        return torch.tensordot(op, x, dims=1)

    return fn


def rotate_z_safe(amount):
    def fn(x):
        c = np.cos(amount)
        s = np.sin(amount)
        rotation_matrix = [
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        op = torch.tensor(rotation_matrix, device=x.device, dtype=x.dtype)
        return torch.tensordot(op, x, dims=1)

    return fn


def make_rms_bending_fn(rms_amount):
    return bend_util.multiply(1.0 + rms_amount)


def make_centroid_bending_fn(centroid_amount):
    return rotate_y_safe(centroid_amount)


def chroma_to_circle_of_fifths_amount(chroma_index, chroma_strength, chroma_bend_max):
    chroma_index = int(chroma_index) % 12
    chroma_strength = float(chroma_strength)

    circle_of_fifths_rank = {
        0: 0,    # C
        7: 1,    # G
        2: 2,    # D
        9: 3,    # A
        4: 4,    # E
        11: 5,   # B
        6: 6,    # F#
        1: 7,    # C#
        8: 8,    # G#
        3: 9,    # D#
        10: 10,  # A#
        5: 11,   # F
    }

    rank = circle_of_fifths_rank[chroma_index]
    angle = (rank / 12.0) * 2.0 * np.pi
    return np.sin(angle) * chroma_strength * chroma_bend_max


def make_chroma_bending_fn(chroma_amount):
    return rotate_y_safe(chroma_amount)


def make_onset_bending_fn(onset_pulse, onset_bend_max):
    amount = onset_pulse * onset_bend_max
    return bend_util.add_full(amount)


def update_chroma_stability(
    raw_chroma,
    raw_strength,
    stable_chroma,
    stable_chroma_strength,
    candidate_chroma,
    candidate_count,
    chroma_release_count,
    strength_min,
    stability_frames,
    release_frames,
):
    raw_chroma = int(raw_chroma) % 12
    raw_strength = float(raw_strength)

    if raw_strength >= strength_min:
        if candidate_chroma == raw_chroma:
            candidate_count += 1
        else:
            candidate_chroma = raw_chroma
            candidate_count = 1

        if candidate_count >= stability_frames:
            stable_chroma = raw_chroma
            stable_chroma_strength = raw_strength
            chroma_release_count = release_frames
    else:
        candidate_count = 0
        candidate_chroma = None

        if chroma_release_count > 0:
            chroma_release_count -= 1
            stable_chroma_strength *= 0.92
        else:
            stable_chroma_strength = 0.0

    return (
        stable_chroma,
        stable_chroma_strength,
        candidate_chroma,
        candidate_count,
        chroma_release_count,
    )


def compose_bending_fns(*fns):
    def fn(x):
        for bending_fn in fns:
            x = bending_fn(x)
        return x

    return fn


def add_bending_fn(bending_map, layer, bending_fn):
    if layer in bending_map:
        bending_map[layer] = compose_bending_fns(bending_map[layer], bending_fn)
    else:
        bending_map[layer] = bending_fn


class NoisePath:
    def __init__(self, stream, mode="walk", seed=1234, walk_step=0.006):
        self.stream = stream
        self.mode = mode
        self.walk_step = walk_step
        self.phase = 0.0

        generator = torch.Generator(device=stream.stream.device)
        generator.manual_seed(seed)

        shape = (1, 4, stream.stream.latent_height, stream.stream.latent_width)

        self.noise_a = torch.randn(
            shape,
            generator=generator,
            device=stream.stream.device,
            dtype=stream.stream.dtype,
        )

        self.noise_b = torch.randn(
            shape,
            generator=generator,
            device=stream.stream.device,
            dtype=stream.stream.dtype,
        )

    def next(self, activity=1.0):
        if self.mode == "fixed":
            return self.noise_a

        if self.mode == "random":
            return torch.randn_like(self.noise_a)

        if self.mode == "walk":
            activity = max(0.0, min(float(activity), 1.0))

            c = math.cos(self.phase)
            s = math.sin(self.phase)
            noise = c * self.noise_a + s * self.noise_b

            self.phase += self.walk_step * activity
            return noise

        raise ValueError(f"Unknown noise mode: {self.mode}")


def render_frame(stream, noise, seed=None):
    if seed is not None:
        stream.stream.seed_everything(seed)

    image_tensor = stream.stream.predict_x0_batch(noise)
    image_tensor = stream.stream.decode_image(image_tensor).detach().clone()
    output = stream.postprocess_image(image_tensor, output_type=stream.output_type)

    if isinstance(output, list):
        output = output[0]

    return output


def pil_to_bgr(image):
    arr = np.array(image.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--udp-host", default="0.0.0.0")
    parser.add_argument("--udp-port", type=int, default=5006)

    parser.add_argument("--model-id-or-path", default="KBlueLeaf/kohaku-v2.1")
    parser.add_argument(
        "--prompt",
        default="one centered colorful geometric crystal, single faceted object, symmetric shape, simple dark background, stable composition, abstract, vivid",
    )
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--target-fps", type=float, default=6.0)
    parser.add_argument("--acceleration", default="none", choices=["none", "xformers", "tensorrt"])

    parser.add_argument("--db-min", type=float, default=-55.0)
    parser.add_argument("--db-max", type=float, default=-15.0)
    parser.add_argument("--rms-bend-min", type=float, default=0.0)
    parser.add_argument("--rms-bend-max", type=float, default=6.28318530718)

    parser.add_argument("--centroid-min", type=float, default=300.0)
    parser.add_argument("--centroid-max", type=float, default=5000.0)
    parser.add_argument("--centroid-bend-min", type=float, default=-0.7)
    parser.add_argument("--centroid-bend-max", type=float, default=0.7)

    parser.add_argument("--rms-layer", type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument("--centroid-layer", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--onset-layer", type=int, default=1, choices=[0, 1, 2, 3])
    parser.add_argument("--chroma-layer", type=int, default=1, choices=[0, 1, 2, 3])

    parser.add_argument("--onset-bend-max", type=float, default=0.8)
    parser.add_argument("--onset-decay", type=float, default=0.82)

    parser.add_argument("--chroma-bend-max", type=float, default=0.8)
    parser.add_argument("--chroma-strength-min", type=float, default=0.35)
    parser.add_argument("--chroma-stability-frames", type=int, default=4)
    parser.add_argument("--chroma-release-frames", type=int, default=8)
    parser.add_argument("--chroma-smooth", type=float, default=0.12)

    parser.add_argument("--smooth", type=float, default=0.25)

    parser.add_argument("--fixed-noise", action="store_true")
    parser.add_argument("--noise-mode", default="walk", choices=["fixed", "random", "walk"])
    parser.add_argument("--noise-seed", type=int, default=1234)
    parser.add_argument("--noise-walk-step", type=float, default=0.006)
    parser.add_argument("--seed", type=int, default=2)

    args = parser.parse_args()

    if args.fixed_noise:
        args.noise_mode = "fixed"

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.udp_host, args.udp_port))
    sock.setblocking(False)

    print(f"Listening for UDP features on {args.udp_host}:{args.udp_port}")
    print(
        f"RMS layer={args.rms_layer}; "
        f"centroid layer={args.centroid_layer}; "
        f"onset layer={args.onset_layer}; "
        f"chroma layer={args.chroma_layer}"
    )

    stream = StreamDiffusionWrapper(
        model_id_or_path=args.model_id_or_path,
        t_index_list=[0, 16, 32, 45],
        frame_buffer_size=1,
        width=args.width,
        height=args.height,
        warmup=10,
        acceleration=args.acceleration,
        mode="txt2img",
        use_denoising_batch=False,
        cfg_type="none",
        seed=args.seed,
        do_add_noise=True,
    )

    noise_path = NoisePath(
        stream,
        mode=args.noise_mode,
        seed=args.noise_seed,
        walk_step=args.noise_walk_step,
    )

    initial_bending_map = {}
    add_bending_fn(initial_bending_map, args.rms_layer, make_rms_bending_fn(0.0))
    add_bending_fn(initial_bending_map, args.centroid_layer, make_centroid_bending_fn(0.0))
    add_bending_fn(initial_bending_map, args.onset_layer, make_onset_bending_fn(0.0, args.onset_bend_max))
    add_bending_fn(initial_bending_map, args.chroma_layer, make_chroma_bending_fn(0.0))

    stream.prepare(
        prompt=args.prompt,
        num_inference_steps=50,
        bending_fn=initial_bending_map,
        bending_layer=None,
        input_noise=noise_path.next(),
    )

    state = {
        "rms": 0.0,
        "centroid": 0.0,
        "onset": 0.0,
        "onset_strength": 0.0,
        "chroma": 0.0,
        "chroma_strength": 0.0,
    }

    smooth_state = {
        "rms": 0.0,
        "centroid": 0.0,
    }

    onset_pulse = 0.0

    stable_chroma = 0
    stable_chroma_strength = 0.0
    candidate_chroma = None
    candidate_count = 0
    chroma_release_count = 0
    smooth_chroma_amount = 0.0

    frame_delay = 1.0 / max(args.target_fps, 0.1)

    cv2.namedWindow("live_udp_features_layer", cv2.WINDOW_NORMAL)

    while True:
        start = time.time()

        state = recv_latest_features(sock, state)

        (
            stable_chroma,
            stable_chroma_strength,
            candidate_chroma,
            candidate_count,
            chroma_release_count,
        ) = update_chroma_stability(
            state["chroma"],
            state["chroma_strength"],
            stable_chroma,
            stable_chroma_strength,
            candidate_chroma,
            candidate_count,
            chroma_release_count,
            args.chroma_strength_min,
            args.chroma_stability_frames,
            args.chroma_release_frames,
        )

        if state.get("onset", 0.0) >= 1.0:
            onset_strength = max(0.25, min(state.get("onset_strength", 1.0), 1.0))
            onset_pulse = max(onset_pulse, onset_strength)
        else:
            onset_pulse *= args.onset_decay

        smooth_state["rms"] = (
            (1.0 - args.smooth) * smooth_state["rms"]
            + args.smooth * state["rms"]
        )
        smooth_state["centroid"] = (
            (1.0 - args.smooth) * smooth_state["centroid"]
            + args.smooth * state["centroid"]
        )

        db = rms_to_db(smooth_state["rms"])

        rms_amount = scale_clamped(
            db,
            args.db_min,
            args.db_max,
            args.rms_bend_min,
            args.rms_bend_max,
        )

        centroid_amount = scale_clamped(
            smooth_state["centroid"],
            args.centroid_min,
            args.centroid_max,
            args.centroid_bend_min,
            args.centroid_bend_max,
        )

        if stable_chroma_strength >= args.chroma_strength_min:
            target_chroma_amount = chroma_to_circle_of_fifths_amount(
                stable_chroma,
                stable_chroma_strength,
                args.chroma_bend_max,
            )
        else:
            target_chroma_amount = 0.0

        smooth_chroma_amount = (
            (1.0 - args.chroma_smooth) * smooth_chroma_amount
            + args.chroma_smooth * target_chroma_amount
        )

        bending_map = {}

        if centroid_amount != 0.0:
            add_bending_fn(
                bending_map,
                args.centroid_layer,
                make_centroid_bending_fn(centroid_amount),
            )

        if rms_amount != 0.0:
            add_bending_fn(
                bending_map,
                args.rms_layer,
                make_rms_bending_fn(rms_amount),
            )

        if onset_pulse > 0.01 and args.onset_bend_max != 0.0:
            add_bending_fn(
                bending_map,
                args.onset_layer,
                make_onset_bending_fn(onset_pulse, args.onset_bend_max),
            )

        if abs(smooth_chroma_amount) > 0.01:
            add_bending_fn(
                bending_map,
                args.chroma_layer,
                make_chroma_bending_fn(smooth_chroma_amount),
            )

        stream.stream.bending_fn = bending_map
        stream.stream.bending_layer = None

        rms_activity = min(max(rms_amount / max(args.rms_bend_max, 1e-8), 0.0), 1.0)
        audio_activity = max(stable_chroma_strength, onset_pulse, rms_activity)

        noise = noise_path.next(activity=audio_activity)
        output = render_frame(stream, noise, seed=args.seed)
        frame = pil_to_bgr(output)

        hud1 = f"RMS {smooth_state['rms']:.4f}  {db:6.1f} dB  mult {rms_amount:.2f}"
        hud2 = f"centroid {smooth_state['centroid']:.0f} Hz  rotY {centroid_amount:.2f}"
        hud3 = f"onset {state.get('onset', 0.0):.0f}  pulse {onset_pulse:.2f}"
        hud4 = (
            f"raw chroma {int(state['chroma'])} {state['chroma_strength']:.2f}  "
            f"stable {stable_chroma} {stable_chroma_strength:.2f}"
        )
        hud5 = f"chroma smooth amount {smooth_chroma_amount:.2f}  activity {audio_activity:.2f}"

        cv2.putText(frame, hud1, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        cv2.putText(frame, hud2, (18, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        cv2.putText(frame, hud3, (18, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        cv2.putText(frame, hud4, (18, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        cv2.putText(frame, hud5, (18, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

        cv2.imshow("live_udp_features_layer", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord("q")]:
            break

        elapsed = time.time() - start
        if elapsed < frame_delay:
            time.sleep(frame_delay - elapsed)

    sock.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()