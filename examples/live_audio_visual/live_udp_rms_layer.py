import argparse
import os
import socket
import sys
import time
import torch

import cv2
import numpy as np


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


def make_bending_fn(operator, amount):
    if operator == "rotate_x":
        return bend_util.rotate_x(amount)
    if operator == "rotate_y":
        return bend_util.rotate_y(amount)
    if operator == "rotate_z":
        return bend_util.rotate_z(amount)
    if operator == "multiply":
        return bend_util.multiply(1.0 + amount)
    if operator == "add_noise":
        return bend_util.add_noise(amount)
    raise ValueError(f"Unknown operator: {operator}")


def recv_latest_rms(sock, current_rms):
    while True:
        try:
            data, _ = sock.recvfrom(1024)
        except BlockingIOError:
            break

        text = data.decode("utf-8").strip()

        try:
            current_rms = float(text)
            continue
        except ValueError:
            pass

        # Allows future messages like: "rms 0.0123"
        parts = text.replace(",", " ").split()
        if len(parts) >= 2 and parts[0].lower() == "rms":
            try:
                current_rms = float(parts[1])
            except ValueError:
                pass

    return current_rms


def pil_to_bgr(image):
    arr = np.array(image.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def make_fixed_noise(stream, noise_seed):
    generator = torch.Generator(device=stream.stream.device)
    generator.manual_seed(noise_seed)

    return torch.randn(
        (1, 4, stream.stream.latent_height, stream.stream.latent_width),
        generator=generator,
        device=stream.stream.device,
        dtype=stream.stream.dtype,
    )


def render_frame(stream, fixed_noise=None):
    if fixed_noise is None:
        output = stream()
        if isinstance(output, list):
            output = output[0]
        return output

    image_tensor = stream.stream.predict_x0_batch(fixed_noise)
    image_tensor = stream.stream.decode_image(image_tensor).detach().clone()
    output = stream.postprocess_image(image_tensor, output_type=stream.output_type)

    if isinstance(output, list):
        output = output[0]

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--udp-host", default="0.0.0.0")
    parser.add_argument("--udp-port", type=int, default=5006)

    parser.add_argument("--model-id-or-path", default="KBlueLeaf/kohaku-v2.1")
    parser.add_argument("--prompt", default="colorful abstract geometric shapes, vivid, clean composition")
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--target-fps", type=float, default=6.0)
    parser.add_argument("--acceleration", default="none", choices=["none", "xformers", "tensorrt"])

    parser.add_argument("--bending-layer", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--operator", default="rotate_x",
                        choices=["rotate_x", "rotate_y", "rotate_z", "multiply", "add_noise"])

    parser.add_argument("--db-min", type=float, default=-55.0)
    parser.add_argument("--db-max", type=float, default=-15.0)
    parser.add_argument("--bend-min", type=float, default=0.0)
    parser.add_argument("--bend-max", type=float, default=6.28318530718)
    parser.add_argument("--smooth", type=float, default=0.25)

    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--fixed-noise", action="store_true")
    parser.add_argument("--noise-seed", type=int, default=1234)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.udp_host, args.udp_port))
    sock.setblocking(False)

    print(f"Listening for UDP RMS on {args.udp_host}:{args.udp_port}")
    print(f"Layer-based bending: layer={args.bending_layer}, operator={args.operator}")

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
    )

    fixed_noise = None
    if args.fixed_noise:
        fixed_noise = make_fixed_noise(stream, args.noise_seed)

    stream.prepare(
        prompt=args.prompt,
        num_inference_steps=50,
        bending_fn=make_bending_fn(args.operator, 0.0),
        bending_layer=args.bending_layer,
        input_noise=fixed_noise,
    )

    current_rms = 0.0
    smooth_rms = 0.0
    frame_delay = 1.0 / max(args.target_fps, 0.1)

    cv2.namedWindow("live_udp_rms_layer", cv2.WINDOW_NORMAL)

    while True:
        start = time.time()

        current_rms = recv_latest_rms(sock, current_rms)
        smooth_rms = (1.0 - args.smooth) * smooth_rms + args.smooth * current_rms

        db = rms_to_db(smooth_rms)
        amount = scale_clamped(db, args.db_min, args.db_max, args.bend_min, args.bend_max)

        # This is the important layer-based part:
        # the function is applied inside StreamDiffusion.unet_step()
        # only when idx == bending_layer.
        stream.stream.bending_fn = make_bending_fn(args.operator, amount)
        stream.stream.bending_layer = args.bending_layer

        output = render_frame(stream, fixed_noise=fixed_noise)

        frame = pil_to_bgr(output)

        hud = f"RMS {smooth_rms:.4f}  {db:6.1f} dB  bend {amount:.3f}  layer {args.bending_layer}"
        cv2.putText(frame, hud, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        cv2.imshow("live_udp_rms_layer", frame)

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