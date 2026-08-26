"""
Run many audio2video generations in a row
"""

from pathlib import Path
import sys
import os

import librosa.util

# Ensure local imports (e.g. `utils`) resolve when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils.bending as util
import audio2video
try:
    from transformers import EncodecModel, AutoProcessor
except ImportError:
    EncodecModel = None
    AutoProcessor = None


# User Input:
INPUTS_DIR = Path(os.environ.get("JAES_INPUTS_DIR", PROJECT_ROOT / "inputs"))
prompt_file_path = Path(os.environ.get("JAES_PROMPT_FILE", INPUTS_DIR / "test.txt"))


inputs = []
# inputs.append(
#     {
#          "audio": Path(""),
#          "prompt": Path(""),
#          "layer": 1,
#          "bend": util,
#          "feature": util,
#          "smoothing": None,
#          "seed": 46,
#     }
# )

kernels = [41]
alphas = [0.5, 0.7]
thresholds = [0]
layers = [1, 420]
w = 512
h = 512

audio_feature_fn = util.rms
bend_fn = util.multiply

if EncodecModel is not None and AutoProcessor is not None:
    try:
        model = EncodecModel.from_pretrained("facebook/encodec_48khz")
        processor = AutoProcessor.from_pretrained("facebook/encodec_48khz")
        audio_feature_fn = util.encodec(model, processor)
        bend_fn = util.tensor_multiply
        print(">>> Using EnCodec audio features")
    except Exception as exc:
        print(f">>> EnCodec unavailable ({exc}); falling back to RMS + multiply")
else:
    print(">>> EnCodecModel not available in this transformers version; falling back to RMS + multiply")

audio_inputs = [
    INPUTS_DIR / "Singularity.wav",
]

audio_inputs_override = os.environ.get("JAES_AUDIO_FILES")
if audio_inputs_override:
    audio_inputs = [Path(p.strip()) for p in audio_inputs_override.split(",") if p.strip()]

if not prompt_file_path.exists():
    raise FileNotFoundError(
        f"Prompt file not found: {prompt_file_path}. "
        f"Set JAES_PROMPT_FILE or place test.txt under {INPUTS_DIR}."
    )

missing_audio = [path for path in audio_inputs if not path.exists()]
if missing_audio:
    missing_list = ", ".join(str(path) for path in missing_audio)
    raise FileNotFoundError(
        f"Missing audio input(s): {missing_list}. "
        f"Set JAES_AUDIO_FILES or place files under {INPUTS_DIR}."
    )


# for audio in librosa.util.find_files(Path("C:\\Users\dzluk\StreamDiffusion\inputs\handmade")):
#     audio = Path(audio)
#     for layer in layers:
#         inputs.append(
#             {
#                  "audio": audio,
#                  "prompt": prompt_file_path,
#                  "layer": layer,
#                  "bend": util.tensor_multiply,
#                  "feature": util.encodec(model, processor),
#                  "smoothing": util.smooth(kernel_size, envelope_alpha, gate_threshold),
#                  "seed": 42,
#                  "width": w,
#                  "height": h
#             }
#         )
for audio in audio_inputs:
    for layer in layers:
        for k in kernels:
            for a in alphas:
                for t in thresholds:
                    inputs.append(
                        {
                             "audio": audio,
                             "prompt": prompt_file_path,
                             "layer": layer,
                             "bend": bend_fn,
                             "feature": audio_feature_fn,
                             "smoothing": util.smooth(k, a, t),
                             "seed": 42,
                             "width": w,
                             "height": h
                        }
                    )
# inputs.append(
#     {
#          "audio": Path("C:\\Users\dzluk\StreamDiffusion\inputs\Vn-ord_pont-G#4-mf.wav"),
#          "prompt": prompt_file_path,
#          "layer": 1,
#          "bend": util.six_plane_rotation,
#          "feature": util.encodec(model, processor),
#          "smoothing": util.smooth(kernel_size, envelope_alpha, gate_threshold),
#          "seed": 42,
#          "width": w,
#          "height": h
#     }
# )
# inputs.append(
#     {
#          "audio": Path("C:\\Users\dzluk\StreamDiffusion\inputs\Breathing-stems\Breathing-charms+noise-v3.wav"),
#          "prompt": prompt_file_path,
#          "layer": 1,
#          "bend": util.multiply,
#          "feature": util.rms,
#          "smoothing": util.smooth(7, 0.9, 0.05),
#          "seed": 47,
#          "width": w,
#          "height": h
#     }
# )
# inputs.append(
#     {
#          "audio": Path("C:\\Users\dzluk\StreamDiffusion\inputs\Breathing-stems\Breathing-charms-v3.wav"),
#          "prompt": prompt_file_path,
#          "layer": 1,
#          "bend": util.multiply,
#          "feature": util.rms,
#          "smoothing": util.smooth(5, 0.5, 0.05),
#          "seed": 43,
#          "width": w,
#          "height": h
#     }
# )

# inputs.append(
#     {
#          "audio": Path("C:\\Users\dzluk\StreamDiffusion\inputs\Breathing-stems\Breathing-paulstretch-v3.wav"),
#          "prompt": prompt_file_path,
#          "layer": 1,
#          "bend": util.rotate_x,
#          "feature": util.centroid,
#          "smoothing": util.smooth(kernel_size, envelope_alpha, gate_threshold),
#          "seed": 46,
#          "width": w,
#          "height": h
#     }
# )
# inputs.append(
#     {
#          "audio": Path("C:\\Users\dzluk\StreamDiffusion\inputs\Breathing-stems\Breathing-noise-v3.wav"),
#          "prompt": prompt_file_path,
#          "layer": 1,
#          "bend": util.rotate_x,
#          "feature": util.rms,
#          "smoothing": util.smooth(kernel_size, envelope_alpha, gate_threshold),
#          "seed": 47,
#          "width": w,
#          "height": h
#     }
# )
# inputs.append(
#     {
#          "audio": Path("C:\\Users\dzluk\StreamDiffusion\inputs\Breathing-stems\Breathing-modular-v3.wav"),
#          "prompt": prompt_file_path,
#          "layer": 1,
#          "bend": util.rotate_x,
#          "feature": util.rms,
#          "smoothing": util.smooth(kernel_size, envelope_alpha, gate_threshold),
#          "seed": 43,
#          "width": w,
#          "height": h
#     }
# )
# inputs.append(
#     {
#          "audio": Path("C:\\Users\dzluk\StreamDiffusion\inputs\Breathing-stems\Breathing-cycles-v3.wav"),
#          "prompt": prompt_file_path,
#          "layer": 1,
#          "bend": util.rotate_x,
#          "feature": util.rms,
#          "smoothing": util.smooth(kernel_size, envelope_alpha, gate_threshold),
#          "seed": 44,
#          "width": w,
#          "height": h
#     }
# )
# inputs.append(
#     {
#          "audio": Path("C:\\Users\dzluk\StreamDiffusion\inputs\Breathing-stems\Breathing-charms-v3.wav"),
#          "prompt": prompt_file_path,
#          "layer": 1,
#          "bend": util.multiply,
#          "feature": util.rms,
#          "smoothing": None,
#          "seed": 43,
#          "width": w,
#          "height": h
#     }
# )
# inputs.append(
#     {
#          "audio": Path("C:\\Users\dzluk\StreamDiffusion\inputs\ihadyou.wav"),
#          "prompt": Path("C:\\Users\dzluk\StreamDiffusion\inputs\gabe.txt"),
#          "layer": 1,
#          "bend": None,
#          "feature": util.rms,
#          "smoothing": util.smooth(kernel_size, envelope_alpha, gate_threshold),
#          "seed": 43,
#          "width": w,
#          "height": h
#     }
# )
# inputs.append(
#     {
#          "audio": Path("C:\\Users\dzluk\StreamDiffusion\inputs\ihadyou.wav"),
#          "prompt": Path("C:\\Users\dzluk\StreamDiffusion\inputs\gabe.txt"),
#          "layer": 1,
#          "bend": util.threshold,
#          "feature": util.rms,
#          "smoothing": util.smooth(15, 0.9, gate_threshold),
#          "seed": 43,
#          "width": w,
#          "height": h
#     }
# )




# def median_filtering(data):
#     return medfilt(data, kernel_size=9)
#
#
# def gaussian_filtering(data):
#     return gaussian_filter1d(data, sigma=3, radius=4)
#
#
# def log_smoothing(data):
#     return [math.log(x) if x > 0 else 0 for x in data]


for input in inputs:
    audio2video.generate_video(
                          input["audio"],
                          input["prompt"],
                          input["layer"],
                          input["bend"],
                          input["feature"],
                          input["smoothing"],
                          input["seed"],
                          input["width"],
                          input["height"]
    )
