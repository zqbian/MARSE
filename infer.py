"""Minimal command-line inference for MARSE."""

import argparse
from pathlib import Path

import soundfile as sf
import torch

from marse import decompress_cirm, load_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--input",
        nargs=3,
        required=True,
        metavar=("ARRAY_1", "ARRAY_2", "ARRAY_3"),
        help="Three synchronized six-channel WAV files.",
    )
    parser.add_argument("--theta-l", nargs=3, type=float, required=True)
    parser.add_argument("--theta-h", nargs=3, type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--silence-threshold", type=float, default=0.5)
    return parser.parse_args()


def read_arrays(paths):
    arrays = []
    sample_rate = None
    n_samples = None
    for path in paths:
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        if audio.shape[1] != 6:
            raise ValueError(f"{path} must contain six channels, got {audio.shape[1]}.")
        if sample_rate is None:
            sample_rate, n_samples = rate, audio.shape[0]
        if rate != sample_rate or audio.shape[0] != n_samples:
            raise ValueError("All input files must have the same sample rate and length.")
        arrays.append(torch.from_numpy(audio.T))
    if sample_rate != 16000:
        raise ValueError(f"Expected 16-kHz audio, got {sample_rate} Hz.")
    return torch.stack(arrays).unsqueeze(0), sample_rate


@torch.inference_mode()
def main():
    args = parse_args()
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    model = load_model(args.checkpoint, device)
    waveform, sample_rate = read_arrays(args.input)
    waveform = waveform.to(device)
    n_samples = waveform.shape[-1]

    window = torch.sqrt(torch.hann_window(512, device=device))
    flat = waveform.reshape(-1, n_samples)
    stft = torch.stft(
        flat,
        n_fft=512,
        hop_length=256,
        window=window,
        center=True,
        onesided=True,
        return_complex=True,
    ).reshape(1, 3, 6, 257, -1)
    model_input = torch.cat((stft.real, stft.imag), dim=2)
    theta_l = torch.tensor([args.theta_l], device=device)
    theta_h = torch.tensor([args.theta_h], device=device)
    compressed_mask = model(model_input, theta_l, theta_h)
    mask = decompress_cirm(compressed_mask)

    selected = model.last_selector_weights.argmax(dim=1)
    batch_index = torch.arange(1, device=device)
    reference = stft[batch_index, selected, model.reference_channel]
    estimate_stft = reference * mask
    estimate = torch.istft(
        estimate_stft,
        n_fft=512,
        hop_length=256,
        window=window,
        center=True,
        onesided=True,
        length=n_samples,
    )
    speech_gate = (model.last_silence_prob <= args.silence_threshold).to(estimate.dtype)
    estimate = estimate * speech_gate[:, None]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, estimate[0].cpu().numpy(), sample_rate)
    scores = model.last_selector_weights[0].cpu().tolist()
    silence = model.last_silence_prob[0].item()
    print(f"selected_array={selected.item() + 1}")
    print(f"selector_weights={[round(value, 4) for value in scores]}")
    print(f"silence_probability={silence:.4f}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
