"""Single-image, single-language inference for the HarmoniVox release model."""

import argparse
import math
import os
from pathlib import Path

from release_assets import REPO_ROOT, checkpoint_pair, checkpoint_root, missing_assets, nltk_root, release_models, verify_model_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    registry = release_models()
    parser.add_argument("--model-profile", choices=(*registry["profiles"], "custom"), default=registry["default_profile"], help="Exact checkpoint identity; custom disables identity verification")
    parser.add_argument("--model-dir", type=Path, help="Directory with matching G/E checkpoints (default: ../checkpoint)")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/harmonivox.json", help="Matching model JSON configuration (default: bundled release configuration)")
    parser.add_argument("--checkpoint-step", type=int, default=12000, help="Explicit G/E checkpoint step (default: 12000)")
    parser.add_argument("--pretrained-dir", type=Path, help="Root for language BERT models and English text resources")
    parser.add_argument("--image", type=Path, help="Image prompt")
    parser.add_argument("--text", help="Text to synthesize")
    parser.add_argument("--language", choices=("EN", "ZH", "JP"), default="EN")
    parser.add_argument("--output", type=Path, default=Path("outputs/output.wav"))
    parser.add_argument("--device", default="cuda", help="PyTorch device, e.g. cuda, cuda:0, cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-scale", type=float, default=0.6)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--check-only", action="store_true", help="Check local assets without importing ML dependencies")
    parser.add_argument("--verify-weights", action="store_true", help="Also verify model identity during --check-only; synthesis always verifies named profiles")
    args = parser.parse_args()
    if args.pretrained_dir:
        os.environ["HARMONIVOX_PRETRAINED_DIR"] = str(args.pretrained_dir.expanduser().resolve())
    for name in ("HARMONIVOX_PRETRAINED_DIR", "HARMONIVOX_CHECKPOINT_DIR", "HARMONIVOX_NLTK_DIR"):
        if os.environ.get(name):
            os.environ[name] = str(Path(os.environ[name]).expanduser().resolve())
    model_dir = (args.model_dir or checkpoint_root()).expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    errors = missing_assets(model_dir, config_path, [args.language], step=args.checkpoint_step)
    if errors:
        parser.exit(1, "Asset check failed:\n" + "\n".join(f"  - {error}" for error in errors) + "\n")
    if args.check_only:
        if args.verify_weights and args.model_profile != "custom":
            try:
                verify_model_identity(model_dir, config_path, args.model_profile, args.checkpoint_step, registry)
            except ValueError as error:
                parser.exit(1, str(error) + "\n")
            print(f"Verified model profile: {args.model_profile}")
        print(f"Required {args.language} assets found. Model directory: {model_dir}")
        for path in checkpoint_pair(model_dir, step=args.checkpoint_step):
            print(f"  {path}")
        return
    if not args.text or not args.text.strip() or not args.image:
        parser.error("--text (nonempty) and --image are required for synthesis")
    if not all(math.isfinite(value) for value in (args.noise_scale, args.length_scale)):
        parser.error("synthesis controls must be finite numbers")
    if args.length_scale <= 0 or args.noise_scale < 0:
        parser.error("--length-scale must be positive; --noise-scale must be nonnegative")
    image_path = args.image.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not image_path.is_file():
        parser.error(f"Image not found: {image_path}")
    if output_path.exists():
        parser.error(f"Output already exists; choose another path: {output_path}")
    if args.model_profile != "custom":
        print(f"Verifying model profile: {args.model_profile}", flush=True)
        try:
            verify_model_identity(model_dir, config_path, args.model_profile, args.checkpoint_step, registry)
        except ValueError as error:
            parser.exit(1, str(error) + "\n")
        print(f"Verified model profile: {args.model_profile}", flush=True)
    else:
        print("Custom model profile: checkpoint identity is not verified", flush=True)

    # Legacy config.py resolves config.yml relative to the working directory.
    # Resolve user paths first so callers can launch this script from anywhere.
    os.chdir(REPO_ROOT)
    if args.language == "EN":
        # Keep text resources inside the declared asset directory; don't rely
        # on an unrelated user's global NLTK installation or download at import.
        os.environ["NLTK_DATA"] = str(nltk_root())
        import nltk
        nltk.data.path[:] = [str(nltk_root())]
    import numpy as np
    import soundfile as sf
    import torch
    from PIL import Image
    import utils
    from infer_web_qformer_textual_multi import get_net_g, infer, latest_version

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use --device cpu or a CUDA-enabled environment.")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    hps = utils.get_hparams_from_file(str(config_path))
    if getattr(hps.model, "use_sdp", True):
        parser.error("This release entry point expects the validated deterministic duration predictor (use_sdp=false).")
    net_e, net_g, preprocess = get_net_g(
        str(model_dir), getattr(hps, "version", latest_version), args.device, hps,
        checkpoint_step=args.checkpoint_step,
    )
    with Image.open(image_path) as image:
        image_prompt = preprocess(image.convert("RGB"))
    with torch.inference_mode():
        audio = infer(
            text=args.text.strip(), image_prompt=image_prompt, language=args.language,
            hps=hps, net_e=net_e, net_g=net_g, device=args.device,
            sid="imager", emotion=None, sdp_ratio=0.0,
            noise_scale=args.noise_scale, noise_scale_w=0.0,
            length_scale=args.length_scale,
        )
    if audio.size == 0 or not np.isfinite(audio).all():
        raise RuntimeError("Inference produced empty or nonfinite audio")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, hps.data.sampling_rate, subtype="PCM_16", format="WAV")
    print(f"Saved {len(audio) / hps.data.sampling_rate:.2f}s at {hps.data.sampling_rate} Hz to {output_path}")


if __name__ == "__main__":
    main()
