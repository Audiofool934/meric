"""``meric`` command-line interface.

meric models                                  # list available pretrained models
meric generate --image photo.jpg -o outputs/      # image -> music
meric generate --video clip.mp4 -o outputs/        # video -> music
meric generate --text "calm piano, rain" -o outputs/
meric generate --muq-npy emb.npy -o outputs/
"""

import argparse
import sys


def _add_gen_args(p):
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="Image file")
    src.add_argument("--video", help="Video file")
    src.add_argument("--text", help="Text prompt")
    src.add_argument("--muq-npy", dest="muq_npy", help="Pre-computed MuQ embedding (.npy), skips Stage 1")
    p.add_argument("--model", default="meric-sft-v3", help="Pretrained model name (see `meric models`)")
    p.add_argument("-o", "--output-dir", dest="output_dir", default="outputs/meric")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("-n", type=int, default=1, help="Number of variants")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--video-max-frames", dest="video_max_frames", type=int, default=8)
    p.add_argument("--rdm-steps", dest="rdm_steps", type=int, default=20)
    p.add_argument("--rdm-guidance", dest="rdm_guidance", type=float, default=2.0)
    p.add_argument("--guidance-scale", dest="guidance_scale", type=float, default=5.0)
    p.add_argument("--sample-steps", dest="sample_steps", type=int, default=50)
    p.add_argument("--sample-method", dest="sample_method", default="dopri5", choices=["euler", "midpoint", "dopri5"])


def main(argv=None):
    parser = argparse.ArgumentParser(prog="meric", description="Meric: image, video, or text to music")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("models", help="List available pretrained models")
    _add_gen_args(sub.add_parser("generate", help="Generate music from an image, video, text, or MuQ embedding"))
    args = parser.parse_args(argv)

    if args.cmd == "models":
        from meric import hub

        for name, desc in hub.list_models().items():
            print(f"  {name:24s} {desc}")
        return 0

    if args.cmd == "generate":
        from meric.pipeline import generate

        wavs = generate(
            image=args.image,
            video=args.video,
            text=args.text,
            muq=args.muq_npy,
            model_name=args.model,
            device=args.device,
            output_dir=args.output_dir,
            n=args.n,
            seed=args.seed,
            video_max_frames=args.video_max_frames,
            rdm_steps=args.rdm_steps,
            rdm_guidance=args.rdm_guidance,
            guidance_scale=args.guidance_scale,
            sample_steps=args.sample_steps,
            sample_method=args.sample_method,
        )
        for w in wavs:
            print(w)
        return 0


if __name__ == "__main__":
    sys.exit(main())
