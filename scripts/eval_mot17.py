"""Score detectors and trackers on MOT17, the way the README's accuracy tables are measured.

Usage:
    uv run python scripts/eval_mot17.py
    uv run python scripts/eval_mot17.py --models rtdetr-v2-r18,yolo26n --trackers bytetrack,botsort+osnet

The data is the ``val`` split of Roboflow's MOT17 download: the second half of each of the 7
MOT17 training sequences, 2,652 frames. Each detector runs once per sequence, and its detections
are replayed into every tracker, so trackers are compared on the same boxes. A tracker named
``name+osnet`` runs with ``--reid osnet``. Scores are HOTA, IDF1, MOTA and ID switches over all
7 sequences combined, printed as a Markdown table.

Roboflow's ``trackers`` package downloads the data (about 425 MB, once) and scores the runs. It
runs through ``uvx`` in its own environment because it needs the non-headless OpenCV.
"""

from __future__ import annotations

import argparse
import configparser
import copy
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

from trackflow import Detection, TrackflowPipeline

TRACKERS = "trackers==2.7.0"


def trackers_cli(*args: str, quiet: bool = False) -> None:
    if shutil.which("uvx") is None:
        sys.exit("eval_mot17.py runs Roboflow's trackers package through uvx: install uv first.")
    subprocess.run(
        ["uvx", "--from", TRACKERS, "trackers", *args],
        check=True,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.DEVNULL if quiet else None,
    )


def frame_rate(sequence: Path) -> float:
    info = configparser.ConfigParser()
    info.read(sequence / "seqinfo.ini")
    return float(info["Sequence"]["frameRate"])


class Replay:
    """Returns one sequence's stored detections, frame by frame."""

    def __init__(self, frames: list[list[Detection]]) -> None:
        self._frames = iter(frames)

    def detect(self, frame_bgr):
        # The pipeline attaches Re-ID embeddings to detections, so each run gets its own copy.
        return copy.deepcopy(next(self._frames))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", default="rtdetr-v2-r18", help="comma-separated presets")
    parser.add_argument(
        "--trackers", default="bytetrack", help="comma-separated, e.g. bytetrack,botsort+osnet"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu (the README's numbers), coreml for the Apple GPU, cuda; rfdetr-large takes mps",
    )
    parser.add_argument("--data", type=Path, default=Path.home() / ".cache/trackflow/mot17")
    parser.add_argument(
        "--out", type=Path, help="new directory to keep the MOT files and scores in"
    )
    args = parser.parse_args()

    val = args.data / "mot17" / "val"
    if not val.is_dir():
        trackers_cli(
            "download", "--name", "mot17", "--split", "val",
            "--asset", "frames,annotations", "--output", str(args.data),
        )  # fmt: skip
    sequences = sorted(p for p in val.iterdir() if (p / "img1").is_dir())
    if len(sequences) != 7:
        sys.exit(f"Expected the 7 MOT17 val sequences in {val}, found {len(sequences)}.")

    if args.out is not None and args.out.exists():
        sys.exit(f"--out {args.out} already exists; pick a new directory.")
    out = args.out or Path(tempfile.mkdtemp(prefix="trackflow-mot17-"))

    rows = []
    for model in args.models.split(","):
        detector = TrackflowPipeline(model=model, device=args.device).detector
        detections = {}
        for seq in sequences:
            print(f"{model}: detecting {seq.name}", file=sys.stderr)
            frames = sorted((seq / "img1").glob("*.jpg"))
            detections[seq.name] = [detector.detect(cv2.imread(str(f))) for f in frames]

        for name in args.trackers.split(","):
            tracker, _, reid = name.partition("+")
            run_dir = out / f"{model}-{name}"
            (run_dir / "mot").mkdir(parents=True)
            for seq in sequences:
                pipeline = TrackflowPipeline(
                    detector=Replay(detections[seq.name]), tracker=tracker, reid=reid or "none"
                )
                summary = pipeline.run(
                    seq / "img1",
                    output_dir=run_dir / "runs" / seq.name,
                    fps=frame_rate(seq),
                    save_video=False,
                    exports=("mot",),
                )
                shutil.copyfile(summary.output_files["mot"], run_dir / "mot" / f"{seq.name}.txt")
            trackers_cli(
                "eval", "--gt_dir", str(val), "--predictions_dir", str(run_dir / "mot"),
                "--metrics", "[CLEAR,HOTA,Identity]", "--output", str(run_dir / "scores.json"),
                quiet=True,
            )  # fmt: skip
            scores = json.loads((run_dir / "scores.json").read_text())["aggregate"]
            label = f"{tracker} --reid {reid}" if reid else tracker
            rows.append(
                f"| `{model}` | `{label}` | {100 * scores['HOTA']['HOTA']:.1f} "
                f"| {100 * scores['Identity']['IDF1']:.1f} | {100 * scores['CLEAR']['MOTA']:.1f} "
                f"| {scores['CLEAR']['IDSW']} |"
            )

    print("| Model | Tracker | HOTA | IDF1 | MOTA | ID switches |")
    print("| :--- | :--- | ---: | ---: | ---: | ---: |")
    print("\n".join(rows))
    print(f"\nMOT files and scores: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
