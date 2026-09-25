"""Print the descriptors of an audio file as JSON.

Usage (from backend/):  python -m app.extraction path/to/file.mp3
"""

import argparse
import json
import sys

from app.core.audio import load_audio
from app.core.errors import AudioError
from app.extraction import extract_features


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m app.extraction", description=__doc__.splitlines()[0])
    parser.add_argument("path", help="audio file (mp3, wav, flac, ogg, m4a)")
    parser.add_argument("--no-model-input", action="store_true", help="omit the model input vector")
    args = parser.parse_args()

    try:
        audio, sr = load_audio(args.path)
        result = extract_features(audio, sr)
    except AudioError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.no_model_input:
        del result["model_input"]
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
