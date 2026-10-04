"""Fetch the paper's checkpoint and verify its required SHA-256 before use."""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request

EXPECTED = '5a1395716f7913741cc51d98581b9b1228d80987a9f7d3664106742eb06bba83'
URL = ('https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/'
       'resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf')

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    out = args.output.expanduser().resolve()
    if out.exists():
        if digest(out) != EXPECTED:
            raise SystemExit(f'Checkpoint digest mismatch: {out}; existing file left unchanged')
        print(f'Verified {EXPECTED}  {out}')
        return
    if args.verify_only:
        raise SystemExit(f'Checkpoint not found: {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=out.parent, prefix=out.name+'.',
                                         suffix='.partial', delete=False) as tmp:
            name = Path(tmp.name)
            request = urllib.request.Request(URL, headers={'User-Agent': 'attention-replay/1'})
            with urllib.request.urlopen(request, timeout=60) as response:
                for chunk in iter(lambda: response.read(1 << 20), b''):
                    tmp.write(chunk)
        if digest(name) != EXPECTED:
            raise RuntimeError('Downloaded checkpoint does not match the paper digest')
        os.replace(name, out)
        print(f'Verified {EXPECTED}  {out}')
    finally:
        if name is not None and name.exists():
            name.unlink()

if __name__ == '__main__':
    main()
