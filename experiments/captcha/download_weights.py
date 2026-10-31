"""Download every model weight this project uses into ./weights (the layout the code expects).

Run from the project folder with the main environment:
    .venv/bin/python download_weights.py              # models used in the paper experiments + Halligan
    .venv/bin/python download_weights.py --all        # also the optional extra presets
    .venv/bin/python download_weights.py --only owlv2 halligan-gdino
    .venv/bin/python download_weights.py --list

Hugging Face models are pinned to the exact revisions used in the experiments. Files already present
are skipped, so the script can simply be re-run after a network error.
"""
import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')  # Xet transfers stalled on our network; plain HTTP works

ROOT = Path(__file__).resolve().parent
WEIGHTS = Path(os.environ.get('CAPTCHA_WEIGHTS', ROOT / 'weights'))  # same override as solver_backends.py
HF_MAIN = WEIGHTS / 'huggingface'                 # used by captcha_solver.py (solver_backends.py)
HF_HALLIGAN = WEIGHTS / 'halligan' / 'huggingface'  # used by halligan/ (its own environment)
HF_FILES = ['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja']

# name: (kind, source, destination, extra, used for, approx size)
MODELS = {
    # --- main solvers (captcha_solver.py) ---
    'yolov8-cls': ('ultralytics', 'yolov8m-cls.pt', WEIGHTS / 'ultralytics', None,
                   'presets yolov8, yolov8-cam', '34 MB'),
    'yolo-world': ('ultralytics', 'yolov8x-worldv2.pt', WEIGHTS / 'ultralytics', None,
                   'preset yolo-world', '146 MB'),
    'yolo-world-text': ('url', 'https://openaipublic.azureedge.net/clip/models/'
                        '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt',
                        WEIGHTS / 'clip', '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af',
                        'YOLO-World text encoder', '338 MB'),
    'clip': ('hf', 'openai/clip-vit-base-patch32', HF_MAIN,
             ('3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268', HF_FILES + ['pytorch_model.bin']),
             'preset clip, captcha_clip_solver.py', '605 MB'),
    'owlv2': ('hf', 'google/owlv2-base-patch16-ensemble', HF_MAIN,
              ('cfd3195ba4ea9592eec887ded089f4c08eff231d', HF_FILES), 'preset owlv2', '620 MB'),
    'qwen2.5-vl': ('hf', 'Qwen/Qwen2.5-VL-3B-Instruct', HF_MAIN,
                   ('66285546d2b821cf421d4f5eb2576359d3770cd3', HF_FILES),
                   'presets qwen2.5-vl, qwen2.5-vl-ground', '7.5 GB'),
    'llava': ('hf', 'llava-hf/llava-1.5-7b-hf', HF_MAIN,
              ('b234b804b114d9e37bb655e11cbbb5f5e971b7a9', HF_FILES), 'presets llava, llava-ground', '14 GB'),
    # --- Halligan (halligan/, pixi environment) ---
    'halligan-fastsam': ('ultralytics', 'FastSAM-x.pt', WEIGHTS / 'halligan', None, 'Halligan segmenter', '145 MB'),
    'halligan-gdino': ('hf', 'IDEA-Research/grounding-dino-base', HF_HALLIGAN,
                       ('12bdfa3120f3e7ec7b434d90674b3396eccf88eb', HF_FILES), 'Halligan mark()/focus()', '930 MB'),
    'halligan-clip': ('hf', 'openai/clip-vit-base-patch32', HF_HALLIGAN,
                      ('3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268', HF_FILES + ['pytorch_model.bin']),
                      'Halligan get_element()', '605 MB'),
    # --- optional presets (not used in the paper experiments) ---
    'qwen2-vl': ('hf', 'Qwen/Qwen2-VL-2B-Instruct', HF_MAIN,
                 ('895c3a49bc3fa70a340399125c650a463535e71c', HF_FILES), 'preset qwen2-vl (optional)', '4.4 GB'),
    'internvl': ('hf', 'OpenGVLab/InternVL3-2B-hf', HF_MAIN,
                 ('cb57a075cb75a2e6d1b668b128d48bb00ae321d2', HF_FILES), 'preset internvl (optional)', '4.2 GB'),
    'llava-onevision': ('hf', 'llava-hf/llava-onevision-qwen2-0.5b-ov-hf', HF_MAIN,
                        ('74dd0bf867a4cda7950c17663794267c60cf4b40', HF_FILES),
                        'preset llava-onevision (optional)', '1.8 GB'),
}
OPTIONAL = {'qwen2-vl', 'internvl', 'llava-onevision'}


def download_url(url, folder, sha256):
    target = folder / url.rsplit('/', 1)[1]
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == sha256:
        return target
    folder.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + '.part')
    urllib.request.urlretrieve(url, partial)
    if hashlib.sha256(partial.read_bytes()).hexdigest() != sha256:
        partial.unlink()
        raise RuntimeError(f'Checksum mismatch for {url}')
    partial.rename(target)
    return target


def download_ultralytics(name, folder):
    from ultralytics.utils.downloads import attempt_download_asset
    folder.mkdir(parents=True, exist_ok=True)
    return Path(attempt_download_asset(str(folder / name)))


def download_hf(repo, cache_dir, revision, patterns):
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(repo, revision=revision, cache_dir=str(cache_dir), allow_patterns=patterns))
    # A pinned download is stored under its commit hash only; point 'main' at it so the code (which
    # loads the default revision) finds it, also offline.
    refs = Path(cache_dir) / ('models--' + repo.replace('/', '--')) / 'refs'
    refs.mkdir(parents=True, exist_ok=True)
    (refs / 'main').write_text(revision)
    return path


def size_of(path, follow_links=True):
    """GiB; a Hugging Face snapshot is a folder of links, the whole cache counts each real file once."""
    path = Path(path)
    files = [path] if path.is_file() else [p for p in path.rglob('*')
                                           if p.is_file() and (follow_links or not p.is_symlink())]
    return sum(f.stat().st_size for f in files) / 2**30


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--all', action='store_true', help='Also download the optional presets')
    parser.add_argument('--only', nargs='+', choices=MODELS, help='Download only these entries')
    parser.add_argument('--list', action='store_true', help='List the entries and exit')
    args = parser.parse_args()
    if args.list:
        for name, (_, source, dest, _, used, size) in MODELS.items():
            print(f'{name:18s} {size:>7s}  {source:42s} -> {dest}  ({used})')
        return
    names = args.only or [n for n in MODELS if args.all or n not in OPTIONAL]
    failed = []
    for name in names:
        kind, source, dest, extra, used, size = MODELS[name]
        print(f'[{name}] {source} (~{size}) -> {dest}', flush=True)
        try:
            if kind == 'url':
                path = download_url(source, dest, extra)
            elif kind == 'ultralytics':
                path = download_ultralytics(source, dest)
            else:
                path = download_hf(source, dest, *extra)
            print(f'    ok: {size_of(path):.2f} GiB', flush=True)
        except Exception as exc:  # keep going; report at the end
            failed.append(name)
            print(f'    FAILED: {exc}', flush=True)
    if failed:
        sys.exit(f'Failed: {", ".join(failed)}. Re-run the same command to resume.')
    print(f'Done. weights/ is now {size_of(WEIGHTS, follow_links=False):.1f} GiB.')


if __name__ == '__main__':
    main()
