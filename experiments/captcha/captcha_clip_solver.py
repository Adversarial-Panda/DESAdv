"""CLIP baseline for your local Gradio CAPTCHA; not hCaptcha Challenger.
Install: python -m pip install torch transformers pillow playwright
        python -m playwright install chromium
Run: python captcha_clip_solver.py --rounds 10
Attach: python captcha_clip_solver.py --cdp http://127.0.0.1:9222 --rounds 10
Default: slow visible clicks; launched browser waits for Enter at end.
Attached Chrome remains open when this script exits. No answer labels are read.
"""
import argparse
import csv
import io
import json
import os
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

# Model weights live in ./weights/huggingface (see README.md).
os.environ.setdefault('HF_HUB_CACHE', str(Path(__file__).resolve().parent / 'weights' / 'huggingface'))

CLASSES = ['goldfish', 'tiger', 'zebra', 'school bus', 'banana',
           'ant', 'pizza', 'bee', 'traffic light', 'elephant']
LOCAL_HOSTS = {'localhost', '127.0.0.1', '::1'}
READY = """() => document.querySelectorAll('.captcha-tile').length === 9 &&
    Array.from(document.querySelectorAll('.captcha-tile img')).every(
    i => i.complete && i.naturalWidth > 0)"""


def same_app(left, right):
    a, b = urlparse(left), urlparse(right)
    return (a.hostname in LOCAL_HOSTS and b.hostname in LOCAL_HOSTS
            and a.scheme == b.scheme
            and (a.port or (443 if a.scheme == 'https' else 80)) ==
                (b.port or (443 if b.scheme == 'https' else 80))
            and a.path.rstrip('/') == b.path.rstrip('/'))


def crop_box(box, grid_box, width, height, screenshot_size):
    sx, sy = screenshot_size[0] / width, screenshot_size[1] / height
    return (round((box['x'] - grid_box['x']) * sx),
            round((box['y'] - grid_box['y']) * sy),
            round((box['x'] + box['width'] - grid_box['x']) * sx),
            round((box['y'] + box['height'] - grid_box['y']) * sy))


def new_challenge(page):
    # Observe completion even in the unlikely case that the same images recur.
    page.evaluate("""() => {
        window.__solverGridChanged = false;
        const grid = document.querySelector('.captcha-grid');
        if (window.__solverObserver) window.__solverObserver.disconnect();
        window.__solverObserver = new MutationObserver(() => {
            if (!grid.isConnected || document.querySelector('.captcha-grid') !== grid) {
                window.__solverGridChanged = true;
            }
        });
        window.__solverObserver.observe(document.body, {childList:true, subtree:true});
    }""")
    old = page.locator('.captcha-grid').inner_html()
    page.locator('#new-btn').click()
    page.wait_for_function("""old => {
        const grid = document.querySelector('.captcha-grid');
        return grid && (window.__solverGridChanged || grid.innerHTML !== old)
            && !document.querySelector('.result-success, .result-error');
    }""", arg=old)
    page.evaluate("() => window.__solverObserver && window.__solverObserver.disconnect()")
    page.wait_for_function(READY)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:7860')
    parser.add_argument('--cdp', help='Attach to Chrome with remote debugging enabled')
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--model', default='openai/clip-vit-base-patch32')
    parser.add_argument('--output', default='solver_results')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--close-on-finish', action='store_true',
                        help='Close a launched browser automatically; never closes attached Chrome')
    parser.add_argument('--click-delay', type=float, default=0.6, help='Seconds between tile clicks')
    parser.add_argument('--verify-delay', type=float, default=1.0)
    parser.add_argument('--result-delay', type=float, default=2.5)
    parser.add_argument('--start-delay', type=float, default=1.5)
    parser.add_argument('--fast', action='store_true', help='Disable demonstration pauses')
    args = parser.parse_args()
    if urlparse(args.url).hostname not in LOCAL_HOSTS:
        parser.error('This adapter supports your localhost app only.')
    if args.cdp and urlparse(args.cdp).hostname not in LOCAL_HOSTS:
        parser.error('--cdp must be a localhost HTTP endpoint.')
    if args.cdp and args.headless:
        parser.error('--headless cannot change an existing Chrome window.')
    delays = [args.click_delay, args.verify_delay, args.result_delay, args.start_delay]
    if args.rounds < 1 or any(d < 0 or d > 60 for d in delays):
        parser.error('Rounds must be positive; delays must be between 0 and 60 seconds.')
    if args.fast:
        args.click_delay = args.verify_delay = args.result_delay = args.start_delay = 0

    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    from playwright.sync_api import sync_playwright

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Loading', args.model, 'on', device, flush=True)
    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model)
    output = Path(args.output) / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    fields = ['round', 'target', 'predictions', 'selected_indices', 'success',
              'seconds_with_delays', 'inference_seconds', 'model', 'device', 'error']
    successes = completed = errors = 0
    with sync_playwright() as p:
        browser = None
        try:
            if args.cdp:
                browser = p.chromium.connect_over_cdp(args.cdp)
                candidates = [tab for ctx in browser.contexts for tab in ctx.pages
                              if same_app(tab.url, args.url)]
                if len(candidates) != 1:
                    raise RuntimeError('Keep exactly one tab open at your CAPTCHA URL in debugging Chrome. '
                                       f'Found {len(candidates)} matching tabs.')
                page = candidates[0]
                print('Connected to existing CAPTCHA tab:', page.url, flush=True)
            else:
                browser = p.chromium.launch(headless=args.headless)
                page = browser.new_page(viewport={'width': 1000, 'height': 1000}, device_scale_factor=1)
                page.goto(args.url)
                print('Opened solver browser. Use --cdp to attach to debugging Chrome.', flush=True)
            page.set_default_timeout(30000)
            page.bring_to_front()
            with (output / 'results.csv').open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for round_id in range(1, args.rounds + 1):
                    row = dict.fromkeys(fields, '')
                    row.update(round=round_id, model=args.model, device=device)
                    started = time.perf_counter()
                    try:
                        page.wait_for_function(READY)
                        # Reused tabs may contain an already-submitted challenge.
                        if round_id == 1 and page.locator('.result-success, .result-error').count():
                            new_challenge(page)
                        tiles = page.locator('.captcha-tile')
                        # Clear prior manual selections using the app's normal click handler.
                        for tile in page.locator('.captcha-tile.selected').all():
                            tile.click()
                        page.wait_for_timeout(args.start_delay * 1000)
                        target = page.locator('.captcha-target').inner_text().strip().lower()
                        if target not in CLASSES:
                            raise ValueError(f'Unknown target category: {target}')
                        row['target'] = target
                        folder = output / f'round_{round_id:04d}'
                        folder.mkdir()
                        grid = page.locator('.captcha-grid')
                        grid.scroll_into_view_if_needed()
                        page.wait_for_timeout(150)  # allow selection/scroll animations to settle
                        grid_box = grid.bounding_box()
                        tile_boxes = [tiles.nth(i).bounding_box() for i in range(9)]
                        if not grid_box or any(box is None for box in tile_boxes):
                            raise RuntimeError('Could not measure all visible tiles.')
                        # One grid screenshot avoids scrolling separately to each tile.
                        data = grid.screenshot(animations='disabled')
                        (folder / 'grid.png').write_bytes(data)
                        snapshot = Image.open(io.BytesIO(data)).convert('RGB')
                        indices = [int(tiles.nth(i).get_attribute('data-index')) for i in range(9)]
                        images = []
                        for index, box in zip(indices, tile_boxes):
                            image = snapshot.crop(crop_box(box, grid_box, grid_box['width'],
                                                          grid_box['height'], snapshot.size))
                            image.save(folder / f'tile_{index}.png')
                            images.append(image)
                        inputs = processor(text=[f'a photo of {c}' for c in CLASSES],
                                           images=images, return_tensors='pt', padding=True).to(device)
                        if device == 'cuda':
                            torch.cuda.synchronize()
                        inference_start = time.perf_counter()
                        with torch.inference_mode():
                            logits = model(**inputs).logits_per_image
                        if device == 'cuda':
                            torch.cuda.synchronize()
                        row['inference_seconds'] = round(time.perf_counter() - inference_start, 4)
                        predictions = [CLASSES[i] for i in logits.argmax(dim=1).cpu().tolist()]
                        selected = [i for i, label in zip(indices, predictions) if label == target]
                        row.update(predictions=json.dumps(predictions), selected_indices=json.dumps(selected))
                        (folder / 'prediction.json').write_text(json.dumps({
                            'target': target, 'classes': CLASSES, 'tile_indices': indices,
                            'predictions': predictions, 'selected_indices': selected,
                            'logits': logits.cpu().tolist(), 'model': args.model
                        }, indent=2), encoding='utf-8')
                        print(f'Round {round_id}: {target}; selecting {selected}', flush=True)
                        for index in selected:
                            page.locator(f'.captcha-tile[data-index="{index}"]').click()
                            page.wait_for_timeout(args.click_delay * 1000)
                        # Wait until Gradio's selection textbox matches actual tile clicks.
                        page.wait_for_function("""expected => {
                            const input = document.querySelector('#selected-json textarea, #selected-json input');
                            try {return input && JSON.stringify(JSON.parse(input.value).sort((a,b)=>a-b))
                                === JSON.stringify(expected.slice().sort((a,b)=>a-b));}
                            catch(e) {return false;}
                        }""", arg=selected)
                        page.wait_for_timeout(args.verify_delay * 1000)
                        page.locator('#verify-btn').click()
                        page.locator('.result-success, .result-error').wait_for(state='visible')
                        success = page.locator('.result-success').count() > 0
                        row['success'] = success
                        successes += int(success)
                        completed += 1
                        print('  PASS' if success else '  FAIL', flush=True)
                    except Exception as exc:
                        errors += 1
                        row['error'] = str(exc)
                        print(f'Round {round_id}: ERROR: {exc}', flush=True)
                    row['seconds_with_delays'] = round(time.perf_counter() - started, 3)
                    writer.writerow(row)
                    f.flush()
                    if row['error']:
                        print('Stopped; inspect the browser and terminal error.', flush=True)
                        break
                    page.wait_for_timeout(args.result_delay * 1000)
                    if round_id < args.rounds:
                        new_challenge(page)
            print(f'Completed: {completed}; errors: {errors}; successful: {successes}', flush=True)
            if completed:
                print(f'Exact challenge success: {successes / completed:.2%}', flush=True)
        except KeyboardInterrupt:
            print('\nStopped by user.', flush=True)
        except Exception as exc:
            print(f'ERROR: {exc}', flush=True)
            if args.cdp:
                print('Start Chrome with --remote-debugging-port=9222 and a separate --user-data-dir.', flush=True)
        finally:
            print('Results:', output.resolve(), flush=True)
            if browser and not args.cdp:
                if not args.headless and not args.close_on_finish:
                    try:
                        input('Browser remains open. Press Enter here to close it...')
                    except (EOFError, KeyboardInterrupt):
                        pass
                browser.close()
            elif browser:
                print('Disconnected from attached Chrome; its window remains open.', flush=True)


if __name__ == '__main__':
    main()
