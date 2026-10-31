"""Model-based solver for your local Gradio CAPTCHA (YOLOv8, Qwen2-VL, Qwen2.5-VL, LLaVA, InternVL).
Install: python -m pip install ultralytics transformers accelerate bitsandbytes pillow playwright
        python -m playwright install chromium
Run: python captcha_solver.py --model yolov8 --tag benign --rounds 100 --fast
     python captcha_solver.py --model qwen2.5-vl --tag adv --rounds 100 --fast
Presets: see solver_backends.PRESETS; any YOLO .pt path or Hugging Face model id also works.
Attach: python captcha_solver.py --cdp http://127.0.0.1:9222 --rounds 10
Default: slow visible clicks; launched browser waits for Enter at end.
Attached Chrome remains open when this script exits. No answer labels are read.
"""
import argparse
import csv
import io
import json
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from solver_backends import CLASSES, PRESETS, load_backend, resolve

LOCAL_HOSTS = {'localhost', '127.0.0.1', '::1'}
# A grid challenge (9 tiles) or an area challenge (one canvas image) whose images have loaded.
READY = """() => {
    const loaded = imgs => imgs.length > 0 && imgs.every(i => i.complete && i.naturalWidth > 0);
    const tiles = Array.from(document.querySelectorAll('.captcha-tile img'));
    const canvas = Array.from(document.querySelectorAll('.area-canvas img'));
    return (tiles.length === 9 && loaded(tiles)) || loaded(canvas);
}"""


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
        const grid = document.querySelector('.captcha-card');
        if (window.__solverObserver) window.__solverObserver.disconnect();
        window.__solverObserver = new MutationObserver(() => {
            if (!grid.isConnected || document.querySelector('.captcha-card') !== grid) {
                window.__solverGridChanged = true;
            }
        });
        window.__solverObserver.observe(document.body, {childList:true, subtree:true});
    }""")
    old = page.locator('.captcha-card').inner_html()
    page.locator('#new-btn').click()
    page.wait_for_function("""old => {
        const grid = document.querySelector('.captcha-card');
        return grid && (window.__solverGridChanged || grid.innerHTML !== old)
            && !document.querySelector('.result-success, .result-error');
    }""", arg=old)
    page.evaluate("() => window.__solverObserver && window.__solverObserver.disconnect()")
    page.wait_for_function(READY)


def save_detections(image, details, points, path):
    """Yellow: boxes accepted as the target; red circles: click points."""
    from PIL import ImageDraw
    heatmap = details.pop('heatmap', None)
    if heatmap is not None:
        draw = ImageDraw.Draw(heatmap)
        for x, y in points:
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline='white', width=3)
        heatmap.save(path.with_name('heatmap.png'))
    debug = image.copy()
    draw = ImageDraw.Draw(debug)
    for kept in details['kept_boxes']:
        draw.rectangle(kept, outline='yellow', width=2)
    for x, y in points:
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline='red', width=3)
    debug.save(path)


def solve_area(page, backend, target, folder, row, args, device):
    """Area challenge: screenshot the canvas, let the backend locate the target, click each point."""
    import torch
    from PIL import Image, ImageDraw
    if not hasattr(backend, 'locate'):
        raise RuntimeError(f'{row["model"]} does not support area challenges yet; use --model clip or yolo-world.')
    canvas = page.locator('.area-canvas img')
    canvas.scroll_into_view_if_needed()
    page.wait_for_timeout(150)
    box = canvas.bounding_box()
    if getattr(backend, 'full_resolution', False):
        # The same picture the browser displays, at its natural size instead of the scaled-down render.
        import base64
        src = canvas.get_attribute('src')
        data = base64.b64decode(src.split(',', 1)[1])
    else:
        data = canvas.screenshot(animations='disabled')
    (folder / 'canvas.png').write_bytes(data)
    image = Image.open(io.BytesIO(data)).convert('RGB')
    if device == 'cuda':
        torch.cuda.synchronize()
    inference_start = time.perf_counter()
    points, details = backend.locate(image, target)
    if device == 'cuda':
        torch.cuda.synchronize()
    row['inference_seconds'] = round(time.perf_counter() - inference_start, 4)

    save_detections(image, details, points, folder / 'detections.png')
    (folder / 'prediction.json').write_text(json.dumps({
        'target': target, 'points_screenshot_px': points, **details,
        'model': row['model'], 'tag': args.tag}, indent=2), encoding='utf-8')

    print(f'Round {row["round"]}: {target}; clicking {len(points)} point(s) {points}', flush=True)
    sx, sy = box['width'] / image.width, box['height'] / image.height
    for x, y in points:
        page.mouse.click(box['x'] + x * sx, box['y'] + y * sy)
        page.wait_for_timeout(args.click_delay * 1000)
    # Wait until the app registered every click (markers within 14 px of each other would toggle).
    page.wait_for_function("""n => {
        const input = document.querySelector('#selected-json textarea, #selected-json input');
        try {return input && JSON.parse(input.value).length === n;} catch(e) {return false;}
    }""", arg=len(points))
    return points, [target] * len(points)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:7860')
    parser.add_argument('--cdp', help='Attach to Chrome with remote debugging enabled')
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--model', default='yolov8',
                        help=f'Preset ({", ".join(PRESETS)}), YOLO .pt weights or a Hugging Face model id')
    parser.add_argument('--imgsz', type=int, default=224, help='YOLO input size')
    parser.add_argument('--image-size', type=int, default=448, help='Tile size given to VLMs')
    parser.add_argument('--batch-size', type=int, default=9, help='Tiles per VLM generate call; lower on OOM')
    quant = parser.add_mutually_exclusive_group()
    quant.add_argument('--4bit', dest='load_4bit', action='store_true', default=None,
                       help='Force 4-bit loading of a VLM (default: per preset)')
    quant.add_argument('--no-4bit', dest='load_4bit', action='store_false')
    parser.add_argument('--tag', default='',
                        help='Label for this run, e.g. benign or adv; added to the output folder name')
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
    from playwright.sync_api import sync_playwright

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    short, backend_name, model_id, load_4bit = resolve(args.model, args.load_4bit)
    print(f'Loading {model_id} ({backend_name}{", 4-bit" if load_4bit else ""}) on {device}', flush=True)
    # Distinguish solving methods that share weights (e.g. YOLOv8-cls tiles vs class activation maps).
    model_label = model_id + {'yolo-cam': ' (CAM)', 'vlm-ground': ' (grounding)'}.get(backend_name, '')
    backend = load_backend(backend_name, model_id, device, imgsz=args.imgsz, load_4bit=load_4bit,
                           image_size=args.image_size, batch_size=args.batch_size)
    run_name = '_'.join(filter(None, [short, args.tag, datetime.now().strftime('%Y%m%d_%H%M%S_%f')]))
    output = Path(args.output) / run_name
    output.mkdir(parents=True, exist_ok=False)
    fields = ['round', 'tag', 'captcha_type', 'challenge_id', 'target', 'predictions', 'selected_indices', 'success',
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
                    row.update(round=round_id, tag=args.tag, model=model_label, device=device)
                    started = time.perf_counter()
                    try:
                        page.wait_for_function(READY)
                        # Reused tabs may contain an already-submitted challenge.
                        if round_id == 1 and page.locator('.result-success, .result-error').count():
                            new_challenge(page)
                        is_area = page.locator('.area-canvas').count() > 0
                        row['captcha_type'] = 'area' if is_area else 'grid'
                        tiles = page.locator('.captcha-tile')
                        # Clear prior manual selections using the app's normal click handler.
                        for tile in page.locator('.captcha-tile.selected').all():
                            tile.click()
                        page.wait_for_timeout(args.start_delay * 1000)
                        target = page.locator('.captcha-target').inner_text().strip().lower()
                        if target not in CLASSES:
                            raise ValueError(f'Unknown target category: {target}')
                        row['target'] = target
                        # Lets compare_results.py join this round with the app's log; not an answer.
                        row['challenge_id'] = page.locator('.captcha-card').get_attribute(
                            'data-challenge-id') or ''
                        folder = output / f'round_{round_id:04d}'
                        folder.mkdir()
                        if is_area:
                            selected, predictions = solve_area(page, backend, target, folder, row, args, device)
                            row.update(predictions=json.dumps(predictions), selected_indices=json.dumps(selected))
                        else:
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
                            if device == 'cuda':
                                torch.cuda.synchronize()
                            inference_start = time.perf_counter()
                            if hasattr(backend, 'predict') and not getattr(backend, 'whole_image', False):
                                predictions, details = backend.predict(images)
                            else:
                                # Detector on the whole grid screenshot: a tile is chosen when the centre
                                # of a target box lands inside it.
                                points, details = backend.locate(snapshot, target)
                                tiles_px = [crop_box(box, grid_box, grid_box['width'], grid_box['height'],
                                                     snapshot.size) for box in tile_boxes]
                                hits = {i for x, y in points for i, b in enumerate(tiles_px)
                                        if b[0] <= x <= b[2] and b[1] <= y <= b[3]}
                                predictions = [target if i in hits else 'other' for i in range(9)]
                                details = {**details, 'points_screenshot_px': points}
                                save_detections(snapshot, details, points, folder / 'detections.png')
                            if device == 'cuda':
                                torch.cuda.synchronize()
                            row['inference_seconds'] = round(time.perf_counter() - inference_start, 4)
                            selected = [i for i, label in zip(indices, predictions) if label == target]
                            row.update(predictions=json.dumps(predictions), selected_indices=json.dumps(selected))
                            (folder / 'prediction.json').write_text(json.dumps({
                                'target': target, 'classes': CLASSES, 'tile_indices': indices,
                                'predictions': predictions, 'selected_indices': selected,
                                **details, 'model': model_id, 'tag': args.tag
                            }, indent=2), encoding='utf-8')
                            print(f'Round {round_id}: {target}; selecting {selected}; predictions {predictions}', flush=True)
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
