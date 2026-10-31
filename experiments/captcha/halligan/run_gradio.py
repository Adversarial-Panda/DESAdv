"""Run Halligan (GPT-4o) on the local Gradio CAPTCHA and log results like captcha_solver.py.

Halligan only sees a screenshot of the CAPTCHA region and acts with mouse clicks
(its own vision/action tools); no DOM, labels or filenames are given to it. The harness
uses the DOM only to find the region, start new challenges and read the verdict.

Setup:  put OPENAI_API_KEY=sk-... in halligan/.env
Run (from the halligan folder, with the app running):
    ~/.pixi/bin/pixi run -e cuda python run_gradio.py --tag benign --rounds 100

--mode cached (default): Halligan's evaluation mode (execute.py); runs the solution script in
    cache/gradio_grid.py (3x3 grid) or cache/gradio_area.py (area select), chosen per round from the
    page; both call GPT-4o through Halligan's ask() tool every round.
--mode full: Halligan's full pipeline (objective identification, structure abstraction and
    solution composition by GPT-4o) every round; several times slower and more expensive.
"""
import argparse
import csv
import importlib.util
import json
import os
import sys
import time
import traceback
from datetime import datetime
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
load_dotenv(BASE / '.env')  # must happen before Halligan modules create their GPT agent
sys.path.insert(0, str(ROOT))

from captcha_solver import LOCAL_HOSTS, READY, new_challenge  # noqa: E402  (shared browser helpers)

# USD per 1M tokens for gpt-4o-2024-11-20; used only for the running cost estimate.
PRICES = {'input': 2.50, 'output': 10.00}
HIDE_NEW_BUTTON = "() => { const b = document.querySelector('#new-btn'); if (b) b.style.visibility = '%s'; }"


class TokenCounter:
    """Counts OpenAI usage for every GPT-4o call Halligan makes (agent and vision tools)."""
    prompt = completion = calls = 0

    @classmethod
    def install(cls):
        from openai.resources.chat.completions import Completions
        original = Completions.create

        def create(self, *args, **kwargs):
            response = original(self, *args, **kwargs)
            if response.usage:
                cls.prompt += response.usage.prompt_tokens
                cls.completion += response.usage.completion_tokens
            cls.calls += 1
            return response
        Completions.create = create

    @classmethod
    def cost(cls, prompt, completion):
        return (prompt * PRICES['input'] + completion * PRICES['output']) / 1e6


def captcha_region(page):
    """CAPTCHA card plus the VERIFY button, as one screen rectangle (Halligan's input)."""
    card = page.locator('.captcha-card').bounding_box()
    verify = page.locator('#verify-btn').bounding_box()
    x0, y0 = min(card['x'], verify['x']), card['y']
    x1 = max(card['x'] + card['width'], verify['x'] + verify['width'])
    y1 = verify['y'] + verify['height']
    pad = 8
    return {'x': x0 - pad, 'y': y0 - pad, 'width': x1 - x0 + 2 * pad, 'height': y1 - y0 + 2 * pad}


def load_cache(path):
    spec = importlib.util.spec_from_file_location('gradio_cache', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--url', default='http://127.0.0.1:7860')
    parser.add_argument('--tag', default='', help='benign, adv or adv2')
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--mode', choices=['cached', 'full'], default='cached')
    parser.add_argument('--cache', default=None,
                        help='Solution script to use for every round (default: grid or area script per page)')
    parser.add_argument('--model', default='gpt-4o-2024-11-20')
    parser.add_argument('--output', default=str(ROOT / 'solver_results'))
    parser.add_argument('--show', action='store_true',
                        help='Show the browser window (keep it visible: a hidden/minimised window stops rendering)')
    # Same demonstration pacing as captcha_solver.py; applied with --show, all 0 when headless.
    parser.add_argument('--click-delay', type=float, default=None, help='Seconds after each click (default 0.6 with --show)')
    parser.add_argument('--start-delay', type=float, default=None, help='Seconds before each round (default 1.5 with --show)')
    parser.add_argument('--result-delay', type=float, default=None, help='Seconds to show each verdict (default 2.5 with --show)')
    parser.add_argument('--max-cost', type=float, default=None,
                        help='Stop before a round once the estimated USD cost reaches this value')
    args = parser.parse_args()
    for name, shown in [('click_delay', 0.6), ('start_delay', 1.5), ('result_delay', 2.5)]:
        if getattr(args, name) is None:
            setattr(args, name, shown if args.show else 0)
    from urllib.parse import urlparse
    if urlparse(args.url).hostname not in LOCAL_HOSTS:
        parser.error('This adapter supports your localhost app only.')
    if not os.getenv('OPENAI_API_KEY'):
        parser.error(f'OPENAI_API_KEY is not set; add it to {BASE / ".env"}')

    from PIL import Image
    from playwright.sync_api import sync_playwright
    import halligan.utils.action_tools as action_tools
    import halligan.utils.vision_tools as vision_tools
    from halligan.agents import GPTAgent
    from halligan.utils.logger import Trace
    from halligan.utils.layout import get_frames, get_observation
    from halligan.stages.stage1 import objective_identification
    from halligan.stages.stage2 import structure_abstraction
    from halligan.stages.stage3 import solution_composition

    TokenCounter.install()
    vision_tools.agent.model = args.model  # the agent behind ask()/compare()/rank()
    caches = {}
    if args.mode == 'cached':
        caches = ({'grid': load_cache(args.cache), 'area': load_cache(args.cache)} if args.cache else
                  {'grid': load_cache(BASE / 'cache' / 'gradio_grid.py'),
                   'area': load_cache(BASE / 'cache' / 'gradio_area.py')})
    model_name = f'Halligan ({args.model}, {args.mode})'
    run = '_'.join(filter(None, ['halligan', args.mode, args.tag, datetime.now().strftime('%Y%m%d_%H%M%S_%f')]))
    output = Path(args.output) / run
    output.mkdir(parents=True)
    fields = ['round', 'tag', 'captcha_type', 'challenge_id', 'target', 'predictions', 'selected_indices', 'success',
              'seconds_with_delays', 'inference_seconds', 'model', 'device', 'error',
              'gpt_calls', 'prompt_tokens', 'completion_tokens', 'cost_usd']
    successes = completed = consecutive_errors = 0
    print(f'{model_name}; results in {output}', flush=True)

    with sync_playwright() as p, (output / 'results.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        # Halligan's benchmark uses a 1344x768 viewport; taller here so the whole card fits.
        # Headless by default, as in Halligan's benchmark. A visible window that is covered or minimised
        # stops producing frames and screenshots time out; these flags reduce that, but keep it in view.
        browser = p.chromium.launch(headless=not args.show, args=[
            '--disable-backgrounding-occluded-windows', '--disable-renderer-backgrounding',
            '--disable-background-timer-throttling'])
        page = browser.new_page(viewport={'width': 1344, 'height': 1000}, device_scale_factor=1)
        page.goto(args.url)
        page.set_default_timeout(30000)
        action_tools.set_page(page)
        if args.click_delay:
            # Halligan clicks via page.mouse.click; pause after each so tile selection is visible.
            mouse_click = page.mouse.click

            def paced_click(*a, **kw):
                mouse_click(*a, **kw)
                page.wait_for_timeout(args.click_delay * 1000)
            page.mouse.click = paced_click
        try:
            for round_id in range(1, args.rounds + 1):
                spent = TokenCounter.cost(TokenCounter.prompt, TokenCounter.completion)
                if args.max_cost is not None and spent >= args.max_cost:
                    print(f'Stopping: estimated cost ${spent:.2f} reached --max-cost', flush=True)
                    break
                row = dict.fromkeys(fields, '')
                row.update(round=round_id, tag=args.tag, model=model_name, device='openai-api')
                started = time.perf_counter()
                before = (TokenCounter.calls, TokenCounter.prompt, TokenCounter.completion)
                folder = output / f'round_{round_id:04d}'
                folder.mkdir()
                tracing = False
                try:
                    page.wait_for_function(READY)
                    if round_id == 1 and page.locator('.result-success, .result-error').count():
                        new_challenge(page)
                    # Evaluation bookkeeping only; none of this is passed to Halligan.
                    row['target'] = page.locator('.captcha-target').inner_text().strip().lower()
                    row['challenge_id'] = page.locator('.captcha-card').get_attribute('data-challenge-id') or ''
                    page.wait_for_timeout(args.start_delay * 1000)
                    page.evaluate(HIDE_NEW_BUTTON % 'hidden')  # keep the harness-only button out of reach
                    page.locator('.captcha-card').scroll_into_view_if_needed()
                    page.wait_for_timeout(300)
                    region = captcha_region(page)
                    captcha = Image.open(BytesIO(page.screenshot(clip=region))).convert('RGB')
                    captcha.save(folder / 'captcha.png')

                    Trace.start(captcha, str(folder / 'trace.ipynb'))
                    tracing = True
                    solve_start = time.perf_counter()
                    frames = get_frames(region['x'], region['y'], captcha)
                    captcha_type = 'area' if page.locator('.area-canvas').count() else 'grid'
                    row['captcha_type'] = captcha_type
                    cache = caches.get(captcha_type)
                    if cache:
                        cache.stage2(frames)
                        leaf_frames, *_ = get_observation(frames)
                        cache.stage3(leaf_frames)
                    else:
                        agent = GPTAgent(api_key=os.getenv('OPENAI_API_KEY'), model=args.model)
                        objective = objective_identification(agent, frames)
                        structure_abstraction(agent, frames, objective)
                        solution_composition(agent, frames, objective)
                    row['inference_seconds'] = round(time.perf_counter() - solve_start, 3)

                    result = page.locator('.result-success, .result-error')
                    try:
                        result.wait_for(state='visible', timeout=15000)
                        success = page.locator('.result-success').count() > 0
                    except Exception:
                        success = False  # Halligan never pressed VERIFY: counts as a failed challenge
                        print('  Halligan did not submit', flush=True)
                    if captcha_type == 'area':  # canvas pixel coordinates of the registered clicks
                        selected = json.loads(page.locator('#selected-json textarea, #selected-json input')
                                              .first.input_value() or '[]')
                    else:
                        selected = sorted(int(t.get_attribute('data-index'))
                                          for t in page.locator('.captcha-tile.selected').all())
                    row.update(selected_indices=json.dumps(selected), success=success)
                    successes += int(success)
                    completed += 1
                    print(f'Round {round_id}: {row["target"]}; selected {selected}; '
                          f'{"PASS" if success else "FAIL"}', flush=True)
                except Exception as exc:
                    row['error'] = str(exc)
                    (folder / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
                    print(f'Round {round_id}: ERROR: {exc}', flush=True)
                finally:
                    if tracing:
                        Trace.stop()
                    page.evaluate(HIDE_NEW_BUTTON % 'visible')

                calls = TokenCounter.calls - before[0]
                prompt = TokenCounter.prompt - before[1]
                completion = TokenCounter.completion - before[2]
                row.update(gpt_calls=calls, prompt_tokens=prompt, completion_tokens=completion,
                           cost_usd=round(TokenCounter.cost(prompt, completion), 5),
                           seconds_with_delays=round(time.perf_counter() - started, 3))
                writer.writerow(row)
                f.flush()
                consecutive_errors = consecutive_errors + 1 if row['error'] else 0
                if consecutive_errors >= 3:
                    print('Stopping after 3 errors in a row; see error.txt in the round folders.', flush=True)
                    break
                page.wait_for_timeout(args.result_delay * 1000)
                if round_id < args.rounds:
                    try:
                        new_challenge(page)
                    except Exception as exc:
                        print(f'Could not start a new challenge: {exc}', flush=True)
                        break
        except KeyboardInterrupt:
            print('\nStopped by user.', flush=True)
        finally:
            browser.close()

    total = TokenCounter.cost(TokenCounter.prompt, TokenCounter.completion)
    print(f'Completed: {completed}; successful: {successes}'
          + (f' ({successes / completed:.2%})' if completed else ''), flush=True)
    print(f'GPT calls: {TokenCounter.calls}; estimated cost: ${total:.2f}', flush=True)
    print('Results:', output, flush=True)


if __name__ == '__main__':
    main()
