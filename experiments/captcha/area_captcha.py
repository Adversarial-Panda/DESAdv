"""Area-select CAPTCHA (hCaptcha-style point selection) for captcha_gradio_app.py --type area.

One canvas: a noisy, blurred background with several dataset images scattered at random,
non-overlapping positions. The user clicks a point on every image of the target class.
The answer key (each image's box) stays in server state; the browser only gets the
rendered canvas as one JPEG.
"""
import base64
import io
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CANVAS_SIZE = (600, 450)      # rendered canvas (width, height); displayed scaled to the card width
OBJECTS = (6, 8)              # images scattered per challenge (min, max)
TARGETS = (2, 3)              # images of the target class (min, max)
OBJECT_SIZE = (90, 140)       # side length of each scattered image in canvas pixels
GAP = 10                      # minimum spacing between images
CLUTTER_SHAPES = 60           # random lines/ellipses drawn into the background
NOISE_STD = 14                # Gaussian pixel noise over the whole canvas (objects included)
FEATHER = 6                   # px of soft edge blending each image into the background


def _background(rng):
    width, height = CANVAS_SIZE
    # Low-frequency colour blobs: a tiny random image upscaled smoothly.
    blobs = np.array([[[rng.randrange(256) for _ in range(3)] for _ in range(8)] for _ in range(6)], np.uint8)
    canvas = Image.fromarray(blobs).resize(CANVAS_SIZE, Image.BICUBIC)
    draw = ImageDraw.Draw(canvas)
    for _ in range(CLUTTER_SHAPES):
        colour = tuple(rng.randrange(256) for _ in range(3))
        x0, y0 = rng.randrange(width), rng.randrange(height)
        x1, y1 = x0 + rng.randint(-120, 120), y0 + rng.randint(-120, 120)
        if rng.random() < 0.5:
            draw.line((x0, y0, x1, y1), fill=colour, width=rng.randint(1, 6))
        else:
            draw.ellipse((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)),
                         outline=colour, width=rng.randint(1, 4))
    return canvas.filter(ImageFilter.GaussianBlur(3))


def _place(rng, sizes):
    """Random non-overlapping top-left corners for squares of the given sizes."""
    width, height = CANVAS_SIZE
    for _ in range(200):
        boxes = []
        for size in sizes:
            for _ in range(300):
                x, y = rng.randint(0, width - size), rng.randint(0, height - size)
                box = (x, y, x + size, y + size)
                if all(box[2] + GAP <= b[0] or b[2] + GAP <= box[0] or
                       box[3] + GAP <= b[1] or b[3] + GAP <= box[1] for b in boxes):
                    boxes.append(box)
                    break
            else:
                break
        if len(boxes) == len(sizes):
            return boxes
    raise RuntimeError('Could not place images; reduce OBJECTS or OBJECT_SIZE.')


def _square(path, size):
    image = Image.open(path).convert('RGB')
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    return image.crop((left, top, left + side, top + side)).resize((size, size), Image.LANCZOS)


def _feather_mask(size):
    mask = Image.new('L', (size, size), 0)
    ImageDraw.Draw(mask).rectangle((FEATHER, FEATHER, size - FEATHER, size - FEATHER), fill=255)
    return mask.filter(ImageFilter.GaussianBlur(FEATHER / 2))


def make_area_challenge(df, rng=random):
    """Returns (target_label, canvas data URI, objects) where objects carry the answer boxes."""
    target = rng.choice(sorted(df['label'].unique()))
    positives, negatives = df[df['label'] == target], df[df['label'] != target]
    n_objects = rng.randint(*OBJECTS)
    n_targets = rng.randint(TARGETS[0], min(TARGETS[1], len(positives), n_objects - 1))
    seed = rng.randrange(2**32)
    chosen = ([(row, True) for _, row in positives.sample(n=n_targets, random_state=seed).iterrows()] +
              [(row, False) for _, row in negatives.sample(n=n_objects - n_targets, random_state=seed).iterrows()])
    rng.shuffle(chosen)

    sizes = [rng.randint(*OBJECT_SIZE) for _ in chosen]
    boxes = _place(rng, sizes)
    canvas = _background(rng)
    objects = []
    for (row, is_target), size, box in zip(chosen, sizes, boxes):
        canvas.paste(_square(row['image_path'], size), box[:2], _feather_mask(size))
        objects.append({'image_name': row['image_name'], 'label': row['label'],
                        'is_target': is_target, 'box': list(box)})

    pixels = np.asarray(canvas, np.float32)
    pixels += np.random.default_rng(seed).normal(0, NOISE_STD, pixels.shape)
    canvas = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))

    buffer = io.BytesIO()
    canvas.save(buffer, format='JPEG', quality=90)
    data_uri = 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')
    return target, data_uri, objects


def check_clicks(clicks, objects):
    """Correct when every target image got a click and no click landed elsewhere."""
    hit = set()
    for x, y in clicks:
        inside = [i for i, o in enumerate(objects)
                  if o['box'][0] <= x <= o['box'][2] and o['box'][1] <= y <= o['box'][3]]
        if not inside or not objects[inside[0]]['is_target']:
            return False
        hit.add(inside[0])
    return hit == {i for i, o in enumerate(objects) if o['is_target']}
