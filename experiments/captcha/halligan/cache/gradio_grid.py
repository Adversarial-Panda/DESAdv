# Cached solution for the local Gradio 3x3 CAPTCHA (used by run_gradio.py --mode cached).
# Derived from GPT-4o's own full-pipeline output for this CAPTCHA (stage 2: VERIFY button as NEXT,
# 3x3 SELECTABLE choices; stage 3: ask target, mark + ask per tile, select, click VERIFY), with two fixes:
#   1. get_frames() returns header + grid as one frame, so the card is cut into a header subframe and a
#      square grid subframe before tiling (otherwise the top row of "tiles" lands on the header).
#   2. mark() runs GroundingDINO one tile at a time; all 9 at once runs out of memory on an 8 GB GPU.
import numpy as np

from halligan.utils.layout import Frame
from halligan.utils.action_tools import select, click
from halligan.utils.vision_tools import ask, mark


def _split_card(card):
    """Header = rows of the card that are mostly the blue instruction band; grid = square below it."""
    pixels = np.asarray(card.image.convert('RGB')).astype(int)
    r, g, b = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    blue_rows = np.nonzero(((b - r > 80) & (b - g > 40)).mean(axis=1) > 0.5)[0]
    header_bottom = int(blue_rows.max()) + 1 if blue_rows.size else card.h // 5
    grid_top = header_bottom + 1
    while grid_top < card.h and pixels[grid_top].std() < 8:  # skip the plain gap under the header
        grid_top += 1
    # Square grid centred horizontally: card.w = side + 2*margin, card.h = grid_top + side + margin.
    margin = max(0, card.w - (card.h - grid_top))
    side = card.w - 2 * margin
    header = Frame(card.x, card.y, card.image.crop((0, 0, card.w, header_bottom)))
    grid = Frame(card.x + margin, card.y + grid_top,
                 card.image.crop((margin, grid_top, margin + side, grid_top + side)))
    card.subframes = [header, grid]
    return header, grid


def stage1(frames):
    return


def stage2(frames):
    # Frame 0: page with the VERIFY button
    verify_button = frames[0].get_element(position="down", details="orange 'VERIFY' button")
    verify_button.set_element_as(interactable="NEXT")

    # Frame 1: instruction header + 3x3 grid of images
    _, grid = _split_card(frames[1])
    for row in grid.grid(tiles=9):
        for element in row:
            element.set_element_as(interactable="SELECTABLE")


def stage3(frames):
    # Leaf frames after get_observation: [page with VERIFY, header, grid]
    page, header, grid = frames[0], frames[1], frames[2]

    target_object = "".join(ask([header.image], "What is the target object?", "str"))

    tiles = [grid.get_interactable(i) for i in range(9)]
    marked_images = [mark([tile.image], target_object)[0] for tile in tiles]
    answers = ask(marked_images, f"Does this image contain a {target_object}?", answer_type="bool")

    for tile, contains in zip(tiles, answers):
        if contains:
            select(tile)

    click(page.get_interactable(0))
