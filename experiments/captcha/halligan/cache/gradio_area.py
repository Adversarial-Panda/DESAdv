# Cached solution for the local Gradio area-select CAPTCHA (run_gradio.py picks it for area challenges).
# Uses Halligan's POINTABLE tools, as in its pointable example: number keypoints on the canvas with
# show_keypoints(), ask GPT-4o which numbers lie on the target, point() at them, then click VERIFY.
# Like gradio_grid.py, the card is first cut into the instruction header and the canvas, because
# get_frames() returns them as one frame.
import re

import numpy as np

from halligan.utils.layout import Frame
from halligan.utils.action_tools import click, point
from halligan.utils.vision_tools import ask


def _split_card(card):
    """Header = rows that are mostly the blue instruction band; canvas = colourful block below it."""
    pixels = np.asarray(card.image.convert('RGB')).astype(int)
    r, g, b = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    blue_rows = np.nonzero(((b - r > 80) & (b - g > 40)).mean(axis=1) > 0.5)[0]
    header_bottom = int(blue_rows.max()) + 1 if blue_rows.size else card.h // 5
    below = pixels[header_bottom:]
    content = (below.max(axis=2) - below.min(axis=2)) > 30  # colourful canvas, not white/grey card
    rows, cols = np.nonzero(content.mean(axis=1) > 0.5)[0], np.nonzero(content.mean(axis=0) > 0.5)[0]
    top, bottom = header_bottom + int(rows.min()), header_bottom + int(rows.max()) + 1
    left, right = int(cols.min()), int(cols.max()) + 1
    header = Frame(card.x, card.y, card.image.crop((0, 0, card.w, header_bottom)))
    canvas = Frame(card.x + left, card.y + top, card.image.crop((left, top, right, bottom)))
    card.subframes = [header, canvas]
    return header, canvas


def stage1(frames):
    return


def stage2(frames):
    # Frame 0: page with the VERIFY button
    verify_button = frames[0].get_element(position="down", details="orange 'VERIFY' button")
    verify_button.set_element_as(interactable="NEXT")

    # Frame 1: instruction header + canvas with scattered photos
    _, canvas = _split_card(frames[1])
    canvas.set_frame_as(interactable="POINTABLE")


def stage3(frames):
    # Leaf frames after get_observation: [page with VERIFY, header, canvas]
    page, header, canvas = frames[0], frames[1], frames[2]

    target_object = "".join(ask([header.image], "What is the target object?", "str"))

    keypoint_image = canvas.show_keypoints(region="all")
    # ask() returns one answer per image, so an "int" answer can only name one keypoint; ask for a
    # string listing all of them instead (there are always several target photos).
    answer = ask([keypoint_image],
                 f"Several small photos are scattered on a noisy background, and two or three of them show "
                 f"a {target_object}. Check every photo. For EACH photo that shows a {target_object}, give "
                 f"one numbered keypoint that lies well inside that photo (not on its edge or on the "
                 f"background). Reply with the keypoint numbers separated by commas, e.g. \"3, 17\".",
                 answer_type="str")
    ids = [int(n) for n in re.findall(r"\d+", " ".join(str(a) for a in answer))]

    for keypoint_id in dict.fromkeys(ids):  # unique, in answer order
        if 0 <= keypoint_id < len(canvas.keypoints):
            point(to=canvas.get_keypoint(keypoint_id))

    click(page.get_interactable(0))
