import argparse
import os
import random
import json
from datetime import datetime

import pandas as pd
import gradio as gr

from area_captcha import make_area_challenge, check_clicks


# ============================================================
# CONFIG
# ============================================================

IMAGE_DIRS = {
    "benign": "dataset/captcha_images",
    "adv": "dataset/captcha_images_adv",
    "adv2": "dataset/CAPTCHA_ADV_IMAGES_2"
}

parser = argparse.ArgumentParser()
parser.add_argument("--images", choices=IMAGE_DIRS, default="benign",
                    help="Image set used to build challenges")
parser.add_argument("--type", choices=["grid", "area"], default="grid",
                    help="grid: 3x3 image selection; area: click points on images scattered on a noisy canvas")
parser.add_argument("--port", type=int, default=7860)
cli_args = parser.parse_args()

IMAGE_DIR = IMAGE_DIRS[cli_args.images]
LABEL_CSV = "dataset/captcha_images_labels.csv"
# Grid results keep their original file names (used by compare_results.py).
RESULTS_CSV = (f"dataset/captcha_results_{cli_args.images}.csv" if cli_args.type == "grid"
               else f"dataset/captcha_results_area_{cli_args.images}.csv")

GRID_SIZE = 9
MIN_POSITIVES = 2
MAX_POSITIVES = 4


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(LABEL_CSV)

required_cols = {"image_name", "label"}

if not required_cols.issubset(df.columns):
    raise ValueError(
        f"CSV must contain columns {required_cols}. "
        f"Found: {list(df.columns)}"
    )

df["image_path"] = df["image_name"].apply(
    lambda x: os.path.join(IMAGE_DIR, str(x))
)

# keep only existing images
df = df[
    df["image_path"].apply(os.path.exists)
].copy()

df.reset_index(drop=True, inplace=True)

if df.empty:
    raise ValueError(
        "No valid images found. "
        "Check IMAGE_DIR and LABEL_CSV."
    )


# only keep labels with enough positive images
label_counts = df["label"].value_counts()

valid_labels = label_counts[
    label_counts >= MIN_POSITIVES
].index.tolist()

df = df[
    df["label"].isin(valid_labels)
].copy()

df.reset_index(drop=True, inplace=True)

if len(valid_labels) < 2:
    raise ValueError(
        "Need at least two valid classes."
    )


print("\n==============================")
print("CAPTCHA DATASET")
print("==============================")

print("Image set:", cli_args.images, "->", IMAGE_DIR)
print("Images:", len(df))
print("Classes:", len(valid_labels))

print("\nClass distribution:")
print(df["label"].value_counts())


# ============================================================
# CREATE CAPTCHA
# ============================================================

def make_challenge():

    if cli_args.type == "area":
        return make_area()

    target_label = random.choice(
        df["label"].unique().tolist()
    )

    positives = df[
        df["label"] == target_label
    ]

    negatives = df[
        df["label"] != target_label
    ]

    max_possible_pos = min(
        MAX_POSITIVES,
        len(positives)
    )

    n_pos = random.randint(
        MIN_POSITIVES,
        max_possible_pos
    )

    n_neg = GRID_SIZE - n_pos

    if len(negatives) < n_neg:
        raise ValueError(
            "Not enough negative images."
        )

    pos = positives.sample(
        n=n_pos,
        replace=False
    )

    neg = negatives.sample(
        n=n_neg,
        replace=False
    )

    challenge = pd.concat(
        [pos, neg]
    )

    challenge = challenge.sample(
        frac=1
    ).reset_index(drop=True)


    # ========================================================
    # CREATE STATE
    # ========================================================

    samples = []
    correct_indices = []

    for i, row in challenge.iterrows():

        samples.append({
            "image_name": row["image_name"],
            "label": row["label"],
            "image_path": row["image_path"]
        })

        if row["label"] == target_label:
            correct_indices.append(i)


    challenge_id = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    state = {
        "challenge_id": challenge_id,
        "target_label": target_label,
        "correct_indices": correct_indices,
        "samples": samples,
        "created_at": datetime.now().isoformat()
    }


    # ========================================================
    # BUILD HTML GRID
    # ========================================================

    tiles_html = ""

    for i, sample in enumerate(samples):

        image_path = os.path.abspath(
            sample["image_path"]
        ).replace("\\", "/")

        # Gradio serves allowed local files through this route
        src = f"/gradio_api/file={image_path}"

        tiles_html += f"""
        <div
            class="captcha-tile"
            data-index="{i}"
            onclick="toggleCaptchaTile(this)"
        >

            <img
                src="{src}"
                draggable="false"
            >

            <div class="captcha-check">
                ✓
            </div>

        </div>
        """


    html = f"""
    <div class="captcha-card" data-challenge-id="{challenge_id}">

        <div class="captcha-header">

            <div class="captcha-instruction">
                Select all images containing
            </div>

            <div class="captcha-target">
                {target_label.upper()}
            </div>

            <div class="captcha-footer-text">
                Click VERIFY when finished
            </div>

        </div>

        <div class="captcha-grid">

            {tiles_html}

        </div>

    </div>
    """

    return (
        html,
        state,
        "",
        "[]"
    )


# ============================================================
# VERIFY CAPTCHA
# ============================================================

# ============================================================
# AREA-SELECT CAPTCHA
# ============================================================

def make_area():

    target_label, canvas_uri, objects = make_area_challenge(df)

    challenge_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    state = {
        "type": "area",
        "challenge_id": challenge_id,
        "target_label": target_label,
        "objects": objects,
        "created_at": datetime.now().isoformat()
    }

    # Only the rendered canvas reaches the browser; object boxes stay in state.
    html = f"""
    <div class="captcha-card area-card" data-challenge-id="{challenge_id}">

        <div class="captcha-header">

            <div class="captcha-instruction">
                Click on every image containing
            </div>

            <div class="captcha-target">
                {target_label.upper()}
            </div>

            <div class="captcha-footer-text">
                Click a marker again to remove it, then click VERIFY
            </div>

        </div>

        <div class="area-canvas" onclick="areaClick(event, this)">

            <img src="{canvas_uri}" draggable="false">

        </div>

    </div>
    """

    return html, state, "", "[]"


def verify_area(selected_json, state):

    try:
        clicks = [(float(x), float(y)) for x, y in json.loads(selected_json or "[]")]
    except Exception as e:
        print("JSON ERROR:", e)
        clicks = []

    is_correct = check_clicks(clicks, state["objects"])

    print("Clicks :", clicks)
    print("Targets:", [o["box"] for o in state["objects"] if o["is_target"]])
    print("Correct answer?", is_correct)

    row = {
        "challenge_id": state["challenge_id"],
        "target_label": state["target_label"],
        "n_targets": sum(o["is_target"] for o in state["objects"]),
        "n_objects": len(state["objects"]),
        "clicks": json.dumps(clicks),
        "is_correct": is_correct,
        "created_at": state["created_at"],
        "submitted_at": datetime.now().isoformat(),
        "objects": json.dumps(state["objects"])
    }

    pd.DataFrame([row]).to_csv(
        RESULTS_CSV,
        mode="a" if os.path.exists(RESULTS_CSV) else "w",
        header=not os.path.exists(RESULTS_CSV),
        index=False
    )

    if is_correct:
        return """
        <div class="result-success">
            ✓ Verification successful
        </div>
        """

    return """
    <div class="result-error">
        ✕ Incorrect selection
    </div>
    """


def verify(selected_json, state):

    if state and state.get("type") == "area":
        return verify_area(selected_json, state)

    print("\n==============================")
    print("VERIFY")
    print("==============================")

    print("Raw selected_json:")
    print(repr(selected_json))


    if not state:
        return "⚠️ No challenge loaded."


    # If empty
    if not selected_json:
        selected_json = "[]"


    try:

        selected = json.loads(
            selected_json
        )

    except Exception as e:

        print(
            "JSON ERROR:",
            e
        )

        selected = []


    selected = sorted(
        int(x)
        for x in selected
    )

    correct = sorted(
        state["correct_indices"]
    )


    print("Selected:", selected)
    print("Correct :", correct)


    is_correct = (
        set(selected)
        ==
        set(correct)
    )


    print(
        "Correct answer?",
        is_correct
    )


    # ========================================================
    # SAVE RESULT
    # ========================================================

    row = {
        "challenge_id": state["challenge_id"],
        "target_label": state["target_label"],
        "correct_indices": ",".join(
            map(str, correct)
        ),
        "selected_indices": ",".join(
            map(str, selected)
        ),
        "is_correct": is_correct,
        "created_at": state["created_at"],
        "submitted_at": datetime.now().isoformat()
    }


    for i, sample in enumerate(
        state["samples"],
        start=1
    ):

        row[
            f"cell{i}_image"
        ] = sample["image_name"]

        row[
            f"cell{i}_label"
        ] = sample["label"]


    result_df = pd.DataFrame(
        [row]
    )


    if os.path.exists(
        RESULTS_CSV
    ):

        result_df.to_csv(
            RESULTS_CSV,
            mode="a",
            header=False,
            index=False
        )

    else:

        result_df.to_csv(
            RESULTS_CSV,
            index=False
        )


    if is_correct:

        return """
        <div class="result-success">
            ✓ Verification successful
        </div>
        """

    else:

        return """
        <div class="result-error">
            ✕ Incorrect selection
        </div>
        """


# ============================================================
# CSS
# ============================================================

css = """

body {
    background: #f5f5f5;
}


/* ==========================================
   APP WIDTH
========================================== */

.gradio-container {

    max-width: 490px !important;

    margin-left: auto !important;
    margin-right: auto !important;

}


/* ==========================================
   CAPTCHA CARD
========================================== */

.captcha-card {

    width: 420px;

    background: #ffffff;

    padding: 7px;

    border: 1px solid #bdbdbd;

    box-shadow:
        0 2px 8px
        rgba(0, 0, 0, 0.18);

    margin:

        12px auto
        8px auto;

}


/* ==========================================
   HEADER
========================================== */

.captcha-header {

    background: #4285f4;

    color: white;

    padding:

        14px
        18px;

    margin-bottom: 4px;

}


.captcha-instruction {

    font-size: 14px;

    line-height: 18px;

}


.captcha-target {

    font-size: 27px;

    line-height: 32px;

    font-weight: 700;

    margin:

        2px 0
        3px 0;

}


.captcha-footer-text {

    font-size: 12px;

    opacity: 0.95;

}


/* ==========================================
   GRID
========================================== */

.captcha-grid {

    display: grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap: 3px;

}


/* ==========================================
   TILE
========================================== */

.captcha-tile {

    position: relative;

    width: 100%;

    aspect-ratio: 1 / 1;

    overflow: hidden;

    cursor: pointer;

    box-sizing: border-box;

    background: #ddd;

    user-select: none;

    transition:
        transform 0.08s ease,
        border 0.08s ease;

}


.captcha-tile img {

    width: 100%;

    height: 100%;

    object-fit: cover;

    display: block;

    user-select: none;

    pointer-events: none;

}


/* ==========================================
   SELECTED TILE
========================================== */

.captcha-tile.selected {

    border: 5px solid #4285f4;

}


.captcha-tile.selected img {

    transform: scale(0.92);

}


/* ==========================================
   CHECKMARK
========================================== */

.captcha-check {

    position: absolute;

    top: 6px;

    left: 6px;

    width: 27px;

    height: 27px;

    border-radius: 50%;

    background: #4285f4;

    color: white;

    font-weight: bold;

    font-size: 18px;

    display: none;

    align-items: center;

    justify-content: center;

    box-shadow:

        0 1px 4px
        rgba(0,0,0,.45);

}


.captcha-tile.selected
.captcha-check {

    display: flex;

}


/* ==========================================
   BUTTONS
========================================== */

#new-btn {

    max-width: 190px;

}


#verify-btn {

    max-width: 190px;

    margin-left: auto;

}


/* ==========================================
   RESULT
========================================== */

.result-success {

    text-align: center;

    font-size: 17px;

    font-weight: 600;

    color: #188038;

    margin-top: 8px;

}


.result-error {

    text-align: center;

    font-size: 17px;

    font-weight: 600;

    color: #d93025;

    margin-top: 8px;

}


/* ==========================================
   HIDDEN STATE TEXTBOX
========================================== */

.area-canvas {

    position: relative;

    cursor: crosshair;

    user-select: none;

    line-height: 0;

}


.area-canvas img {

    width: 100%;

    display: block;

    pointer-events: none;

}


.area-marker {

    position: absolute;

    width: 22px;

    height: 22px;

    margin: -11px 0 0 -11px;

    border-radius: 50%;

    background: rgba(66, 133, 244, 0.85);

    border: 2px solid white;

    box-shadow: 0 1px 4px rgba(0,0,0,.6);

    pointer-events: none;

}


#selected-json {

    display: none !important;

}


/* ==========================================
   REMOVE EXTRA GRADIO SPACING
========================================== */

.block {

    margin-top: 4px !important;
    margin-bottom: 4px !important;

}

"""


# ============================================================
# JAVASCRIPT
# ============================================================

js = """
function () {

    // Holds selected cell indices
    window.captchaSelected = [];


    // =======================================================
    // Update hidden Gradio textbox
    // =======================================================

    window.updateGradioSelection = function() {

        const container =
            document.querySelector(
                "#selected-json"
            );

        if (!container) {

            console.error(
                "selected-json container not found"
            );

            return;
        }


        const textbox =
            container.querySelector(
                "textarea, input"
            );


        if (!textbox) {

            console.error(
                "selected-json input not found"
            );

            return;
        }


        const value =
            JSON.stringify(
                window.captchaSelected
            );


        // ---------------------------------------------------
        // IMPORTANT:
        //
        // Use native setter so React / Gradio notices change
        // ---------------------------------------------------

        const prototype =
            textbox.tagName === "TEXTAREA"
                ? HTMLTextAreaElement.prototype
                : HTMLInputElement.prototype;


        const valueSetter =
            Object.getOwnPropertyDescriptor(
                prototype,
                "value"
            ).set;


        valueSetter.call(
            textbox,
            value
        );


        textbox.dispatchEvent(

            new Event(
                "input",
                {
                    bubbles: true
                }
            )

        );


        textbox.dispatchEvent(

            new Event(
                "change",
                {
                    bubbles: true
                }
            )

        );


        console.log(
            "Selected indices:",
            window.captchaSelected
        );

        console.log(
            "Sent to Gradio:",
            value
        );
    };


    // =======================================================
    // CLICK TILE
    // =======================================================

    window.toggleCaptchaTile = function(tile) {

        const index =
            parseInt(
                tile.dataset.index
            );


        // Toggle CSS
        tile.classList.toggle(
            "selected"
        );


        // Add / remove index
        if (

            window.captchaSelected.includes(
                index
            )

        ) {

            window.captchaSelected =
                window.captchaSelected.filter(
                    x => x !== index
                );

        }

        else {

            window.captchaSelected.push(
                index
            );

        }


        // Sort
        window.captchaSelected.sort(
            (a, b) => a - b
        );


        // Push to Gradio
        window.updateGradioSelection();

    };


    // =======================================================
    // RESET
    // =======================================================

    // =======================================================
    // AREA-SELECT: click points on the canvas
    // =======================================================

    window.areaClick = function(event, canvas) {

        const img = canvas.querySelector("img");
        const rect = img.getBoundingClientRect();

        // Displayed position -> canvas pixel coordinates
        const scale = img.naturalWidth / rect.width;
        const x = Math.round((event.clientX - rect.left) * scale);
        const y = Math.round((event.clientY - rect.top) * scale);

        // Clicking near an existing marker removes it
        const radius = 14 * scale;
        const existing = window.captchaSelected.findIndex(
            p => Math.hypot(p[0] - x, p[1] - y) <= radius
        );

        if (existing >= 0) {
            window.captchaSelected.splice(existing, 1);
        } else {
            window.captchaSelected.push([x, y]);
        }

        canvas.querySelectorAll(".area-marker").forEach(m => m.remove());

        window.captchaSelected.forEach(p => {
            const marker = document.createElement("div");
            marker.className = "area-marker";
            marker.style.left = (100 * p[0] / img.naturalWidth) + "%";
            marker.style.top = (100 * p[1] / img.naturalHeight) + "%";
            canvas.appendChild(marker);
        });

        window.updateGradioSelection();

    };


    window.resetCaptchaSelection =
        function() {

            window.captchaSelected = [];

            document
                .querySelectorAll(
                    ".captcha-tile.selected"
                )
                .forEach(
                    tile =>
                        tile.classList.remove(
                            "selected"
                        )
                );

            window.updateGradioSelection();

        };

}
"""


# ============================================================
# GRADIO APP
# ============================================================

with gr.Blocks(
    css=css,
    js=js,
    title="Image CAPTCHA"
) as demo:


    # ========================================================
    # PYTHON STATE
    # ========================================================

    challenge_state = gr.State(
        None
    )


    # ========================================================
    # CAPTCHA
    # ========================================================

    captcha_html = gr.HTML()


    # ========================================================
    # HIDDEN SELECTED INDICES
    # ========================================================

    # IMPORTANT:
    #
    # do NOT use visible=False
    #
    # because JavaScript needs this element to exist
    # in the browser DOM.
    #
    # CSS hides it instead.

    selected_json = gr.Textbox(
        value="[]",
        label="Selected indices",
        elem_id="selected-json"
    )


    # ========================================================
    # RESULT
    # ========================================================

    result = gr.HTML()


    # ========================================================
    # BUTTONS
    # ========================================================

    with gr.Row():

        new_btn = gr.Button(
            "↻ New Challenge",
            elem_id="new-btn"
        )

        verify_btn = gr.Button(
            "VERIFY",
            variant="primary",
            elem_id="verify-btn"
        )


    # ========================================================
    # NEW CAPTCHA
    # ========================================================

    new_btn.click(

        fn=make_challenge,

        inputs=[],

        outputs=[
            captcha_html,
            challenge_state,
            result,
            selected_json
        ],

        js="""
        () => {

            window.captchaSelected = [];

            return [];
        }
        """
    )


    # ========================================================
    # VERIFY
    # ========================================================

    verify_btn.click(

        fn=verify,

        inputs=[
            selected_json,
            challenge_state
        ],

        outputs=[
            result
        ]
    )


    # ========================================================
    # LOAD FIRST CAPTCHA
    # ========================================================

    demo.load(

        fn=make_challenge,

        inputs=[],

        outputs=[
            captcha_html,
            challenge_state,
            result,
            selected_json
        ]
    )


# ============================================================
# LAUNCH
# ============================================================

demo.launch(

    server_port=cli_args.port,

    allowed_paths=[
        os.path.abspath(
            IMAGE_DIR
        )
    ]

)