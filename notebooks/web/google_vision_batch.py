import os
import time
import base64
import requests
import pandas as pd
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

API_KEY = os.environ.get("GOOGLE_VISION_API_KEY", "")

IMAGE_FOLDER = r"google_test_images\adversarial"

OUTPUT_CSV = "google_vision_adversarial_results.csv"

TOP_K = 10

# Small delay between requests
SLEEP_TIME = 0.10


# ============================================================
# GOOGLE VISION REQUEST
# ============================================================

def get_google_labels(image_path):

    with open(image_path, "rb") as f:
        image_content = base64.b64encode(
            f.read()
        ).decode("utf-8")

    url = (
        "https://vision.googleapis.com/v1/images:annotate"
        f"?key={API_KEY}"
    )

    request_body = {
        "requests": [
            {
                "image": {
                    "content": image_content
                },
                "features": [
                    {
                        "type": "LABEL_DETECTION",
                        "maxResults": TOP_K
                    }
                ]
            }
        ]
    }

    response = requests.post(
        url,
        json=request_body,
        timeout=60
    )

    response.raise_for_status()

    result = response.json()

    # Check API-level error
    if (
        "responses" not in result
        or len(result["responses"]) == 0
    ):
        raise RuntimeError(
            f"Unexpected response: {result}"
        )

    api_response = result["responses"][0]

    if "error" in api_response:
        raise RuntimeError(
            api_response["error"]
        )

    return api_response.get(
        "labelAnnotations",
        []
    )


# ============================================================
# FIND IMAGES
# ============================================================

extensions = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp"
)

images = sorted([
    f
    for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith(extensions)
])

print("=" * 65)
print("GOOGLE CLOUD VISION - LABEL DETECTION")
print("=" * 65)
print("Folder:", IMAGE_FOLDER)
print("Total images found:", len(images))
print("Output:", OUTPUT_CSV)
print("=" * 65)


# ============================================================
# RESUME SUPPORT
# ============================================================

processed = set()
rows = []

if os.path.exists(OUTPUT_CSV):

    old_df = pd.read_csv(
        OUTPUT_CSV
    )

    rows = old_df.to_dict(
        orient="records"
    )

    if "image" in old_df.columns:
        processed = set(
            old_df["image"]
            .dropna()
            .astype(str)
        )

    print(
        f"Existing results found: "
        f"{len(processed)} images"
    )


remaining_images = [
    f for f in images
    if f not in processed
]

print(
    "Remaining images:",
    len(remaining_images)
)


# ============================================================
# PROCESS
# ============================================================

for filename in tqdm(
    remaining_images,
    desc="Google Vision"
):

    image_path = os.path.join(
        IMAGE_FOLDER,
        filename
    )

    row = {
        "image": filename
    }

    try:

        labels = get_google_labels(
            image_path
        )

        row["status"] = "success"

        for i, item in enumerate(
            labels[:TOP_K],
            start=1
        ):

            row[f"top{i}_label"] = (
                item.get(
                    "description",
                    ""
                )
            )

            # Google returns 0-1 score
            row[f"top{i}_confidence"] = (
                item.get(
                    "score",
                    0
                ) * 100
            )

    except Exception as e:

        row["status"] = "failed"
        row["error"] = str(e)

        print(
            f"\nError on {filename}: {e}"
        )

    rows.append(row)


    # ========================================================
    # SAVE AFTER EVERY IMAGE
    # ========================================================
    #
    # This is slower than saving only at the end,
    # but means you don't lose progress if the script stops.
    # ========================================================

    pd.DataFrame(
        rows
    ).to_csv(
        OUTPUT_CSV,
        index=False
    )

    time.sleep(
        SLEEP_TIME
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

df = pd.DataFrame(rows)

success_count = (
    (df["status"] == "success").sum()
    if "status" in df.columns
    else 0
)

failed_count = (
    (df["status"] == "failed").sum()
    if "status" in df.columns
    else 0
)


print("\n" + "=" * 65)
print("FINISHED")
print("=" * 65)

print(
    "Total images:",
    len(df)
)

print(
    "Successful:",
    success_count
)

print(
    "Failed:",
    failed_count
)

print(
    "Saved:",
    OUTPUT_CSV
)