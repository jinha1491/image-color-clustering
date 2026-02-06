import os
import uuid
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from skimage import color  # used for LAB color space conversion

# --------------------------------------------------
# Flask app setup
# --------------------------------------------------

app = Flask(__name__)

# Folder where uploaded images are stored
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Only allow common image formats
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def allowed_file(filename: str) -> bool:
    """Check if uploaded file has an allowed image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def rgb_to_hex(rgb: np.ndarray) -> str:
    """Convert an RGB triplet to HEX format for display."""
    r, g, b = rgb.astype(int).tolist()
    return f"#{r:02x}{g:02x}{b:02x}"


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

def extract_pixels_rgb(
    image_path: str,
    resize_to: int = 200,
    brightness_low: int = 50,
    brightness_high: int = 240,
) -> np.ndarray:
    """
    Load an image and return pixel-level RGB values in [0,1].

    Extremely dark and extremely bright pixels are filtered out
    to reduce noise from shadows, highlights, and backgrounds.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((resize_to, resize_to))

    rgb = np.asarray(img, dtype=np.float32) / 255.0
    pixels_rgb = rgb.reshape(-1, 3)

    # Brightness filtering (simple average across channels)
    brightness = pixels_rgb.mean(axis=1) * 255.0
    mask = (brightness > brightness_low) & (brightness < brightness_high)
    filtered = pixels_rgb[mask]

    # If filtering removes too many pixels, fall back to unfiltered data
    if filtered.shape[0] < 200:
        return pixels_rgb

    return filtered


# --------------------------------------------------
# Model selection (Auto-K)
# --------------------------------------------------

def pick_k_auto(
    pixels_lab: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int = 42,
    sample_size: int = 3000,
) -> int:
    """
    Select the number of clusters using silhouette score.

    A subset of pixels is used for efficiency on larger images.
    """
    n = pixels_lab.shape[0]
    if n < (k_min + 1):
        return 1

    # Sample pixels for faster silhouette computation
    if n > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=sample_size, replace=False)
        sample = pixels_lab[idx]
    else:
        sample = pixels_lab

    # Silhouette requires k <= n - 1
    k_max = min(k_max, sample.shape[0] - 1)
    if k_max < k_min:
        return max(1, k_max)

    best_k = k_min
    best_score = -1.0

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = km.fit_predict(sample)

        # Skip degenerate cases
        if len(np.unique(labels)) < 2:
            continue

        score = silhouette_score(sample, labels)
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


# --------------------------------------------------
# Palette extraction
# --------------------------------------------------

def get_palette(
    image_path: str,
    max_colors: int = 10,
    auto_k: bool = True,
):
    """
    Extract dominant colors from an image.

    Colors are clustered in LAB space for perceptual similarity.
    Returns both HEX colors and their pixel composition percentages.
    """
    pixels_rgb = extract_pixels_rgb(image_path)

    # Convert RGB to LAB before clustering
    pixels_lab = color.rgb2lab(pixels_rgb.reshape(1, -1, 3)).reshape(-1, 3)

    n = pixels_lab.shape[0]
    if n == 0:
        return [], 0

    max_colors = min(max_colors, n)

    # Choose number of clusters
    if auto_k and max_colors >= 2:
        chosen_k = pick_k_auto(pixels_lab, 2, max_colors)
    else:
        chosen_k = max_colors

    chosen_k = max(1, min(chosen_k, n))

    # Final clustering on all pixels
    kmeans = KMeans(n_clusters=chosen_k, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(pixels_lab)

    # Count pixels per cluster and sort by dominance
    counts = np.bincount(labels, minlength=chosen_k)
    order = np.argsort(counts)[::-1]
    total = counts.sum()

    centers_lab = kmeans.cluster_centers_[order]

    # Convert cluster centers back to RGB for display
    centers_rgb = color.lab2rgb(centers_lab.reshape(1, -1, 3)).reshape(-1, 3)
    centers_rgb = np.clip(centers_rgb * 255.0, 0, 255).astype(int)

    palette = []
    for i, rgb_center in enumerate(centers_rgb):
        percent = (counts[order[i]] / total) * 100.0
        palette.append({
            "hex": rgb_to_hex(rgb_center),
            "percent": round(percent, 1),
        })

    return palette, chosen_k


# --------------------------------------------------
# Flask route
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    palette = []
    image_url = None
    error = None
    chosen_k = None

    if request.method == "POST":
        file = request.files.get("image")

        if not file or file.filename == "":
            error = "No file selected."
        elif not allowed_file(file.filename):
            error = "Unsupported file type."
        else:
            # Save uploaded image with a unique filename
            ext = file.filename.rsplit(".", 1)[1].lower()
            safe_name = secure_filename(file.filename.rsplit(".", 1)[0]) or "image"
            filename = f"{safe_name}-{uuid.uuid4().hex[:10]}.{ext}"

            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(image_path)

            # Extract color palette
            palette, chosen_k = get_palette(image_path)
            image_url = f"uploads/{filename}"

    return render_template(
        "index.html",
        palette=palette,
        image_url=image_url,
        error=error,
        chosen_k=chosen_k,
    )


if __name__ == "__main__":
    app.run(debug=True)
