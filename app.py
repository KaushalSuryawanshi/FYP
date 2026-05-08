import os
import io
import csv
import cv2
import math
import numpy as np
import gradio as gr
import tempfile
import torch
import pandas as pd
import rasterio
from PIL import Image
from deepforest import main as df_main
from ultralytics import YOLO

# ===================== CONFIG =====================
TILE_SIZE   = 640
OVERLAP     = 0.15

# Confidence colour thresholds
def get_colour(score):
    if score >= 0.7:
        return (0, 255, 0)       # Green  — high confidence
    elif score >= 0.4:
        return (0, 165, 255)     # Orange — medium confidence
    else:
        return (0, 0, 255)       # Red    — low confidence

# ===================== LOAD BOTH MODELS AT STARTUP =====================
print("Loading DeepForest model...")
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

df_main.deepforest.create_trainer = lambda self, *args, **kwargs: None
df_main.deepforest.setup_metrics  = lambda self, *args, **kwargs: None

deepforest_model = df_main.deepforest.load_from_checkpoint(
    "deepforest_trained.ckpt",
    map_location=torch.device("cpu")
)
print("DeepForest ready.")

print("Loading YOLO model...")
yolo_model = YOLO("best2.pt")
print("YOLO ready.")

# ===================== NMS HELPER (for DeepForest tiling) =====================
def nms(detections, iou_thresh=0.4):
    if not detections:
        return []
    boxes  = np.array([[d[0], d[1], d[2], d[3]] for d in detections], dtype=np.float32)
    scores = np.array([d[4] for d in detections], dtype=np.float32)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas  = (x2 - x1) * (y2 - y1)
    order  = scores.argsort()[::-1]
    keep   = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1   = np.maximum(x1[i], x1[order[1:]])
        yy1   = np.maximum(y1[i], y1[order[1:]])
        xx2   = np.minimum(x2[i], x2[order[1:]])
        yy2   = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_thresh]
    return [detections[k] for k in keep]

# ===================== DRAW DETECTIONS =====================
def draw_detections(image, detections, show_not_tree=False):
    """
    detections: list of [x1, y1, x2, y2, score, label]
    Draws numbered coloured boxes. Returns annotated image.
    """
    annotated = image.copy()
    tree_num  = 1

    for (x1, y1, x2, y2, score, label) in detections:
        is_tree = label.lower() == "tree"

        if not is_tree and not show_not_tree:
            continue

        colour = get_colour(score) if is_tree else (128, 128, 128)

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

        # Build label text
        if is_tree:
            text = f"#{tree_num} {score:.2f}"
            tree_num += 1
        else:
            text = f"Not Tree {score:.2f}"

        # Draw filled background rectangle for text
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, max(y1-18, 0)), (x1+tw+4, y1), colour, -1)
        cv2.putText(annotated, text, (x1+2, max(y1-4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return annotated

# ===================== DEEPFOREST INFERENCE =====================
def run_deepforest(image, conf_threshold, is_ortho, progress):
    raw_dets = []

    if is_ortho:
        h, w, _ = image.shape
        stride   = int(TILE_SIZE * (1 - OVERLAP))
        ys = list(range(0, max(h - TILE_SIZE + 1, 1), stride))
        xs = list(range(0, max(w - TILE_SIZE + 1, 1), stride))
        total_tiles = len(ys) * len(xs)
        tile_counter = 0

        progress(0.1, desc="Starting tiled inference (DeepForest)...")

        for yi, y in enumerate(ys):
            for xi, x in enumerate(xs):
                y_end = min(y + TILE_SIZE, h)
                x_end = min(x + TILE_SIZE, w)
                tile  = image[y:y_end, x:x_end]

                if np.mean(tile) < 1:
                    continue

                pad_h = TILE_SIZE - tile.shape[0]
                pad_w = TILE_SIZE - tile.shape[1]
                if pad_h > 0 or pad_w > 0:
                    tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)))

                tile_counter += 1
                preds = deepforest_model.predict_image(tile)

                if preds is not None and len(preds) > 0:
                    for _, row in preds.iterrows():
                        score = float(row["score"])
                        if score < conf_threshold:
                            continue
                        raw_dets.append([
                            int(row["xmin"]) + x,
                            int(row["ymin"]) + y,
                            int(row["xmax"]) + x,
                            int(row["ymax"]) + y,
                            score,
                            "Tree"
                        ])

                done = (yi * len(xs) + xi + 1) / total_tiles
                progress(0.1 + 0.7 * done,
                         desc=f"Tile {tile_counter}/{total_tiles} — {len(raw_dets)} detections")

        progress(0.82, desc="Applying NMS...")
        # NMS expects 5-element lists; pass without label then re-attach
        nms_input  = [[d[0],d[1],d[2],d[3],d[4]] for d in raw_dets]
        nms_result = nms(nms_input, 0.4)
        final_dets = [[d[0],d[1],d[2],d[3],d[4],"Tree"] for d in nms_result]

    else:
        # Singular image — direct inference, no tiling
        progress(0.3, desc="Running DeepForest on full image...")
        preds = deepforest_model.predict_image(image)
        final_dets = []
        if preds is not None and len(preds) > 0:
            for _, row in preds.iterrows():
                score = float(row["score"])
                if score < conf_threshold:
                    continue
                final_dets.append([
                    int(row["xmin"]), int(row["ymin"]),
                    int(row["xmax"]), int(row["ymax"]),
                    score, "Tree"
                ])

    return final_dets

# ===================== YOLO INFERENCE =====================
def run_yolo(image, conf_threshold, is_ortho, show_not_tree, progress):
    raw_dets = []

    # YOLO class names — adjust if your model uses different order
    CLASS_NAMES = {0: "Tree", 1: "Not Tree"}

    if is_ortho:
        h, w, _ = image.shape
        stride   = int(TILE_SIZE * (1 - OVERLAP))
        ys = list(range(0, max(h - TILE_SIZE + 1, 1), stride))
        xs = list(range(0, max(w - TILE_SIZE + 1, 1), stride))
        total_tiles = len(ys) * len(xs)
        tile_counter = 0

        progress(0.1, desc="Starting tiled inference (YOLO)...")

        for yi, y in enumerate(ys):
            for xi, x in enumerate(xs):
                y_end = min(y + TILE_SIZE, h)
                x_end = min(x + TILE_SIZE, w)
                tile  = image[y:y_end, x:x_end]

                if np.mean(tile) < 1:
                    continue

                pad_h = TILE_SIZE - tile.shape[0]
                pad_w = TILE_SIZE - tile.shape[1]
                if pad_h > 0 or pad_w > 0:
                    tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)))

                tile_counter += 1
                results = yolo_model.predict(tile, conf=conf_threshold, verbose=False)

                for result in results:
                    for box in result.boxes:
                        score = float(box.conf[0])
                        cls   = int(box.cls[0])
                        label = CLASS_NAMES.get(cls, f"Class {cls}")
                        xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
                        raw_dets.append([
                            int(xmin) + x, int(ymin) + y,
                            int(xmax) + x, int(ymax) + y,
                            score, label
                        ])

                done = (yi * len(xs) + xi + 1) / total_tiles
                progress(0.1 + 0.7 * done,
                         desc=f"Tile {tile_counter}/{total_tiles} — {len(raw_dets)} detections")

        progress(0.82, desc="Applying NMS per class...")
        tree_dets     = [[d[0],d[1],d[2],d[3],d[4]] for d in raw_dets if d[5]=="Tree"]
        nottree_dets  = [[d[0],d[1],d[2],d[3],d[4]] for d in raw_dets if d[5]=="Not Tree"]
        tree_nms      = [[*d, "Tree"]     for d in nms(tree_dets,    0.4)]
        nottree_nms   = [[*d, "Not Tree"] for d in nms(nottree_dets, 0.4)]
        final_dets    = tree_nms + nottree_nms

    else:
        # Singular image
        progress(0.3, desc="Running YOLO on full image...")
        results    = yolo_model.predict(image, conf=conf_threshold, verbose=False)
        final_dets = []
        for result in results:
            for box in result.boxes:
                score = float(box.conf[0])
                cls   = int(box.cls[0])
                label = CLASS_NAMES.get(cls, f"Class {cls}")
                xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
                final_dets.append([
                    int(xmin), int(ymin),
                    int(xmax), int(ymax),
                    score, label
                ])

    return final_dets

# ===================== BUILD CSV =====================
def build_csv(final_dets):
    output   = io.StringIO()
    writer   = csv.writer(output)
    writer.writerow(["Tree_Number", "x1", "y1", "x2", "y2", "Score", "Label"])
    tree_num = 1
    for (x1, y1, x2, y2, score, label) in final_dets:
        num = tree_num if label.lower() == "tree" else "-"
        writer.writerow([num, x1, y1, x2, y2, f"{score:.4f}", label])
        if label.lower() == "tree":
            tree_num += 1
    return output.getvalue()

# ===================== BUILD SUMMARY =====================
def build_summary(final_dets, tile_counter, raw_count, w, h, model_name):
    tree_dets     = [d for d in final_dets if d[5].lower() == "tree"]
    nottree_dets  = [d for d in final_dets if d[5].lower() != "tree"]

    high   = sum(1 for d in tree_dets if d[4] >= 0.7)
    medium = sum(1 for d in tree_dets if 0.4 <= d[4] < 0.7)
    low    = sum(1 for d in tree_dets if d[4] < 0.4)
    avg_score = np.mean([d[4] for d in tree_dets]) if tree_dets else 0

    summary = (
        f"╔══════════════════════════════════════╗\n"
        f"║         DETECTION RESULTS            ║\n"
        f"╚══════════════════════════════════════╝\n\n"
        f"🤖 Model          : {model_name}\n"
        f"🖼️  Image size     : {w} × {h} px\n"
        f"🧩 Tiles processed : {tile_counter}\n"
        f"📦 Raw detections  : {raw_count}\n\n"
        f"🌲 Trees detected  : {len(tree_dets)}\n"
        f"🌿 Not-Tree regions: {len(nottree_dets)}\n"
        f"📊 Avg confidence  : {avg_score:.3f}\n\n"
        f"Confidence Breakdown:\n"
        f"  🟢 High  (≥0.70) : {high}\n"
        f"  🟠 Medium(0.40-0.69): {medium}\n"
        f"  🔴 Low   (<0.40) : {low}\n\n"
        f"{'─'*42}\n"
        f"{'#':<6} {'x1':>6} {'y1':>6} {'x2':>6} {'y2':>6} {'score':>7}  label\n"
        f"{'─'*42}\n"
    )
    tree_num = 1
    for (x1, y1, x2, y2, score, label) in final_dets:
        num = str(tree_num) if label.lower() == "tree" else "-"
        summary += f"{num:<6} {x1:>6} {y1:>6} {x2:>6} {y2:>6} {score:>7.3f}  {label}\n"
        if label.lower() == "tree":
            tree_num += 1

    return summary

# ===================== MAIN PIPELINE =====================
def run_pipeline(
    uploaded_file,
    model_choice,
    image_type,
    conf_threshold,
    show_not_tree,
    progress=gr.Progress()
):
    if uploaded_file is None:
        return None, None, None, None, "⚠️ Please upload an image first."

    ext      = os.path.splitext(uploaded_file.name)[-1].lower()
    is_ortho = (image_type == "Orthophoto (GeoTIFF / large aerial)")

    # ── Read image ──────────────────────────────
    progress(0.05, desc="Reading image...")
    if ext in [".tif", ".tiff"]:
        with rasterio.open(uploaded_file.name) as src:
            image = src.read([1, 2, 3])
            image = np.transpose(image, (1, 2, 0)).astype(np.uint8)
    else:
        pil   = Image.open(uploaded_file.name).convert("RGB")
        image = np.array(pil)

    h, w, _ = image.shape
    original_pil = Image.fromarray(image)

    # ── Run selected model ───────────────────────
    if model_choice == "DeepForest":
        final_dets   = run_deepforest(image, conf_threshold, is_ortho, progress)
        raw_count    = len(final_dets)
        tile_counter = math.ceil(h / TILE_SIZE) * math.ceil(w / TILE_SIZE) if is_ortho else 1
        model_name   = "DeepForest (RetinaNet + ResNet50)"
    else:
        # YOLO — we need raw count before NMS which is inside run_yolo
        final_dets   = run_yolo(image, conf_threshold, is_ortho, show_not_tree, progress)
        raw_count    = len(final_dets)
        tile_counter = math.ceil(h / TILE_SIZE) * math.ceil(w / TILE_SIZE) if is_ortho else 1
        model_name   = "YOLOv8 (best2.pt)"

    # ── Draw detections ──────────────────────────
    progress(0.90, desc="Drawing detections...")
    annotated     = draw_detections(image, final_dets, show_not_tree)
    annotated_pil = Image.fromarray(annotated)

    # ── Save annotated image to temp file ────────
    progress(0.93, desc="Saving output...")
    out_img_path = tempfile.mktemp(suffix="_annotated.jpg")
    cv2.imwrite(out_img_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    # ── Build CSV ────────────────────────────────
    progress(0.95, desc="Building CSV...")
    csv_string = build_csv(final_dets)
    csv_path   = tempfile.mktemp(suffix="_detections.csv")
    with open(csv_path, "w") as f:
        f.write(csv_string)

    # ── Build summary ────────────────────────────
    summary = build_summary(final_dets, tile_counter, raw_count, w, h, model_name)

    progress(1.0, desc="Done!")
    return original_pil, annotated_pil, out_img_path, csv_path, summary

# ===================== GRADIO UI =====================
with gr.Blocks(title="🌲 Tree Detection App", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🌲 Tree Detection App
    Detect trees in aerial imagery using **DeepForest** or **YOLO**.
    Upload your image, select your settings and click **Run Detection**.
    """)

    with gr.Row():

        # ── LEFT COLUMN — Inputs & Model Info ───────────────────────────
        with gr.Column(scale=1):

            gr.Markdown("### ⚙️ Settings")

            uploaded_file = gr.File(
                label="Upload Image (.tif / .jpg / .png)",
                file_types=[".tif", ".tiff", ".jpg", ".jpeg", ".png"]
            )

            model_choice = gr.Radio(
                choices=["DeepForest", "YOLO"],
                value="DeepForest",
                label="🤖 Select Model"
            )

            image_type = gr.Radio(
                choices=["Orthophoto (GeoTIFF / large aerial)", "Singular Drone Image"],
                value="Orthophoto (GeoTIFF / large aerial)",
                label="🖼️ Image Type"
            )

            conf_slider = gr.Slider(
                0.1, 1.0, value=0.25, step=0.05,
                label="Confidence Threshold"
            )

            show_not_tree = gr.Checkbox(
                label="Show 'Not Tree' detections (YOLO only)",
                value=False
            )

            run_btn = gr.Button("🚀 Run Detection", variant="primary")


        # ── RIGHT COLUMN — Outputs ───────────────────────────────────────
        with gr.Column(scale=2):

            gr.Markdown("### 📊 Results")

            with gr.Row():
                original_out  = gr.Image(type="pil", label="Original Image")
                annotated_out = gr.Image(type="pil", label="Annotated Output")

            with gr.Row():
                download_img = gr.File(label="⬇️ Download Annotated Image")
                download_csv = gr.File(label="⬇️ Download Detections CSV")

            summary_out = gr.Textbox(
                label="Detection Summary",
                lines=25,
                max_lines=50
            )

    run_btn.click(
        fn=run_pipeline,
        inputs=[
            uploaded_file,
            model_choice,
            image_type,
            conf_slider,
            show_not_tree
        ],
        outputs=[
            original_out,
            annotated_out,
            download_img,
            download_csv,
            summary_out
        ]
    )

demo.launch()