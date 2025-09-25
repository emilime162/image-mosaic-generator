import numpy as np
from PIL import Image
import gradio as gr
import glob, os, zipfile, tempfile, shutil
from skimage.metrics import structural_similarity as ssim

# ---- Step 1: Preprocessing ----
def preprocess_image(image, target_size):
    pil_img = Image.fromarray(image)
    pil_img = pil_img.resize((target_size, target_size))
    return np.array(pil_img)

# ---- Helper: compute variance ----
def compute_variance(block):
    """Compute grayscale variance of a block."""
    if block.size == 0:
        return 0
    gray = np.dot(block[..., :3], [0.299, 0.587, 0.114])
    return np.var(gray)

# ---- Recursive split function (Simplified) ----
def split_and_classify(img, x, y, size, min_size=8, var_thresh=500,
                       tiles=None, tile_avgs=None):
    """
    Recursively split a block of the image if variance is high.
    Returns a list of ((x, y, size), classified_block).
    """
    h, w, _ = img.shape
    if x >= w or y >= h:
        return []
    
    block = img[y:min(y + size, h), x:min(x + size, w)]
    if block.size == 0:
        return []

    variance = compute_variance(block)

    if variance > var_thresh and size > min_size:
        half = size // 2
        return (
            split_and_classify(img, x, y, half, min_size, var_thresh, tiles, tile_avgs) +
            split_and_classify(img, x + half, y, half, min_size, var_thresh, tiles, tile_avgs) +
            split_and_classify(img, x, y + half, half, min_size, var_thresh, tiles, tile_avgs) +
            split_and_classify(img, x + half, y + half, half, min_size, var_thresh, tiles, tile_avgs)
        )
    else:
        # ✅ REMOVED: Logic for "Color Blocks" is gone.
        # Now it either uses an image tile or defaults to the original block if no tiles are loaded.
        if tiles is not None and tile_avgs is not None:
            avg_color = block.mean(axis=(0,1))
            dists = np.linalg.norm(tile_avgs - avg_color, axis=1)
            idx = np.argmin(dists)
            block_h, block_w, _ = block.shape
            classified = np.array(Image.fromarray(tiles[idx]).resize((block_w, block_h)))
        else:
            classified = block # Fallback if tiles failed to load

        return [((x, y, size), classified)]

# ---- Adaptive mosaic generator (Simplified) ----
def mosaic_generator(image, start_size=32, tile_folder=None, blend=0.0,
                     target_size=256, var_thresh=500, min_size=8):
    """
    Generate mosaic with adaptive non-fixed grid sizes.
    """
    img = preprocess_image(image, target_size)
    mosaic = np.zeros_like(img)
    
    tiles, tile_avgs = None, None
    if tile_folder:
        files = glob.glob(os.path.join(tile_folder, "*"))
        tile_list, avg_list = [], []
        for f in files:
            if os.path.isdir(f) or ".DS_Store" in f or "__MACOSX" in f: continue
            try:
                tile_img = Image.open(f).convert("RGB")
                arr = np.array(tile_img)
                tile_list.append(arr)
                avg_list.append(arr.mean(axis=(0, 1)))
            except Exception as e: print(f"[WARN] Skipping {f}: {e}")
        
        if tile_list:
            tiles = tile_list
            tile_avgs = np.array(avg_list)
        else:
            print("[ERROR] No usable tiles found in the provided folder/zip.")
    
    all_blocks = []
    h, w, _ = img.shape
    for y_start in range(0, h, start_size):
        for x_start in range(0, w, start_size):
            processed_blocks = split_and_classify(
                img, x_start, y_start, start_size,
                min_size=min_size, var_thresh=var_thresh,
                tiles=tiles, tile_avgs=tile_avgs
            )
            all_blocks.extend(processed_blocks)

    for (x, y, size), block in all_blocks:
        block_h, block_w, _ = block.shape
        cell = img[y:y+block_h, x:x+block_w]
        blended = (blend * cell + (1 - blend) * block).astype(np.uint8)
        mosaic[y:y+block_h, x:x+block_w] = blended

    return mosaic

# ---- Step 5: Similarity Metric ----
def compare_similarity(original, mosaic, target_size=256):
    original_processed = preprocess_image(original, target_size)
    mse = np.mean((original_processed - mosaic) ** 2)
    gray_original = np.dot(original_processed[..., :3], [0.299, 0.587, 0.114]).astype(np.float64)
    gray_mosaic = np.dot(mosaic[..., :3], [0.299, 0.587, 0.114]).astype(np.float64)
    ssim_val = ssim(gray_original, gray_mosaic, data_range=255)
    return float(mse), float(ssim_val)

# ---- Step 6: Pipeline (Simplified) ----
DEFAULT_TILESET = "./tiles.zip" 

def pipeline(image, grid_size, zip_file, blend, var_thresh, min_size, target_size=256):
    if image is None: return None, "No image uploaded"
    
    # ✅ REMOVED: Logic that depended on tile_mode
    # The app now always expects to unpack a tile set.
    tmpdir = tempfile.mkdtemp()
    zip_path = zip_file.name if zip_file is not None else DEFAULT_TILESET
    tile_folder = None
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.infolist():
                if not member.is_dir() and not member.filename.startswith('__MACOSX'):
                    zip_ref.extract(member, tmpdir)
        tile_folder = tmpdir
    except Exception as e: return image, f"[ERROR] Could not load tile set: {str(e)}"

    mosaic = mosaic_generator(image,
                              start_size=grid_size,
                              tile_folder=tile_folder,
                              blend=blend,
                              target_size=target_size,
                              var_thresh=var_thresh,
                              min_size=min_size)

    try:
        mse, ssim_val = compare_similarity(image, mosaic, target_size)
        metrics = f"MSE: {mse:.2f}, SSIM: {ssim_val:.3f}"
    except Exception as e: metrics = f"Error computing similarity: {str(e)}"

    if tmpdir: shutil.rmtree(tmpdir)
    return image, mosaic, metrics

# ---- Step 7: Gradio Interface (Simplified) ----
with gr.Blocks(title="Adaptive Mosaic Generator") as demo:
    gr.Markdown("## 🖼️ Interactive Image Mosaic Generator")
    gr.Markdown("This tool creates a photo mosaic using an adaptive grid. **Upload an image and a ZIP file of tiles to begin.**")
    with gr.Row():
        input_img = gr.Image(type="numpy", label="Upload Your Image")
        output_img = gr.Image(type="numpy", label="Mosaic Image")

    with gr.Accordion("Settings", open=True):
        grid_size = gr.Slider(8, 64, step=16, value=64, label="Starting Grid Size")
        min_size = gr.Slider(4, 32, step=4, value=8, label="Minimum Grid Size")
        var_thresh = gr.Slider(10, 200, step=10, value=100, label="Variance Threshold (Detail Level)")
        blend = gr.Slider(0, 1, step=0.05, value=0.0, label="Blending (0=Tiles, 1=Original)")
        
    zip_file = gr.File(label="Upload Tile Set (ZIP of images)", file_types=[".zip"])
    # ✅ REMOVED: The tile_mode dropdown is gone.

    metrics_box = gr.Textbox(label="📊 Similarity Metrics")
    run_btn = gr.Button("Generate Mosaic", variant="primary")
    
    # ✅ REMOVED: tile_mode removed from inputs list
    inputs = [input_img, grid_size, zip_file, blend, var_thresh, min_size]
    outputs = [input_img,output_img, metrics_box]
    
    run_btn.click(pipeline, inputs=inputs, outputs=outputs)
    
    # Create dummy files for examples
    if not os.path.exists("sample1.png"): Image.new('RGB', (256, 256), color='teal').save("sample1.png")
    if not os.path.exists("sample2.png"):
        img = Image.new('RGB', (256, 256), color='orange')
        img.paste(Image.new('RGB', (128, 128), color='navy'), (64, 64))
        img.save("sample2.png")
    if not os.path.exists(DEFAULT_TILESET):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f: fname = f.name
        Image.new('RGB', (32, 32), color = 'red').save(fname, format='PNG')
        with zipfile.ZipFile(DEFAULT_TILESET, 'w') as z: z.write(fname, arcname='red.png')
        os.remove(fname)

    # ✅ REMOVED: tile_mode argument from examples
    gr.Examples(
        examples=[
            ["sample2.png", 32, None, 0.2, 50, 4],
            ["sample1.png", 32, None, 0.2, 40, 8],
        ],
        inputs=inputs,
        outputs=outputs,
        fn=pipeline
    )

if __name__ == "__main__":
    demo.launch(share = True)