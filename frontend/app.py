"""Web demo backend: upload an image and run the NPR checkpoints on it.

Run from the repository root:
    python -m uvicorn frontend.app:app --port 8000
"""
from contextlib import asynccontextmanager
from io import BytesIO
import os
from pathlib import Path
import sys

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "baseline"))
from inference import CHECKPOINT, load_model  # noqa: E402

MAX_BYTES = 10 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG"}
JPEG_QUALITIES = {90, 70, 50}
MODEL_DIR = Path(os.environ.get("NPR_MODEL_DIR", ROOT / "outputs" / "main-experiment"))

# key -> (display name, description, checkpoint path)
MODEL_SPECS = {
    "baseline": ("NPR gốc", "Trọng số của tác giả bài báo", CHECKPOINT),
    "model_a": ("Mô hình A", "Nhóm tự huấn luyện, không nén ảnh", MODEL_DIR / "npr_A.pth"),
    "model_b": ("Mô hình B", "Nhóm tự huấn luyện, có tăng cường nén JPEG", MODEL_DIR / "npr_B_jpeg.pth"),
}

models = {}
load_errors = {}


def load_models():
    for key, (_, _, path) in MODEL_SPECS.items():
        if not Path(path).is_file():
            load_errors[key] = f"Không tìm thấy checkpoint: {path}"
            continue
        try:
            models[key] = load_model(Path(path).resolve())
        except Exception as error:  # report, keep the other models usable
            load_errors[key] = f"Lỗi nạp checkpoint: {error}"


@asynccontextmanager
async def lifespan(_app):
    load_models()  # once at startup; every request reuses the loaded models
    yield


app = FastAPI(title="NPR demo", lifespan=lifespan)


def model_info(key):
    name, description, _ = MODEL_SPECS[key]
    detector = models.get(key)
    return {
        "key": key,
        "name": name,
        "description": description,
        "available": detector is not None,
        "model_version": detector.model_version if detector else None,
        "error": load_errors.get(key),
    }


@app.get("/api/health")
def health():
    status = "ok" if "baseline" in models else "degraded"
    return {"status": status, "models": [model_info(k) for k in MODEL_SPECS]}


@app.get("/api/models")
def list_models():
    return [model_info(k) for k in MODEL_SPECS]


def read_image(data):
    if not data:
        raise HTTPException(422, "File rỗng.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Ảnh lớn hơn 10 MB.")
    try:
        with Image.open(BytesIO(data)) as probe:
            image_format = probe.format
            probe.verify()
        with Image.open(BytesIO(data)) as image:
            rgb = image.convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise HTTPException(422, "Không đọc được file ảnh. File có thể bị hỏng hoặc không phải ảnh.")
    if image_format not in ALLOWED_FORMATS:
        raise HTTPException(415, f"Chỉ nhận ảnh JPG hoặc PNG (file là {image_format}).")
    return rgb, image_format


def recompress(rgb, quality):
    """Same JPEG settings as the experiments: 4:2:0 chroma subsampling."""
    buffer = BytesIO()
    rgb.save(buffer, "JPEG", quality=quality, subsampling="4:2:0")
    size = buffer.tell()
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB"), size


@app.post("/api/predict")
def predict(image: UploadFile = File(...), models_csv: str = Form("baseline"),
            jpeg_quality: int = Form(0)):
    data = image.file.read(MAX_BYTES + 1)
    rgb, image_format = read_image(data)

    keys = [k.strip() for k in models_csv.split(",") if k.strip()]
    if not keys:
        raise HTTPException(422, "Chưa chọn mô hình nào.")
    unknown = [k for k in keys if k not in MODEL_SPECS]
    if unknown:
        raise HTTPException(422, f"Mô hình không tồn tại: {', '.join(unknown)}")
    missing = [MODEL_SPECS[k][0] for k in keys if k not in models]
    if missing:
        raise HTTPException(503, f"Mô hình chưa sẵn sàng: {', '.join(missing)}")

    compressed_bytes = None
    if jpeg_quality:
        if jpeg_quality not in JPEG_QUALITIES:
            raise HTTPException(422, "Mức nén JPEG chỉ nhận 90, 70 hoặc 50.")
        rgb, compressed_bytes = recompress(rgb, jpeg_quality)

    results = []
    for key in keys:
        result = models[key].predict(rgb)
        results.append({**model_info(key), **result})
    return {
        "filename": image.filename,
        "format": image_format,
        "width": rgb.width,
        "height": rgb.height,
        "file_bytes": len(data),
        "jpeg_quality": jpeg_quality or None,
        "compressed_bytes": compressed_bytes,
        "results": results,
    }


SAMPLES = {
    "progan_real_goc.png": "ProGAN, ảnh thật",
    "progan_fake_goc.png": "ProGAN, ảnh giả",
    "stylegan2_fake_goc.png": "StyleGAN2, ảnh giả",
    "sdv1_fake_goc.png": "SD v1, ảnh giả",
}


@app.get("/api/samples")
def samples():
    return [{"file": f, "label": label, "url": f"/samples/{f}"} for f, label in SAMPLES.items()
            if (HERE / "samples" / f).is_file()]


app.mount("/samples", StaticFiles(directory=HERE / "samples"), name="samples")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")
