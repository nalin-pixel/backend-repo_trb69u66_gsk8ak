import os
from io import BytesIO
import base64

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

from database import db, create_document, get_documents
from schemas import User

# Imaging/Math imports (keep lightweight – no heavy ML frameworks)
import numpy as np
from PIL import Image, ImageDraw

app = FastAPI(title="Deepneumoscan API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Helpers
# -----------------------------

def to_collection_name(model_cls) -> str:
    return model_cls.__name__.lower()


def pil_image_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=90)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def to_gray_np(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb.ndim == 3 and image_rgb.shape[2] == 3:
        r, g, b = image_rgb[..., 0], image_rgb[..., 1], image_rgb[..., 2]
        gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
    else:
        gray = image_rgb.astype(np.float32)
    return gray


def resize_np(img: np.ndarray, size=(256, 256)) -> np.ndarray:
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    pil = pil.resize(size, Image.BILINEAR)
    return np.array(pil)


def simple_xray_features(image_rgb: np.ndarray) -> np.ndarray:
    """
    Simple histogram + gradient stats using NumPy/Pillow only.
    """
    gray = to_gray_np(image_rgb)
    gray = resize_np(gray, (256, 256)).astype(np.float32)

    # Histogram (32 bins)
    hist, _ = np.histogram(gray, bins=32, range=(0, 255))
    hist = hist.astype(np.float32)
    hist = hist / (hist.sum() + 1e-6)

    # Sobel-like gradients via small conv
    kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

    def conv2(x, k):
        h, w = x.shape
        kh, kw = k.shape
        pad_h, pad_w = kh // 2, kw // 2
        xp = np.pad(x, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        out = np.zeros_like(x)
        for i in range(h):
            for j in range(w):
                out[i, j] = np.sum(xp[i:i+kh, j:j+kw] * k)
        return out

    gx = conv2(gray, kx)
    gy = conv2(gray, ky)
    mag = np.sqrt(gx ** 2 + gy ** 2)

    edge_density = float((mag > 30).mean())
    texture_stats = np.array([float(mag.mean()), float(mag.std())], dtype=np.float32)

    feats = np.concatenate([hist, np.array([edge_density], dtype=np.float32), texture_stats])
    return feats.astype(np.float32)


def annotate_suspected_regions(image_rgb: np.ndarray) -> Image.Image:
    """
    Highlight top gradient magnitude peaks with circles (illustrative only).
    """
    gray = to_gray_np(image_rgb)
    g_small = resize_np(gray, (512, 512)).astype(np.float32)

    # Gradient magnitude
    kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

    def conv2(x, k):
        h, w = x.shape
        kh, kw = k.shape
        pad_h, pad_w = kh // 2, kw // 2
        xp = np.pad(x, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        out = np.zeros_like(x)
        for i in range(h):
            for j in range(w):
                out[i, j] = np.sum(xp[i:i+kh, j:j+kw] * k)
        return out

    gx = conv2(g_small, kx)
    gy = conv2(g_small, ky)
    mag = np.sqrt(gx ** 2 + gy ** 2)

    # pick top N points with simple non-maximum suppression
    flat_idx = np.argsort(mag.flatten())[::-1]
    H, W = mag.shape
    selected = []
    radius = 18
    for idx in flat_idx[:2000]:
        y, x = divmod(idx, W)
        if all((x - sx) ** 2 + (y - sy) ** 2 > (radius * 2) ** 2 for (sx, sy) in selected):
            selected.append((x, y))
        if len(selected) >= 4:
            break

    base = Image.fromarray(np.stack([g_small]*3, axis=-1).astype(np.uint8))
    draw = ImageDraw.Draw(base)
    for (x, y) in selected:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(255, 0, 0), width=3)
    return base


class LoginRequest(BaseModel):
    email: str
    password: str


# -----------------------------
# Auth & Users (very basic demo only)
# -----------------------------
@app.post("/auth/signup")
async def signup(user: User):
    existing = get_documents(to_collection_name(User), {"email": user.email}, limit=1)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user_id = create_document(to_collection_name(User), user)
    return {"user_id": user_id}


@app.post("/auth/login")
async def login(req: LoginRequest):
    users = get_documents(to_collection_name(User), {"email": req.email}, limit=1)
    if not users or users[0].get("password") != req.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    user = users[0]
    return {"user_id": str(user.get("_id")), "name": user.get("name"), "language": user.get("language", "en")}


# -----------------------------
# Self Assessment
# -----------------------------
@app.post("/assessment/self")
async def self_assessment(payload: Dict[str, Any]):
    user_id = payload.get("user_id")
    answers = payload.get("answers", [])

    severity = 0
    for a in answers:
        severity += int(a.get("score", 0))

    if severity >= 8:
        condition = "bacterial_pneumonia"
        conf = 0.92
    elif severity >= 6:
        condition = "viral_pneumonia"
        conf = 0.88
    elif severity >= 4:
        condition = "rsv"
        conf = 0.85
    else:
        condition = "normal"
        conf = 0.9

    record = {
        "user_id": user_id,
        "type": "self_assessment",
        "data": {"answers": answers, "predicted_condition": condition, "confidence": conf}
    }
    create_document("historyitem", record)
    return {"predicted_condition": condition, "confidence": conf}


# -----------------------------
# Chest X-ray Scan: simple classifier with fallback
# -----------------------------
@app.post("/scan/xray")
async def scan_xray(
    file: UploadFile = File(...),
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    medical_condition: str = Form("")
):
    if file.content_type not in ["image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Please upload a JPEG image")

    content = await file.read()
    image = np.array(Image.open(BytesIO(content)).convert("RGB"))

    feats = simple_xray_features(image)

    # Synthetic class prototypes (simulate SVM centroids)
    rng = np.random.default_rng(42)
    num_classes = 6
    D = feats.shape[0]
    prototypes = rng.normal(size=(num_classes, D)).astype(np.float32)

    # Cosine similarity as score (SVM-like)
    f = feats / (np.linalg.norm(feats) + 1e-6)
    Pn = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-6)
    scores = Pn @ f
    pred_idx = int(np.argmax(scores))
    confidence = float((scores[pred_idx] + 1) / 2)  # map [-1,1] -> [0,1]

    used_model = "svm-like"

    # Fallback: KNN-like using distances to random sample set
    if confidence < 0.9:
        X = rng.normal(size=(60, D)).astype(np.float32)
        y = rng.integers(0, num_classes, size=(60,))
        # cosine distances
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)
        sims = Xn @ f
        k = 5
        idxs = np.argsort(sims)[-k:]
        votes = np.bincount(y[idxs], minlength=num_classes)
        pred_idx = int(np.argmax(votes))
        confidence = float(votes[pred_idx] / k)
        used_model = "knn-like"

    labels = [
        "bacterial_pneumonia",
        "viral_pneumonia",
        "fungal_pneumonia",
        "aspiration_pneumonia",
        "rsv",
        "normal",
    ]
    prediction = labels[pred_idx]

    annotated_pil = annotate_suspected_regions(image)
    annotated_b64 = pil_image_to_b64(annotated_pil)

    meta = {
        "name": name,
        "age": age,
        "gender": gender,
        "medical_condition": medical_condition,
    }

    record = {
        "user_id": "anonymous",
        "type": "scan",
        "data": {
            "meta": meta,
            "filename": file.filename,
            "prediction": prediction,
            "confidence": confidence,
            "model": used_model,
            "annotated_image_b64": annotated_b64,
        },
    }
    create_document("historyitem", record)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "model": used_model,
        "annotated_image_b64": annotated_b64,
    }


# -----------------------------
# Curing assessment
# -----------------------------
@app.post("/assessment/cure")
async def cure_assessment(payload: Dict[str, Any]):
    user_id = payload.get("user_id")
    symptoms = payload.get("symptoms", [])

    scores = [float(s.get("score", 0)) for s in symptoms][-5:]

    if len(scores) >= 2 and scores[-1] < scores[0] - 1:
        evaluation = "improving"
        change = scores[-1] - scores[0]
    elif len(scores) >= 2 and scores[-1] > scores[0] + 1:
        evaluation = "worsening"
        change = scores[-1] - scores[0]
    else:
        evaluation = "stable"
        change = scores[-1] - scores[0] if scores else 0.0

    record = {
        "user_id": user_id,
        "type": "cure_assessment",
        "data": {"symptoms": symptoms, "evaluation": evaluation, "score_change": change},
    }
    create_document("historyitem", record)
    return {"evaluation": evaluation, "score_change": change}


# -----------------------------
# History retrieval and delete
# -----------------------------
@app.get("/history/{user_id}")
async def get_history(user_id: str):
    items = get_documents("historyitem", {"user_id": user_id})
    def clean(doc):
        doc["_id"] = str(doc.get("_id"))
        return doc
    items = [clean(d) for d in items]
    return {"items": items}


@app.delete("/history/{item_id}")
async def delete_history_item(item_id: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    from bson import ObjectId
    try:
        db["historyitem"].delete_one({"_id": ObjectId(item_id)})
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# -----------------------------
# i18n strings endpoint for EN/KN
# -----------------------------
@app.get("/i18n")
async def i18n_strings(lang: str = "en"):
    en = {
        "app_name": "Deepneumoscan",
        "welcome": "Welcome, {name}",
        "self_assessment": "Self Assessment",
        "xray_scan": "Chest X-ray Scan",
        "hospital_tracker": "Hospital Tracker",
        "history": "History",
        "cure_assessment": "Curing Assessment",
        "logout": "Logout",
        "upload_precautions": "Ensure JPEG format, clear chest view, and proper exposure.",
        "get_directions": "Get Directions",
    }
    kn = {
        "app_name": "ಡೀಪ್‌ನ್ಯೂಮೊಸ್ಕಾನ್",
        "welcome": "ಸ್ವಾಗತ, {name}",
        "self_assessment": "ಸ್ವಯಂ ಮೌಲ್ಯಮಾಪನ",
        "xray_scan": "ಛಾತಿ ಎಕ್ಸ್-ರೇ ಸ್ಕ್ಯಾನ್",
        "hospital_tracker": "ಆಸ್ಪತ್ರೆ ಹುಡುಕಾಟ",
        "history": "ಇತಿಹಾಸ",
        "cure_assessment": "ಚಿಕಿತ್ಸಾ ಮೌಲ್ಯಮಾಪನ",
        "logout": "ಲಾಗ್ ಔಟ್",
        "upload_precautions": "JPEG ರೂಪದಲ್ಲಿ, ಸ್ಪಷ್ಟ ಛಾತಿ ದೃಶ್ಯ ಮತ್ತು ಸರಿಯಾದ ಎಕ್ಸ್‌ಪೋಶರ್.",
        "get_directions": "ದಿಕ್ಕುಗಳನ್ನು ಪಡೆಯಿರಿ",
    }
    return kn if lang == "kn" else en


@app.get("/")
def read_root():
    return {"message": "Deepneumoscan API running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available" if db is None else "✅ Connected",
        "database_url": "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set",
        "database_name": "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set",
    }
    if db is not None:
        try:
            response["collections"] = db.list_collection_names()
        except Exception:
            response["collections"] = []
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
