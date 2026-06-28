import os
import io
import json
import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DIRECTORY SETUP ---
os.makedirs("videos", exist_ok=True)
app.mount("/videos", StaticFiles(directory="videos"), name="videos")

# --- STARTUP SAFETY CHECKS ---
if not os.path.exists("class_mapping.json") or not os.path.exists("paintings_int8.onnx"):
    print("CRITICAL ERROR: class_mapping.json or paintings_int8.onnx not found! Exiting.")
    exit(1)

with open("class_mapping.json", "r") as f:
    idx_to_class = {int(k): v for k, v in json.load(f).items()}

print("Booting ONNX Runtime Engine for CPU inference...")
# --- Load the ONNX model using CPU Execution Provider ---
ort_session = ort.InferenceSession("paintings_int8.onnx", providers=['CPUExecutionProvider'])
input_name = ort_session.get_inputs()[0].name
print("AI ready.")

# --- ASYNC NETWORK WRAPPER ---
@app.post("/scan")
async def scan_frame_wrapper(file: UploadFile = File(...)):
    # Read bytes from the multipart form asynchronously
    data = await file.read()

    if not data:
        return {"match": False}

    # --- TERMINAL PING ---
    print("PING! Image received from phone.")

    # Pass off the blocking math to the Uvicorn threadpool
    return await run_in_threadpool(process_frame, data)

# --- SYNCHRONOUS INFERENCE WORKER ---
def process_frame(data: bytes):
    try:
        # Catch corrupted network packets and resize using pure PIL
        image = Image.open(io.BytesIO(data)).convert("RGB")
        image = image.resize((224, 224), Image.BILINEAR)
    except Exception as e:
        print(f"Dropped frame / Bad image data: {e}")
        return {"match": False}

    # --- PURE NUMPY PREPROCESSING ---
    # Convert to NumPy and scale pixels to [0, 1]
    img_array = np.array(image).astype(np.float32) / 255.0

    # Apply the exact EfficientNet Normalization math
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std

    # Reorder dimensions from (Height, Width, Channels) to (Channels, Height, Width)
    img_array = np.transpose(img_array, (2, 0, 1))

    # Add the batch dimension (equivalent to PyTorch's .unsqueeze(0))
    tensor = np.expand_dims(img_array, axis=0)

    # --- ONNX INFERENCE ---
    ort_inputs = {input_name: tensor}
    outputs = ort_session.run(None, ort_inputs)[0] # Runs in highly optimized C++

    # Softmax via NumPy
    exp_out = np.exp(outputs[0] - np.max(outputs[0]))
    probabilities = exp_out / exp_out.sum()

    confidence = np.max(probabilities)
    class_idx = np.argmax(probabilities)

    if confidence > 0.94:
        actual_idx = int(class_idx)

        # Safely get class to prevent KeyError
        painting_name = idx_to_class.get(actual_idx, "Unknown_Painting")
        if painting_name == "Unknown_Painting":
            return {"match": False}

        # VERIFY VIDEO EXISTS BEFORE CONFIRMING MATCH
        video_path = f"videos/{painting_name}.mp4"
        if not os.path.exists(video_path):
            print(f"WARNING: Detected '{painting_name}', but '{video_path}' is missing from the drive.")
            return {"match": False}

        return {
            "match": True,
            "painting_name": painting_name,
            "confidence": round(float(confidence), 3),
            "video_url": f"/{video_path}" # Relative URL for cross-device compatibility
        }

    return {"match": False}

# --- SELF-INITIALIZING HTTP SERVER ---
# --- CLOUD INITIALIZATION ---
if __name__ == "__main__":
    # Render and Railway pass the port they want you to bind to via the PORT env variable
    port = int(os.environ.get("PORT", 8282))
    print(f"Starting server on port {port}...")

    # Launch Uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port
    )

# --- CATCH-ALL FRONTEND MOUNT ---
# Must remain at the absolute bottom to prevent routing conflicts
app.mount("/", StaticFiles(directory=".", html=True), name="frontend")
