import os
import io
import json
import torch
import numpy as np
import onnxruntime as ort
import torchvision.transforms as transforms
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

# --- CPU OPTIMIZATION ---
# Limit PyTorch's background thread usage for image preprocessing
torch.set_num_threads(4)

# --- STARTUP SAFETY CHECKS ---
if not os.path.exists("class_mapping.json") or not os.path.exists("paintings_int8.onnx"):
    print("CRITICAL ERROR: class_mapping.json or paintings_int8.onnx not found! Exiting.")
    exit(1)

with open("class_mapping.json", "r") as f:
    idx_to_class = {int(k): v for k, v in json.load(f).items()}

print("Booting ONNX Runtime Engine for CPU inference...")
# --- UPGRADED: Load the ONNX model using CPU Execution Provider ---
ort_session = ort.InferenceSession("paintings_int8.onnx", providers=['CPUExecutionProvider'])
input_name = ort_session.get_inputs()[0].name
print("AI ready.")

transform = transforms.Compose([
    transforms.Resize((224, 224)), # Mandatory for the static ONNX graph
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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
        # Catch corrupted network packets
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        print(f"Dropped frame / Bad image data: {e}")
        return {"match": False}

    # Prepare input tensor and convert to NumPy array for ONNX
    tensor = transform(image).unsqueeze(0).numpy()

    # --- UPGRADED: ONNX Inference ---
    ort_inputs = {input_name: tensor}
    outputs = ort_session.run(None, ort_inputs)[0] # Runs in highly optimized C++

    # Softmax via NumPy (bypassing PyTorch for math to save CPU overhead)
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

# --- SELF-INITIALIZING HTTPS SERVER ---
# --- CLOUD INITIALIZATION ---
if __name__ == "__main__":
    # Render and Railway pass the port they want you to bind to via the PORT env variable
    port = int(os.environ.get("PORT", 8282))
    print(f"Starting server on port {port}...")

    # Launch Uvicorn WITHOUT the ssl_keyfile and ssl_certfile
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port
    )

# --- CATCH-ALL FRONTEND MOUNT ---
# Must remain at the absolute bottom to prevent routing conflicts
app.mount("/", StaticFiles(directory=".", html=True), name="frontend")
