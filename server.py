import os
import io
import json
import torch
import torch.nn as nn
import torchvision.models as models
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
torch.set_num_threads(4)
device = torch.device("cpu")

# --- STARTUP SAFETY CHECKS ---
if not os.path.exists("class_mapping.json"):
    print("CRITICAL ERROR: class_mapping.json not found! Exiting.")
    exit(1)

if not os.path.exists("overfitted_paintings.pth"):
    print("CRITICAL ERROR: overfitted_paintings.pth weights not found! Exiting.")
    exit(1)

with open("class_mapping.json", "r") as f:
    idx_to_class = {int(k): v for k, v in json.load(f).items()}

print("Loading PyTorch model onto CPU...")
weights = models.EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 29)
model.load_state_dict(torch.load("overfitted_paintings.pth", map_location=device))
model.to(device)
model.eval()
print("AI ready.")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- ASYNC NETWORK WRAPPER ---
# --- ASYNC NETWORK WRAPPER ---
@app.post("/scan")
async def scan_frame_wrapper(file: UploadFile = File(...)):
    # Read bytes from the multipart form asynchronously
    data = await file.read()

    if not data:
        return {"match": False}

    # --- TERMINAL PING ---
    # This prints immediately to your console the millisecond the packet arrives
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

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, class_idx = torch.max(probabilities, dim=0)

    if confidence.item() > 0.94:
        actual_idx = int(class_idx.item())

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
            "confidence": round(confidence.item(), 3),
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
