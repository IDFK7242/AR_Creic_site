import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure the videos directory exists
os.makedirs("videos", exist_ok=True)

# Mount the videos folder
app.mount("/videos", StaticFiles(directory="videos"), name="videos")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8282))
    print(f"Starting Static File Server on port {port}...")
    uvicorn.run("server:app", host="0.0.0.0", port=port)

# Mount the root directory last to serve index.html, class_mapping.json, and paintings_int8.onnx
app.mount("/", StaticFiles(directory=".", html=True), name="frontend")
