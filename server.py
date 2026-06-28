import os
from fastapi import FastAPI, Response
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

# --- THE MULTITHREADING UNLOCKER ---
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    # These two headers create a "Cross-Origin Isolated" environment.
    # This proves to the mobile browser that the site is safe,
    # unlocking the SharedArrayBuffer required for CPU multithreading.
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    return response

# Ensure the videos directory exists
os.makedirs("videos", exist_ok=True)

# Mount the videos folder
app.mount("/videos", StaticFiles(directory="videos"), name="videos")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8282))
    print(f"Starting Secure Static Server on port {port}...")
    uvicorn.run("server:app", host="0.0.0.0", port=port)

# Mount the root directory
app.mount("/", StaticFiles(directory=".", html=True), name="frontend")
