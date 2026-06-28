import os
import mimetypes
from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- CRITICAL: Tell the server how to handle WASM files ---
mimetypes.add_type("application/wasm", ".wasm")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SECURITY HEADERS TO UNLOCK MULTITHREADING ---
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    return response

os.makedirs("videos", exist_ok=True)
app.mount("/videos", StaticFiles(directory="videos"), name="videos")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8282))
    print(f"Starting Secure Multithreaded Server on port {port}...")
    uvicorn.run("server:app", host="0.0.0.0", port=port)

# Mount the root directory
app.mount("/", StaticFiles(directory=".", html=True), name="frontend")
