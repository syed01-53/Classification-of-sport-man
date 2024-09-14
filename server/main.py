from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import util
import numpy as np
import cv2
import base64
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def hello():
    return {"message": "hello"}

@app.post("/classify_image")
async def classify_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        result = util.classify_image(image_base64_data=None, file_path=None)  # Adjust this if needed

        return JSONResponse(content=result, headers={"Access-Control-Allow-Origin": "*"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, headers={"Access-Control-Allow-Origin": "*"}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    print("Starting Python FastAPI Server For Sports Celebrity Image Classification")
    util.load_saved_artifacts()
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
