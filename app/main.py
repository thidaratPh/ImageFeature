import base64
import cv2
from fastapi import FastAPI, Request
import numpy as np
from app.HOG import getHog
app = FastAPI()

@app.get("/api/genhog")
async def getInformation(data : Request):
    json = await data.json()
    data = json['img']

    image_bytes = base64.b64decode(data)
    
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    
    image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)


    return {"hog": getHog(image).tolist()}
