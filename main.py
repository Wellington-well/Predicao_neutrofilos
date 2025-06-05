from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, Response
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
import io
import numpy as np

# Carregue o modelo salvo
modelo = YOLO("runs/detect/train/weights/best.pt")

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile):
    # Ler a imagem enviada
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    # Fazer a predição
    results = modelo.predict(image, conf=0.05)
    

    boxes = results[0].boxes
    detections = {
        "boxes": boxes.xyxy.tolist(),  # Coordenadas das bounding boxes
        "confidences": boxes.conf.tolist(),  # Níveis de confiança
        "class_ids": boxes.cls.tolist(),  # IDs das classes
        "class_names": [results[0].names[int(i)] for i in boxes.cls]  # Nomes das classes
    }
    
    plotted_image = results[0].plot()  # Gera um array numpy
    img_pil = Image.fromarray(plotted_image[..., ::-1])  # Converte BGR (OpenCV) para RGB
    img_byte_arr = io.BytesIO()
    img_pil.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    #return JSONResponse(content=detections)
    return StreamingResponse(io.BytesIO(img_byte_arr), media_type="image/jpeg")  # retorna a foto com os neutófilos circulados