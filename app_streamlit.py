import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import torch

# ───────────────────────────────────────
# 1. Configurar dispositivo
# ───────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ───────────────────────────────────────
# 2. Carregar modelo treinado + otimizar
# ───────────────────────────────────────
@st.cache_resource
def load_yolo_model():
    model = YOLO("runs/detect/train/weights/best.pt")
    model.fuse()  # funde Conv+BN → mais rápido/estável
    model.to(device)  # envia para GPU (ou CPU)
    if device == 'cuda':
        model.half()  # half precision se GPU
    return model

modelo = load_yolo_model()

# ───────────────────────────────────────
# 3. Interface Streamlit
# ───────────────────────────────────────
st.title("Detecção de Neutrófilos em Exames de Sangue")
st.write("Faça o upload de uma imagem para identificar neutrófilos usando um modelo de IA otimizado.")

# Upload
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Exibir imagem original
    st.subheader("Imagem Original:")
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption="Imagem Original do Exame", use_column_width=True)

    # Botão de detecção
    if st.button("Detectar Neutrófilos"):
        with st.spinner("Processando imagem e detectando neutrófilos..."):
            # Ler bytes da imagem
            image_bytes = uploaded_file.getvalue()
            image = Image.open(io.BytesIO(image_bytes))

            # Inferência otimizada
            results = modelo.predict(
                image,
                imgsz=1280,    # resolução maior → melhor detecção
                conf=0.15,     # limiar de confiança mais baixo → maior recall
                iou=0.7,       # NMS mais permissivo
                augment=True,  # Test Time Augmentation
                half=(device == 'cuda'),
                device=device
            )

            # Desenhar detecções
            plotted_image_np = results[0].plot()
            plotted_image_rgb = Image.fromarray(plotted_image_np[..., ::-1])  # BGR → RGB

            st.subheader("Resultado da Detecção:")
            st.image(plotted_image_rgb, caption="Neutrófilos Identificados", use_column_width=True)

            # Adicionar botão de download
            buffered = io.BytesIO()
            plotted_image_rgb.save(buffered, format="PNG")
            st.download_button(
                label="Baixar Imagem com Detecções",
                data=buffered.getvalue(),
                file_name="detecao_neutrofilos.png",
                mime="image/png"
            )

            # Mostrar detalhes das detecções
            st.subheader("Detalhes das Detecções:")
            if len(results[0].boxes) > 0:
                df_detections = {
                    "Confiança": [f"{conf:.2f}" for conf in results[0].boxes.conf.tolist()],
                    "Classe ID": results[0].boxes.cls.tolist(),
                    "Nome da Classe": [results[0].names[int(i)] for i in results[0].boxes.cls],
                    "Coordenadas (x1, y1, x2, y2)": results[0].boxes.xyxy.tolist()
                }
                st.dataframe(df_detections)
            else:
                st.write("Nenhum neutrófilo detectado com a confiança mínima.")

else:
    st.info("Por favor, faça o upload de uma imagem para começar a detecção.")
