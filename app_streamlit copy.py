import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np # Importado pois ultralytics.plot() retorna np array

# 1. Carregue o modelo salvo (fora da função principal para carregar apenas uma vez)
@st.cache_resource # Use st.cache_resource para modelos grandes
def load_yolo_model():
    return YOLO("runs/detect/train/weights/best.pt")

modelo = load_yolo_model()

st.title("Detecção de Neutrófilos em Exames de Sangue")
st.write("Faça o upload de uma imagem para identificar neutrófilos usando um modelo de IA.")

# 2. Widget para upload de arquivo
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Exibir a imagem original
    st.subheader("Imagem Original:")
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption="Imagem Original do Exame", use_column_width=True)

    # Botão para iniciar a predição
    if st.button("Detectar Neutrófilos"):
        with st.spinner("Processando imagem e detectando neutrófilos..."):
            # Ler a imagem enviada
            image_bytes = uploaded_file.getvalue()
            image = Image.open(io.BytesIO(image_bytes))

            # Fazer a predição
            # Nota: O limiar de confiança (conf=0.05) pode ser ajustado
            results = modelo.predict(image, conf=0.05)

            # Gerar a imagem com as detecções plotadas
            # results[0].plot() retorna um array numpy (BGR)
            plotted_image_np = results[0].plot()
            # Converter BGR para RGB para exibição correta com PIL/Streamlit
            plotted_image_rgb = Image.fromarray(plotted_image_np[..., ::-1])

            st.subheader("Resultado da Detecção:")
            st.image(plotted_image_rgb, caption="Neutrófilos Identificados", use_column_width=True)

            # Opcional: Mostrar detalhes das detecções (bounding boxes, confianças)
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