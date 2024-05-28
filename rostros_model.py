# Importar las librerías necesarias
import os
import streamlit as st
import torch
from PIL import Image
import numpy as np
from io import BytesIO
from facenet_pytorch import InceptionResnetV1, MTCNN
import requests
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from datetime import datetime

# hide deprication warnings which directly don't affect the working of the application
import warnings
warnings.filterwarnings("ignore")

# Configuración de la página
st.set_page_config(
    page_title="Reconocimiento de Rostros",
    page_icon=":smile:",
    initial_sidebar_state='auto'
)

# Ocultar menú y pie de página de Streamlit
hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) # Oculta el código CSS de la pantalla, ya que están incrustados en el texto de rebajas. Además, permita que Streamlit se procese de forma insegura como HTML

device = torch.device('cpu')
# Cargar el modelo
@st.cache_resource
def load_encoder():
    encoder = InceptionResnetV1(pretrained='vggface2', classify=False, device=device).eval()
    return encoder

@st.cache_resource
def load_mtcnn():
    mtcnn = MTCNN(
        keep_all=True,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        post_process=False,
        image_size=160,
        device=torch.device('cpu')
    )
    return mtcnn

with st.spinner('Cargando modelos...'):
    encoder = load_encoder()
    mtcnn = load_mtcnn()

############################################
# Función para cargar y procesar la imagen
def cargar_imagen(url):
    page = requests.get(url)
    imagen = Image.open(BytesIO(page.content))
    imagen = imagen.convert('RGB')
    return imagen


def procesamiento_imagen(imagen):
    boxes, probs, landmarks = mtcnn.detect(imagen, landmarks=True)
    caras = mtcnn(imagen)
    
    embeddings = []
    for cara in caras:
        cara = cara.unsqueeze(0)  # Añadir una dimensión adicional
        embedding = encoder(cara).detach().cpu()
        embeddings.append(embedding)
    
    return embeddings, boxes


# Función para identificar rostros
def identificar(embedding_cara, embeddings):
    comparaciones = {}
    for nombre, emb_list in embeddings.items():
        min_dist = min(euclidean(embedding_cara.flatten(), emb.flatten()) for emb in emb_list)
        comparaciones[nombre] = min_dist

    nombre_identificado = min(comparaciones, key=comparaciones.get)
    return nombre_identificado


def identificar_multiples_rostros(imagen, embeddings):
    embeddings_imagen, boxes = procesamiento_imagen(imagen)
    nombres_caras = []

    for i, embedding in enumerate(embeddings_imagen):
        nombre = identificar(embedding, embeddings)
        nombres_caras.append((nombre, boxes[i]))

    return nombres_caras

# Cargar los embeddings de referencia
@st.cache_resource
def cargar_embeddings():
    # Cargar las imágenes y obtener los embeddings
    embeddings = {
        "Jair Acevedo": [
            procesamiento_imagen(cargar_imagen('https://raw.githubusercontent.com/kobbii3/Rostros/main/rostros_estudiantes/jair_acevedo/jair1.jpeg'))[0][0],
            procesamiento_imagen(cargar_imagen('https://raw.githubusercontent.com/kobbii3/Rostros/main/rostros_estudiantes/jair_acevedo/jair2.jpeg'))[0][0],
            procesamiento_imagen(cargar_imagen('https://raw.githubusercontent.com/kobbii3/Rostros/main/rostros_estudiantes/jair_acevedo/jair3.jpeg'))[0][0]
        ],
        "Cristian Parada": [
            procesamiento_imagen(cargar_imagen('https://raw.githubusercontent.com/kobbii3/Rostros/main/rostros_estudiantes/cristian_parada/cristian1.jpeg'))[0][0],
            procesamiento_imagen(cargar_imagen('https://raw.githubusercontent.com/kobbii3/Rostros/main/rostros_estudiantes/cristian_parada/cristian2.jpeg'))[0][0],
            procesamiento_imagen(cargar_imagen('https://raw.githubusercontent.com/kobbii3/Rostros/main/rostros_estudiantes/cristian_parada/cristian3.jpeg'))[0][0]
        ],
        "Camilo Sierra": [
            procesamiento_imagen(cargar_imagen('https://raw.githubusercontent.com/kobbii3/Rostros/main/rostros_estudiantes/camilo_sierra/c82d5bfb-f66c-42c1-a68b-8a07f67788cc.jfif'))[0][0],
            procesamiento_imagen(cargar_imagen('https://raw.githubusercontent.com/kobbii3/Rostros/main/rostros_estudiantes/camilo_sierra/c51ae578-dd9a-4722-a53a-7b722f720c1b.jfif'))[0][0],
            procesamiento_imagen(cargar_imagen('https://raw.githubusercontent.com/kobbii3/Rostros/main/rostros_estudiantes/camilo_sierra/c82d5bfb-f66c-42c1-a68b-8a07f67788cc.jfif'))[0][0]
        ],
        "Carlos Escobar": [
            procesamiento_imagen(cargar_imagen('https://raw.githubusercontent.com/kobbii3/Rostros/main/rostros_estudiantes/carlos_escobar/carlos1.jpg'))[0][0],
            procesamiento_imagen(cargar_imagen('https://raw.githubusercontent.com/kobbii3/Rostros/main/rostros_estudiantes/carlos_escobar/carlos2.jpg'))[0][0],
            procesamiento_imagen(cargar_imagen('https://raw.githubusercontent.com/kobbii3/Rostros/main/rostros_estudiantes/carlos_escobar/carlos3.jpg'))[0][0]
        ],
        "Robinson Leon": [
            procesamiento_imagen(cargar_imagen('https://raw.githubusercontent.com/kobbii3/Rostros/main/rostros_estudiantes/robinson_leon/robinson1.jpg'))[0][0],
            procesamiento_imagen(cargar_imagen('https://raw.githubusercontent.com/kobbii3/Rostros/main/rostros_estudiantes/robinson_leon/robinson2.jpg'))[0][0],
            procesamiento_imagen(cargar_imagen('https://raw.githubusercontent.com/kobbii3/Rostros/main/rostros_estudiantes/robinson_leon/robinson3.jpg'))[0][0]
        ],
        "Anghel Gutierrez": [
            procesamiento_imagen(cargar_imagen('https://raw.githubusercontent.com/kobbii3/Rostros/main/rostros_estudiantes/anghel_gutierrez/anghel1.jpg'))[0][0],
            procesamiento_imagen(cargar_imagen('https://raw.githubusercontent.com/kobbii3/Rostros/main/rostros_estudiantes/anghel_gutierrez/anghel2.jpg'))[0][0],
            procesamiento_imagen(cargar_imagen('https://raw.githubusercontent.com/kobbii3/Rostros/main/rostros_estudiantes/anghel_gutierrez/anghel3.jpg'))[0][0]
        ],
        "Camila Villamizar": [
            procesamiento_imagen(cargar_imagen('https://raw.githubusercontent.com/kobbii3/Rostros/main/rostros_estudiantes/camila_villamizar/camila1.jpg'))[0][0],
            procesamiento_imagen(cargar_imagen('https://raw.githubusercontent.com/kobbii3/Rostros/main/rostros_estudiantes/camila_villamizar/camila2.jpg'))[0][0],
            procesamiento_imagen(cargar_imagen('https://raw.githubusercontent.com/kobbii3/Rostros/main/rostros_estudiantes/camila_villamizar/camila3.jpg'))[0][0]
        ],
        "Carlos Rueda": [
            procesamiento_imagen(cargar_imagen('https://raw.githubusercontent.com/kobbii3/Rostros/main/rostros_estudiantes/carlos_rueda/carlos1.JPEG'))[0][0],
            procesamiento_imagen(cargar_imagen('https://raw.githubusercontent.com/kobbii3/Rostros/main/rostros_estudiantes/carlos_rueda/carlos2.JPEG'))[0][0],
            procesamiento_imagen(cargar_imagen('https://raw.githubusercontent.com/kobbii3/Rostros/main/rostros_estudiantes/carlos_rueda/carlos3.JPEG'))[0][0]
        ],
        "Emilton Hernandez": [
            procesamiento_imagen(cargar_imagen('https://raw.githubusercontent.com/kobbii3/Rostros/main/rostros_estudiantes/emilton_hernandez/emilton1.jpg'))[0][0],
            procesamiento_imagen(cargar_imagen('https://raw.githubusercontent.com/kobbii3/Rostros/main/rostros_estudiantes/emilton_hernandez/emilton2.jpg'))[0][0],
            procesamiento_imagen(cargar_imagen('https://raw.githubusercontent.com/kobbii3/Rostros/main/rostros_estudiantes/emilton_hernandez/emilton3.jpg'))[0][0]
        ],
        "Sofia Higuera": [
            procesamiento_imagen(cargar_imagen('https://raw.githubusercontent.com/kobbii3/Rostros/main/rostros_estudiantes/sofia_higuera/sofia1.jpg'))[0][0],
            procesamiento_imagen(cargar_imagen('https://raw.githubusercontent.com/kobbii3/Rostros/main/rostros_estudiantes/sofia_higuera/sofia2.jpg'))[0][0]
        ]
    }
    return embeddings

embeddings = cargar_embeddings()

# Función para registrar asistencia
def registrar_asistencia(nombres, fecha):
    #hoy = datetime.now().date()
    archivo_asistencia = 'asistencia.txt'

    # Leer contenido actual del archivo
    try:
        with open(archivo_asistencia, 'r') as file:
            contenido = file.read()
    except FileNotFoundError:
        contenido = ""

    # Buscar la sección de la fecha proporcionada en el archivo
    if f"Asistencia del {fecha}" in contenido:
        # Actualizar la sección existente
        partes = contenido.split(f"Asistencia del {fecha}\n")
        antes_fecha = partes[0]
        despues_fecha = partes[1].split("\n\n", 1)
        registro_actual = despues_fecha[0]
        resto_contenido = despues_fecha[1] if len(despues_fecha) > 1 else ""

        estudiantes_actuales = set(registro_actual.split("\n"))
        estudiantes_actuales.update(nombres)

        nuevo_registro = antes_fecha + f"Asistencia del {fecha}\n" + "\n".join(estudiantes_actuales) + "\n\n" + resto_contenido
    else:
        # Agregar nueva sección para la fecha de hoy
        nuevo_registro = contenido + f"Asistencia del {fecha}\n" + "\n".join(nombres) + "\n\n"

    # Guardar contenido actualizado en el archivo
    with open(archivo_asistencia, 'w') as file:
        file.write(nuevo_registro)

################################ INTERFAZ DE USUARIO ############################################

# Cargar la imagen del logo de la Unab
logoU = Image.open("logoUnab.png")

# Guardar la imagen en un buffer
import io
import base64
buffer = io.BytesIO()
logoU.save(buffer, format="PNG")
buffer.seek(0)
img_str = base64.b64encode(buffer.read()).decode('utf-8')


# Configuración de la barra lateral y la interfaz
with st.sidebar:
    st.image('rostro_icon.png')
    st.title("Reconocimiento de Rostros")
    st.subheader("Identificación de personas a partir de imágenes")
    confianza = st.slider("Seleccione el nivel de confianza %", 0, 100, 50) / 100

st.image('logo.png')
# HTML y CSS para organizar la imagen y el título en una sola línea
st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{img_str}" alt="logo" style="width: 240px; height: auto; margin-right: 10px;">
        <h1 style="margin: 0;">Universidad Autonoma de Bucaramanga</h1>
    </div>
    """, 
    unsafe_allow_html=True
)
#st.title("Universidad Autonoma de Bucaramanga")
st.write("Modelo funcional de detección de rostros individuales o en grupo con el objetivo de suplir la toma de asistencia en un aula.")
st.write("""
         # Detección de rostros
         """
         )

# Entrada de fecha manual
fecha_manual = st.date_input("Seleccione la fecha para registrar asistencia", datetime.now().date())

img_file_buffer = st.camera_input("Capture una foto para identificar una persona o un grupo")
uploaded_file = st.file_uploader("O suba una imagen", type=["jpg", "jpeg", "png"])


if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
elif uploaded_file is not None:
    image = Image.open(uploaded_file)
else:
    st.text("Por favor tome una foto o suba una imagen")
    image = None

if image is not None:
    st.image(image, use_column_width=True)
    
    # Procesar la imagen
    nombres_caras = identificar_multiples_rostros(image, embeddings)
    if nombres_caras:
        nombres_detectados = [nombre for nombre, _ in nombres_caras]
        registrar_asistencia(nombres_detectados, fecha_manual)

        for nombre, box in nombres_caras:
            x1, y1, x2, y2 = box.astype(int)
            recorte_cara = np.array(image)[y1:y2, x1:x2]
            st.image(recorte_cara, caption=nombre)
        st.success(f"Asistencia registrada para: {', '.join(nombres_detectados)}")
    else:
        st.text("No se detectó ningun rostro en la imagen.")
