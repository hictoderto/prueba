from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import PlainTextResponse
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io
from supabase import create_client, Client
import uuid
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()

@app.post("/procesar_imagen")
async def procesar_imagen(
    file: UploadFile = File(...),
    user_id: str = Form(...),  # se recibe desde el formulario HTTP
    id:str = Form(...),
    date:str = Form(...)
):
    # Leer la imagen desde la solicitud
    contents = await file.read()
    imagen_original = Image.open(io.BytesIO(contents))

    # Recortar la imagen (ajusta estos valores según tu gráfico)
    x_inicio, y_inicio, x_fin, y_fin = 27, 330, 580, 1120
    imagen_recortada = imagen_original.crop((x_inicio, y_inicio, x_fin, y_fin))

    # Escalar la imagen
    nueva_dimensiones = (3250, 4631)
    imagen_redimensionada = imagen_recortada.resize(nueva_dimensiones, Image.LANCZOS)

    # Convertir a formato OpenCV (BGR)
    image = np.array(imagen_redimensionada)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Eliminar ruido (líneas y fondo verde)
    gray_color = np.array([182, 182, 182])
    green_color = np.array([210, 235, 210])
    tolerance_gray = 20
    tolerance_green = 20

    lower_gray = np.clip(gray_color - tolerance_gray, 0, 255)
    upper_gray = np.clip(gray_color + tolerance_gray, 0, 255)
    mask_gray = cv2.inRange(image, lower_gray, upper_gray)

    lower_green = np.clip(green_color - tolerance_green, 0, 255)
    upper_green = np.clip(green_color + tolerance_green, 0, 255)
    mask_green = cv2.inRange(image, lower_green, upper_green)

    mask_combined = cv2.bitwise_or(mask_gray, mask_green)
    mask_combined = cv2.medianBlur(mask_combined, 5)
    graf = cv2.inpaint(image, mask_combined, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # Convertir a escala de grises y binaria
    gray = cv2.cvtColor(graf, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)

    height, width = binary.shape
    x_start = 254
    x_step = 2
    x_values = [x for x in range(x_start, min(x_start + 1500 * x_step, width), x_step)]
    y_search_limit = 3950

    intersections = []
    for x in x_values:
        if x >= width:
            continue
        for y in range(0, y_search_limit):
            if binary[y, x] == 255:
                new_y = min(y + 25, y_search_limit - 1)
                intersections.append((x, new_y))
                break

    # Crear etiquetas de tiempo y calcular glucosa
    time_labels = []
    for i, (x, y) in enumerate(intersections):
        minutos = i
        horas = minutos // 60
        minutos_restantes = minutos % 60
        time_labels.append(f"{date} {horas:02d}:{minutos_restantes:02d}")

    niveles_glucosa = [int((4450 - y) / 12) for (x, y) in intersections]

    # Crear DataFrame con las columnas requeridas
    df = pd.DataFrame({
        "id": range(1, len(intersections) + 1),
        "user_id": user_id,
        "timestamp": time_labels,
        "glucose_level": niveles_glucosa
    })

    # Guardar CSV con las columnas exactas
    output_csv = "medicion_glucosa.csv"
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    # Dibujar los puntos sobre la imagen
    image_contour = graf.copy()
    for (x, y) in intersections:
        cv2.circle(image_contour, (x, y), 5, (0, 0, 255), -1)

    output_image = "resultado.png"
    cv2.imwrite(output_image, image_contour)

    contenido_csv = output_csv

    # Nombre único para Supabase
    
    nombre_unico = f"{user_id}_{uuid.uuid4().hex}.csv"

    # Guardar localmente (opcional)
    local_csv = f"1.csv"
    df.to_csv(local_csv, index=False, encoding='utf-8-sig')



    try:
        with open(local_csv, "rb") as f:
            upload_result = supabase.storage.from_("csvhistorical").upload(nombre_unico, f,file_options={"content-type":"text/csv"})
        public_url = supabase.storage.from_('csvhistorical').get_public_url(nombre_unico)    
        # Obtener fecha y hora actual+
        print(id)
        ahora = datetime.now()
        # Formatear como "YYYY-MM-DD HH:MM:SS"
        fecha_formateada = ahora.strftime("%Y-%m-%d %H:%M:%S")
        response = (
        supabase.table("historical_graphs_csv")
        .insert({"id": id, "user_id": user_id,"csv_link":public_url,"create_at":fecha_formateada,"date_record":date})
        .execute()
)
        return PlainTextResponse(str(response.data))

    except Exception as e:
        return PlainTextResponse(f"Error al subir: {str(e)}")

    
