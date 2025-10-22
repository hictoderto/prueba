from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
import os

app = FastAPI()

@app.post("/procesar_imagen")
async def procesar_imagen(file: UploadFile = File(...)):
    # Leer la imagen desde la solicitud
    contents = await file.read()
    imagen_original = Image.open(io.BytesIO(contents))

    # Recortar la imagen (ajusta estos valores según la imagen original)
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
        time_labels.append(f"{horas:02d}:{minutos_restantes:02d}")

    niveles_glucosa = [int((4450 - y) / 12) for (x, y) in intersections]

    df = pd.DataFrame({
        "Tiempo": time_labels,
        "X": [p[0] for p in intersections],
        "Y": [p[1] for p in intersections],
        "Glucosa": niveles_glucosa
    })

    # Guardar el CSV
    output_csv = "medicion_glucosa.csv"
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    # Dibujar los puntos sobre la imagen
    image_contour = graf.copy()
    for (x, y) in intersections:
        cv2.circle(image_contour, (x, y), 5, (0, 0, 255), -1)

    output_image = "resultado.png"
    cv2.imwrite(output_image, image_contour)

    # Retornar el archivo CSV (podrías devolver también la imagen si quieres)
    return FileResponse(output_csv, filename="medicion_glucosa.csv")

