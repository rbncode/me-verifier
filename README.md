# Instalación

1.  **Clona el repositorio:**
    ```bash
    git clone <repositorio>
    cd me-verifier
    ```

2.  **Crea y activa un entorno virtual:**
    ```bash
    py -m venv venv
    source venv/bin/activate
    ```

3.  **Instala las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configura las variables de entorno:**
    -   Copia `.env.example` a `.env`:
        ```bash
        cp .env.example .env
        ```
    -   Modifica el archivo `.env` según sea necesario.

## Configuración

La configuración se gestiona a través de variables en el archivo `.env`:

-   `MODEL_PATH`: Directorio de los modelos entrenados (por defecto: `models/`).
-   `THRESHOLD`: Umbral de confianza para la verificación (por defecto: `0.75`).
-   `PORT`: Puerto en el que se ejecutará la API (por defecto: `5000`).
-   `MAX_MB`: Tamaño máximo de archivo para las imágenes subidas (por defecto: `10`).

## Uso

Sigue estos pasos para entrenar y ejecutar el sistema:

1.  **Recortar Rostros**:
    ```bash
    python scripts/crop_faces.py
    ```
2.  **Generar Embeddings**:
    ```bash
    python scripts/embeddings.py
    ```
3.  **Entrenar el Modelo**:
    ```bash
    python train.py
    ```
4.  **Ejecutar la API**:
    Para desarrollo:
        ```bash
        python api/app.py
        ```

## Pruebas de la API

### Endpoint `/verify`

-   **Método**: `POST`
-   **Descripción**: Verifica un rostro en una imagen subida.
-   **Respuesta Exitosa (200 OK)**:
    ```json
    {
      "is_me":true,
      "model_version":"me-verifier-v1",
      "score":0.9934,
      "threshold":0.75,
      "timing_ms":3468.67
    }
    ```

### Pruebas de API

En caso de no querer probar con Postman, se puede utilizar `curl`.
```bash
curl -X POST -F "image=@/ruta/a/tu/imagen.jpg" http://localhost:5000/verify
```
