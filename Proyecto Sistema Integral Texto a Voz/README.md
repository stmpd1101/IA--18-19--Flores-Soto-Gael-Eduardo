# Sistema Integral de Reconocimiento de Texto Manuscrito a Voz

**Alumno:** Flores Soto Gael Eduardo  
**Número de Control:** 20170661  
---

## Índice

1. **Descripción General**
2. **Funcionamiento del Programa**
   - `TextDetector`
   - `HandwritingRecognizer`
   - `TextPostProcessor`
   - `SpanishTTSSynthesizer`
   - `HandwrittenTextToSpeechPipeline`
3. **Ejecución desde la Línea de Comandos**
4. **Dependencias y Modelos**
5. **Limitaciones Actuales**

---

## 1. Descripción General

Este proyecto implementa un pipeline completo de inteligencia artificial que toma una imagen con texto manuscrito y genera como salida un archivo de audio en español leyendo el contenido reconocido.

El flujo general del sistema es:

Imagen manuscrita → Detección de texto → Reconocimiento (HTR) → Limpieza de texto → Síntesis de voz → Archivo de audio WAV

---

## 2. Funcionamiento del Programa

El programa está organizado en varias clases, cada una encargada de una etapa del pipeline.

### 2.1. `TextDetector` – Detección de regiones de texto

Esta clase recibe la imagen original y busca las zonas donde probablemente haya texto manuscrito.

Pasos principales:

- Convierte la imagen a escala de grises.
- Aplica binarización adaptativa para resaltar el texto sobre el fondo.
- Utiliza operaciones morfológicas para unir letras en bloques de texto.
- Encuentra contornos y calcula bounding boxes (x, y, ancho, alto) de las posibles líneas o palabras.
- Ordena las regiones de texto de arriba hacia abajo y de izquierda a derecha.

Resultado: una lista de recortes de la imagen, cada uno con una parte del texto manuscrito.

### 2.2. `HandwritingRecognizer` – Reconocimiento de texto manuscrito (HTR)

Esta clase convierte cada recorte de texto manuscrito en texto digital.

Características principales:

- Utiliza el modelo TrOCR de Microsoft (`microsoft/trocr-base-handwritten`) desde Hugging Face.
- Convierte cada recorte a formato PIL y luego a tensores para el modelo.
- El modelo genera una secuencia de caracteres que representa el texto manuscrito.
- El resultado de cada región es una línea de texto reconocida en formato string.

Resultado: una lista de líneas de texto reconocidas.

### 2.3. `TextPostProcessor` – Limpieza y postprocesamiento del texto

El texto que sale del modelo HTR puede contener errores, espacios extra o líneas vacías.  
Esta clase realiza un postprocesado básico:

- Elimina líneas vacías.
- Recorta espacios al inicio y al final.
- Une todas las líneas en un solo texto continuo.
- Normaliza espacios múltiples.

En esta etapa se podrían añadir en el futuro:

- Corrección ortográfica.
- Ajuste automático de puntuación.
- Reglas gramaticales específicas.

Resultado: un bloque de texto limpio y listo para ser leído por el motor de síntesis de voz.

### 2.4. `SpanishTTSSynthesizer` – Síntesis de voz en español

Esta clase genera el audio a partir del texto reconocido.

Características:

- Utiliza un modelo TTS en español de Hugging Face (por defecto `facebook/mms-tts-spa`).
- Convierte el texto en un waveform (señal de audio).
- Normaliza la señal para evitar saturación.
- Guarda el resultado como un archivo WAV con una frecuencia de muestreo de 16 kHz (por defecto).

Resultado: un archivo de audio `.wav` que contiene la lectura en voz alta del texto manuscrito.

### 2.5. `HandwrittenTextToSpeechPipeline` – Pipeline completo

Esta clase integra todas las etapas anteriores en un solo flujo:

1. Carga la imagen de entrada desde la ruta especificada.
2. Llama a `TextDetector` para obtener las regiones de texto.
3. Usa `HandwritingRecognizer` para transcribir cada región detectada.
4. Aplica `TextPostProcessor` para obtener el texto final limpio.
5. Envía ese texto a `SpanishTTSSynthesizer` para generar el archivo de audio.

Métodos principales:

- `process_image(image_path)`: procesa la imagen y devuelve el texto final reconocido.
- `image_to_speech(image_path, audio_output_path)`: ejecuta el pipeline completo y genera el archivo de audio.

---

## 3. Ejecución desde la Línea de Comandos

El archivo principal define una función `main()` que permite ejecutar todo el sistema desde la terminal.

Ejemplo de uso:

```bash
python manuscrito_a_voz.py ruta/a/imagen.jpg salida/audio.wav
