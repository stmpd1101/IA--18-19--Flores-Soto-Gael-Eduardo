import os
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import AutoProcessor, AutoModelForTextToWaveform

from scipy.io.wavfile import write as wav_write


class TextDetector:
    """
    Detector sencillo de regiones de texto manuscrito usando OpenCV.
    Retorna bounding boxes (x, y, w, h).
    """

    def __init__(self, min_area: int = 1000):
        self.min_area = min_area

    def detect_text_regions(self, image_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        # Convertir a escala de grises
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Binarización adaptativa para resaltar texto
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            35, 11
        )

        # Operaciones morfológicas para unir letras en palabras / líneas
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        morph = cv2.dilate(thresh, kernel, iterations=1)

        # Buscar contornos externos
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area < self.min_area:
                continue
            boxes.append((x, y, w, h))

        # Ordenar cajas de arriba a abajo, izquierda a derecha
        boxes_sorted = sorted(boxes, key=lambda b: (b[1], b[0]))
        return boxes_sorted


class HandwritingRecognizer:
    """
    Reconocedor de texto manuscrito usando TrOCR de Microsoft.
    Modelo de HuggingFace: microsoft/trocr-base-handwritten
    """

    def __init__(self, model_name: str = "microsoft/trocr-base-handwritten", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[HTR] Cargando modelo {model_name} en {self.device}...")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def recognize_line(self, image_bgr: np.ndarray) -> str:
        """
        Recibe un recorte en BGR (OpenCV) y regresa el texto reconocido.
        """
        # Convertir a PIL RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Preprocesar y generar
        pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text.strip()


class TextPostProcessor:
    """
    Limpieza básica y post-procesamiento del texto.
    Aquí se pueden agregar corrección ortográfica y reglas gramaticales.
    """

    def clean_text_lines(self, lines: List[str]) -> str:
        # Eliminar líneas vacías, espacios extra y juntar en un solo texto
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            cleaned_lines.append(line)

        # Unir las líneas con espacios (puedes cambiarlo a '\n' si quieres mantener saltos de línea)
        full_text = " ".join(cleaned_lines)

        # Normalizaciones simples
        while "  " in full_text:
            full_text = full_text.replace("  ", " ")
        return full_text.strip()


class SpanishTTSSynthesizer:
    """
    Síntesis de voz en español usando un modelo TTS de HuggingFace.
    Ejemplo: facebook/mms-tts-spa
    """

    def __init__(self, model_name: str = "facebook/mms-tts-spa", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[TTS] Cargando modelo {model_name} en {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForTextToWaveform.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def synthesize_to_file(self, text: str, output_path: str, sample_rate: int = 16000):
        """
        Convierte texto a audio y lo guarda como archivo WAV.
        """
        if not text:
            raise ValueError("El texto para TTS está vacío.")

        print("[TTS] Generando audio...")
        inputs = self.processor(text=text, return_tensors="pt")
        # Mover tensores al dispositivo correcto
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            waveform = outputs.waveform  # Tensor shape: [1, num_samples]

        waveform_np = waveform.squeeze().cpu().numpy()

        # Normalizar a rango [-1, 1] por seguridad
        max_val = np.max(np.abs(waveform_np))
        if max_val > 0:
            waveform_np = waveform_np / max_val

        # Crear carpeta de salida si no existe
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Guardar archivo WAV
        wav_write(output_path, sample_rate, waveform_np.astype(np.float32))
        print(f"[TTS] Audio guardado en: {output_path}")


class HandwrittenTextToSpeechPipeline:
    """
    Pipeline integral: Imagen -> Detección de texto -> Reconocimiento HTR -> Post-procesamiento -> Audio TTS
    """

    def __init__(
        self,
        htr_model_name: str = "microsoft/trocr-base-handwritten",
        tts_model_name: str = "facebook/mms-tts-spa",
        device: Optional[str] = None,
    ):
        self.detector = TextDetector()
        self.recognizer = HandwritingRecognizer(htr_model_name, device=device)
        self.post_processor = TextPostProcessor()
        self.tts = SpanishTTSSynthesizer(tts_model_name, device=device)

    def process_image(self, image_path: str) -> str:
        """
        Procesa una imagen y devuelve el texto final reconocido (ya postprocesado).
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"No se encontró la imagen: {image_path}")

        print(f"[PIPELINE] Cargando imagen desde {image_path}")
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"No se pudo leer la imagen: {image_path}")

        # Detectar regiones de texto
        print("[PIPELINE] Detectando regiones de texto...")
        boxes = self.detector.detect_text_regions(image_bgr)
        print(f"[PIPELINE] Regiones detectadas: {len(boxes)}")

        lines = []
        for i, (x, y, w, h) in enumerate(boxes):
            roi = image_bgr[y : y + h, x : x + w]
            print(f"[PIPELINE] Reconociendo región {i + 1}/{len(boxes)}...")
            try:
                line_text = self.recognizer.recognize_line(roi)
            except Exception as e:
                print(f"[WARNING] Error reconociendo región {i + 1}: {e}")
                line_text = ""
            lines.append(line_text)

        # Post-procesar texto
        print("[PIPELINE] Post-procesando texto...")
        final_text = self.post_processor.clean_text_lines(lines)
        print("[PIPELINE] Texto reconocido:")
        print("------------------------------------")
        print(final_text)
        print("------------------------------------")

        return final_text

    def image_to_speech(self, image_path: str, audio_output_path: str):
        """
        Pipeline completo: imagen -> texto -> audio.
        """
        text = self.process_image(image_path)
        if not text:
            print("[PIPELINE] No se reconoció texto, no se generará audio.")
            return
        self.tts.synthesize_to_file(text, audio_output_path)


def main():
    """
    Ejemplo de uso desde línea de comandos.

    Ejecutar:
        python manuscrito_a_voz.py ruta/a/imagen.jpg salida/audio.wav
    """
    import argparse

    parser = argparse.ArgumentParser(description="Sistema Integral de Reconocimiento de Texto Manuscrito a Voz")
    parser.add_argument("image_path", type=str, help="Ruta de la imagen con texto manuscrito")
    parser.add_argument("audio_output_path", type=str, help="Ruta de salida para el archivo de audio WAV")
    parser.add_argument("--device", type=str, default=None, help="Dispositivo: 'cpu' o 'cuda' (si está disponible)")

    args = parser.parse_args()

    pipeline = HandwrittenTextToSpeechPipeline(device=args.device)
    pipeline.image_to_speech(args.image_path, args.audio_output_path)


if __name__ == "__main__":
    main()
