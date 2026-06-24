
"""
Script de inferencia para generar imágenes utilizando el modelo fine-tuneado.
Uso básico: python generar_ilustracion.py --prompt "a beautiful castle on a hill" --output "castillo.png"
"""

import argparse
import torch
from diffusers import StableDiffusionPipeline

def setup_pipeline():
    """Descarga y configura el modelo desde Hugging Face."""
    model_id = "Victorh1397/stable-diffusion-v1-4-old-book-illustrations"
    print(f"Cargando modelo '{model_id}'...")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    return pipe, device

def main():
    # Configuramos los argumentos
    parser = argparse.ArgumentParser(description="Generador de imágenes con modelo fine-tuneado.")
    parser.add_argument(
        "--prompt", 
        type=str, 
        required=True, 
        help="El texto exacto que describe la imagen a generar."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="ilustracion_generada.png", 
        help="Nombre del archivo de salida (ej. mi_imagen.png)."
    )
    
    args = parser.parse_args()

    # Cargamos el modelo 
    pipe, device = setup_pipeline()
    
    # Generar imagen usando el prompt del usuario
    print(f"Generando imagen: '{args.prompt}'...")
    
    generator = torch.Generator(device).manual_seed(42)
    imagen = pipe(args.prompt, num_inference_steps=50, guidance_scale=7.5, generator=generator).images[0]
    
    # Guardamos el archivo en la misma ruta donde esta el script (a menos que se indique lo contrario)
    imagen.save(args.output)
    print(f"Imagen guardada como: {args.output}")

if __name__ == "__main__":
    main()