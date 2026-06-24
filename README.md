# Proyecto de Fine-Tuning de Stable Diffusion para Generación de Ilustraciones Estilo Libro Antiguo

## Descripción del Proyecto

Este proyecto implementa un flujo completo de fine-tuning de un modelo Stable Diffusion (v1-4) utilizando técnicas de transfer learning. El objetivo principal es adaptar el modelo base para generar imágenes con un estilo visual específico: ilustraciones de libros antiguos.

El proyecto incluye todo el proceso desde la preparación de datos y entrenamiento del modelo, hasta la generación de imágenes y despliegue del modelo fine-tuneado en Hugging Face Hub para uso público.

## Estructura del Repositorio

```
Entrega_Final_Modulo_7/
├── README.md                          # Este archivo
├── requirements.txt                   # Dependencias del proyecto
├── .env                              # Configuración de tokens
├── .gitignore                        # Archivos ignorados por Git
├── carga_modelo.py                   # Script para subir modelo a Hugging Face
├── generacion_imagenes.py            # Script para inferencia desde terminal
├── notebook_test.ipynb               # Notebook principal con todo el flujo
│
├── modelo_generador_v1/              # Modelo fine-tuneado local
│   ├── feature_extractor/
│   ├── safety_checker/
│   ├── scheduler/
│   ├── text_encoder/
│   ├── tokenizer/
│   ├── unet/
│   ├── vae/
│   └── model_index.json
│
├── Imagen_inicial/                    # Imágenes generadas con modelo base
│   └── football_player_inicial.png
│
├── Imagenes_finetuning_notebook/     # Imágenes generadas con modelo fine-tuneado local
│   └── football_player_finetuned.png
│
└── Imagenes_finetuning_hf/           # Imágenes generadas con modelo de Hugging Face
    └── football_player_finetuned_hf.png
```

## Flujo de Trabajo Completado

### 1. Configuración del Entorno
- Instalación de bibliotecas esenciales: `diffusers`, `transformers`, `accelerate`, `torch`
- Configuración del token de Hugging Face para acceso a modelos
- Verificación de disponibilidad de GPU para entrenamiento

### 2. Carga y Evaluación del Modelo Base
- Descarga del modelo pre-entrenado `CompVis/stable-diffusion-v1-4`
- Generación de imágenes iniciales con prompts simples para establecer línea base
- Ejemplo: "an image of a football player" → `Imagen_inicial/football_player_inicial.png`

### 3. Preparación del Dataset
- Selección de imágenes estilo libro antiguo para fine-tuning
- Procesamiento y aumento de datos para entrenamiento
- Creación de dataset personalizado con transformaciones adecuadas

### 4. Fine-Tuning del Modelo
#### 4.1 Fine-Tuning Local (notebook_test.ipynb)
- Ajuste de los pesos del modelo U-Net para adaptarse al nuevo estilo visual
- Configuración de hiperparámetros: learning rate, batch size, épocas
- Implementación de funciones de pérdida y optimización
- Guardado del modelo fine-tuneado en `modelo_generador_v1/`

#### 4.2 Fine-Tuning en Hugging Face
- Adaptación del modelo para distribución pública
- Subida del modelo fine-tuneado a Hugging Face Hub
- Modelo disponible en: [Victorh1397/stable-diffusion-v1-4-old-book-illustrations](https://huggingface.co/Victorh1397/stable-diffusion-v1-4-old-book-illustrations)

### 5. Evaluación y Comparación
- Generación de imágenes con el modelo fine-tuneado local
- Generación de imágenes con el modelo de Hugging Face
- Comparación visual entre modelo base y fine-tuneado
- Análisis de calidad y coherencia con el estilo objetivo

### 6. Automatización y Scripts

#### `carga_modelo.py`
```python
# Función para subir el modelo fine-tuneado a Hugging Face Hub
from huggingface_hub import HfApi

def upload_to_hf():
    api = HfApi()
    repo_id = "Victorh1397/stable-diffusion-v1-4-old-book-illustrations"
    api.upload_folder(
        folder_path="./modelo_generador_v1",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Subiendo pesos del modelo finetuneado"
    )
```

#### `generacion_imagenes.py`
```python
# Script de inferencia para generar imágenes desde terminal
import argparse
import torch
from diffusers import StableDiffusionPipeline

def main():
    parser = argparse.ArgumentParser(description="Generador de imágenes con modelo fine-tuneado.")
    parser.add_argument("--prompt", type=str, required=True, help="Descripción de la imagen")
    parser.add_argument("--output", type=str, default="ilustracion_generada.png", help="Archivo de salida")
    
    args = parser.parse_args()
    
    # Cargar modelo desde Hugging Face
    pipe = StableDiffusionPipeline.from_pretrained(
        "Victorh1397/stable-diffusion-v1-4-old-book-illustrations"
    )
    
    # Generar imagen
    imagen = pipe(args.prompt).images[0]
    imagen.save(args.output)
```

## Resultados y Comparaciones

### Imagen Base vs Fine-Tuned
| Modelo | Prompt | Resultado | Características |
|--------|--------|-----------|-----------------|
| **Base** | "an image of a football player" | `Imagen_inicial/football_player_inicial.png` | Estilo moderno, colores vivos, detalles contemporáneos |
| **Fine-Tuned Local** | "an image of a football player" | `Imagenes_finetuning_notebook/football_player_finetuned.png` | Estilo libro antiguo, textura papel, colores desaturados |
| **Fine-Tuned HF** | "an image of a football player" | `Imagenes_finetuning_hf/football_player_finetuned_hf.png` | Mismo estilo, generado desde modelo en la nube |

### Características del Estilo Logrado
- **Textura**: Simulación de papel antiguo y desgaste
- **Color**: Paleta desaturada, tonos sepia y vintage
- **Líneas**: Trazos similares a grabados o ilustraciones manuales
- **Composición**: Estética reminiscente de ilustraciones de libros del siglo XIX-XX

## Requisitos y Configuración

### Instalación de Dependencias
```bash
pip install -r requirements.txt
```

### Configuración del Entorno
1. Obtener un token de Hugging Face
2. Crear archivo `.env` con:
```
HF_TOKEN=tu_token_aqui
```

## Uso para Usuarios Finales

### 1. Generar Imágenes desde Terminal
```bash
# Generar una imagen con el modelo fine-tuneado
python generacion_imagenes.py --prompt "a beautiful castle on a hill" --output "castillo.png"

# Ejemplo con prompt específico
python generacion_imagenes.py --prompt "A knight in shining armor standing in a medieval courtyard" --output "knight.png"
```

### 2. Usar el Modelo desde Python
```python
from diffusers import StableDiffusionPipeline
import torch

# Cargar el modelo fine-tuneado
pipe = StableDiffusionPipeline.from_pretrained(
    "Victorh1397/stable-diffusion-v1-4-old-book-illustrations",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# Generar imagen
image = pipe("A pirate ship sailing on rough seas").images[0]
image.save("pirate_ship.png")
```

### 3. Integración con APIs
El modelo está disponible públicamente en Hugging Face y puede ser usado desde:
- Aplicaciones web con Gradio/Streamlit
- APIs REST con FastAPI/Flask
- Scripts de automatización
- Entornos de desarrollo como Google Colab

## Detalles Técnicos

### Arquitectura del Modelo
- **Base**: Stable Diffusion v1-4 (CompVis)
- **Componentes Modificados**: U-Net (fine-tuned)
- **Componentes Preservados**: VAE, Text Encoder, Scheduler
- **Parámetros Ajustados**: ~860M parámetros del U-Net

### Hiperparámetros de Fine-Tuning
- Learning Rate: 1e-4
- Batch Size: 2 (optimizado para entrenamiento en CPU)
- Épocas: 2 (debido a limitaciones de tiempo y recursos)
- Optimizador: AdamW
- Loss: Mean Squared Error (MSE)

### Recursos Computacionales Utilizados
**Entrenamiento Realizado:**
- **Hardware**: CPU (procesador sin GPU dedicada)
- **Batch Size**: 2 (optimizado para entrenamiento en CPU)
- **Tiempo de Entrenamiento Total**: Entre 4-5 horas para todo el proceso de fine-tuning
- **Memoria RAM**: 16GB mínimo recomendado

**Configuración del Código:**
El notebook `notebook_test.ipynb` está configurado para detectar automáticamente si hay GPU disponible:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)  # En este caso imprimió "cpu"
```

**Recomendaciones para Mejor Rendimiento:**
- **Entrenamiento**: GPU con al menos 8GB VRAM (reduciría el tiempo de entrenamiento significativamente)
- **Almacenamiento**: ~4GB para el modelo completo

## Limitaciones y Consideraciones

### Limitaciones del Modelo
1. **Estilo Específico**: Optimizado para ilustraciones estilo libro antiguo
2. **Resolución**: Genera imágenes de 512x512 píxeles
3. **Dataset Reducido**: El modelo fue entrenado con solo 50 imágenes debido a limitaciones computacionales (CPU)
4. **Prompts Específicos**: Funciona mejor con descripciones detalladas

### Consideraciones Éticas
1. **Uso Responsable**: Generación de contenido apropiado
2. **Atribución**: Reconocimiento de modelos base y datasets
3. **Transparencia**: Documentación clara de capacidades y limitaciones

## Mejoras Futuras

### Técnicas Avanzadas
1. **LoRA/LoHa**: Fine-tuning más eficiente en memoria
2. **ControlNet**: Control más preciso sobre composición
3. **Hypernetwork**: Ajustes más granulares del estilo
4. **DreamBooth**: Personalización con pocas imágenes

### Aplicaciones Prácticas
1. **Generación de Portadas**: Para libros históricos o de fantasía
2. **Ilustraciones Educativas**: Material didáctico con estilo vintage
3. **Arte Digital**: Creación de piezas con estética retro
4. **Juegos**: Assets para juegos con temática histórica

## Contribución y Contacto

Este proyecto es parte de la entrega final del Módulo 7 de Pontia. Para preguntas o colaboraciones:

- **Autor**: Víctor Méndez
- **Modelo Hugging Face**: [Victorh1397/stable-diffusion-v1-4-old-book-illustrations](https://huggingface.co/Victorh1397/stable-diffusion-v1-4-old-book-illustrations)
- **Licencia**: CreativeML OpenRAIL-M (modelo fine-tuneado), heredada del modelo base Stable Diffusion

## Referencias y Créditos

- **Stable Diffusion v1-4**: CompVis/StabilityAI
- **Diffusers Library**: Hugging Face
- **Dataset de Entrenamiento**: Imágenes de dominio público con estilo libro antiguo
- **Infraestructura**: Hugging Face Hub para distribución del modelo

---

**Nota**: Este proyecto demuestra capacidades avanzadas de fine-tuning de modelos de generación de imágenes, aplicando técnicas de deep learning para adaptar modelos generales a estilos visuales específicos, con aplicaciones prácticas en creatividad digital y automatización de contenido visual.