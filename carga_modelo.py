from huggingface_hub import HfApi

def upload_to_hf():
    api = HfApi()
    
    repo_id = "Victorh1397/stable-diffusion-v1-4-old-book-illustrations"
    carpeta_local = "./modelo_generador_v1" 
    
    print(f"Iniciando la subida del modelo a {repo_id}...")
    

    api.upload_folder(
        folder_path=carpeta_local,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Subiendo pesos del modelo finetuneado"
    )
    
    print("Subida completada con éxito")

if __name__ == "__main__":
    upload_to_hf()