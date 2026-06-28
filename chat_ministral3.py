#!/usr/bin/env python3
"""
CLI optimisé pour discuter avec Ministral-3-3B-Base-2512 sur GPU sans offloading.

Utilisation:
    python chat_ministral3.py [--device DEVICE] [--hf-token TOKEN] [--quantize 4bit]

Exemples:
    python chat_ministral3.py
    python chat_ministral3.py --device cuda:0
    python chat_ministral3.py --hf-token votre_token_huggingface
    python chat_ministral3.py --quantize 4bit
"""

import argparse
import os
import sys
import time
import torch
import warnings
from huggingface_hub import login, whoami, constants, snapshot_download
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

# Ignorer les avertissements de dépréciation
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

# Configuration de HF_HOME pour stocker les modèles dans /home/hollynux/models
os.environ["HF_HOME"] = "/home/hollynux/models"

# Constantes pour Ministral-3-3B-Base-2512
MODEL_NAME = "mistralai/Ministral-3-3B-Base-2512"
MODEL_MEMORY_GB = 3.5  # Mémoire estimée pour le modèle en float16


def get_cache_dir():
    """Retourne le répertoire de cache HuggingFace (utilise HF_HOME si défini)"""
    return Path(os.environ.get("HF_HOME", constants.HF_HUB_CACHE))


def get_local_model_path():
    """Retourne le chemin local du modèle (spécifique à Ministral-3-3B-Base-2512)"""
    return get_cache_dir() / "models" / MODEL_NAME.replace("/", "--")


def check_model_exists_locally():
    """Vérifie si le modèle existe déjà localement dans HF_HOME"""
    local_dir = get_local_model_path()
    
    # Vérifier si le répertoire existe
    if not local_dir.exists():
        return False, None
    
    # Vérifier la présence des fichiers essentiels
    config_file = local_dir / "config.json"
    model_files = list(local_dir.glob("**/*.safetensors")) + list(local_dir.glob("**/*.bin"))
    
    if config_file.exists() and model_files:
        print(f"✅ Modèle trouvé localement dans: {local_dir}")
        return True, local_dir
    else:
        print(f"⚠️  Répertoire {local_dir} existe mais semble incomplet.")
        return False, local_dir


def check_gpu_memory(quantization=None):
    """Vérifie si le GPU a assez de mémoire pour Ministral-3-3B-Base-2512"""
    if torch.cuda.is_available():
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        available_memory_gb = total_memory_gb - (torch.cuda.memory_allocated(0) + torch.cuda.memory_reserved(0)) / (1024**3)
        
        print(f"💾 Mémoire GPU: {total_memory_gb:.2f} Go total, {available_memory_gb:.2f} Go disponible")
        
        # Réduire les exigences si quantification
        required_memory = MODEL_MEMORY_GB
        if quantization == "4bit":
            required_memory *= 0.4  # 4-bit utilise ~40% de la mémoire
        elif quantization == "8bit":
            required_memory *= 0.6  # 8-bit utilise ~60% de la mémoire
        
        if available_memory_gb < required_memory:
            print(f"⚠️  Mémoire GPU insuffisante pour {MODEL_NAME} (nécessite ~{required_memory:.1f} Go avec {quantization or 'pas de'} quantification)")
            return False
        return True
    return True  # Pas de GPU, on utilise le CPU (mais le modèle est optimisé pour GPU)


def check_disk_space():
    """Vérifie l'espace disque disponible pour Ministral-3-3B-Base-2512"""
    import shutil
    cache_dir = get_cache_dir()
    
    # Vérifier l'espace total
    total, used, free = shutil.disk_usage(cache_dir)
    free_gb = free / (1024**3)
    
    if free_gb < MODEL_MEMORY_GB * 2:  # On estime ~7 Go nécessaires pour le téléchargement
        print(f"⚠️  Espace disque insuffisant: {free_gb:.1f} Go disponibles, ~7 Go requis")
        return False
    return True


def download_model(quantization=None):
    """Télécharge Ministral-3-3B-Base-2512 avec gestion des erreurs"""
    local_dir = get_local_model_path()
    local_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Téléchargement de {MODEL_NAME} dans {local_dir}...")
    try:
        snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=str(local_dir),
            token=os.getenv("HF_TOKEN") or None,
            max_workers=4,
        )
        print("✅ Téléchargement terminé!")
        return local_dir
    except Exception as e:
        print(f"❌ Échec du téléchargement: {e}")
        raise


def load_model_and_tokenizer(device, quantization=None):
    """Charge le modèle et le tokenizer pour Ministral-3-3B-Base-2512"""
    # Vérifier d'abord si le modèle existe localement dans HF_HOME
    model_exists, local_dir = check_model_exists_locally()
    
    if not model_exists:
        print(f"⚠️  Modèle non trouvé dans {get_cache_dir()}. Téléchargement...")
        local_dir = download_model(quantization)
    
    # Charger le tokenizer depuis le répertoire local
    print("📖 Chargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(local_dir),
        use_auth_token=True if os.getenv("HF_TOKEN") else None
    )
    print("✅ Tokenizer chargé!")
    
    # Déterminer le dtype et device_map
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    device_map = device if device.startswith("cuda") else "cpu"
    
    # Charger le modèle avec AutoModel (compatible Ministral-3)
    print("🤖 Chargement du modèle...")
    try:
        if quantization:
            from transformers import BitsAndBytesConfig
            
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=False
                )
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
            
            model = AutoModel.from_pretrained(
                str(local_dir),
                dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,  # Nécessaire pour Ministral-3
                quantization_config=quantization_config
            )
        else:
            model = AutoModel.from_pretrained(
                str(local_dir),
                dtype=dtype,
                device_map=device_map,
                trust_remote_code=True
            )
        
        # Déplacer le modèle sur le device spécifié (sans offloading)
        if device.startswith("cuda"):
            model = model.to(device)
        
        print("✅ Modèle chargé avec succès!")
        return model, tokenizer
    except ImportError as e:
        if "bitsandbytes" in str(e):
            print("⚠️  bitsandbytes non installé. Installation...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "bitsandbytes", "accelerate", "--quiet"])
            print("✅ bitsandbytes installé. Veuillez relancer le script.")
            sys.exit(1)
        raise


def get_mistral_eos_token_id(tokenizer):
    """Retourne l'eos_token_id pour Ministral"""
    return tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2


def generate_response(model, tokenizer, messages, max_new_tokens=512, temperature=0.7):
    """Génère une réponse en utilisant une approche simple"""
    try:
        # Formater les messages
        prompt = ""
        for msg in messages:
            if msg['role'] == 'user':
                prompt += f"User: {msg['content']}\n"
            elif msg['role'] == 'assistant':
                prompt += f"Assistant: {msg['content']}\n"
        
        # Ajouter un prompt pour l'assistant
        prompt += "Assistant: "
        
        # Tokenizer le prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Déplacer les inputs sur le bon device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Obtenir l'eos_token_id
        eos_token_id = get_mistral_eos_token_id(tokenizer)
        
        # Générer la réponse
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                max_length=len(inputs['input_ids'][0]) + max_new_tokens,
                temperature=temperature,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=eos_token_id,
                eos_token_id=eos_token_id,
                early_stopping=True
            )
        
        # Décoder la réponse
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Nettoyer la réponse
        if "Assistant: " in response:
            response = response.split("Assistant: ")[-1]
        
        response = ' '.join(response.split())
        return response.strip()
    except Exception as e:
        print(f"❌ Erreur lors de la génération: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description=f"CLI pour discuter avec {MODEL_NAME} sur GPU sans offloading"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Appareil à utiliser (cuda, cuda:0, cpu)"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="Token HuggingFace pour le téléchargement"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Nombre maximum de nouveaux tokens à générer"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Température pour la génération"
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["4bit", "8bit"],
        default=None,
        help="Utiliser la quantification pour réduire la mémoire (4bit ou 8bit)"
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Télécharger le modèle et quitter"
    )
    args = parser.parse_args()

    # Vérification de CUDA
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("⚠️  CUDA n'est pas disponible. Utilisation du CPU.")
        args.device = "cpu"
    
    if torch.cuda.is_available():
        print(f"✅ CUDA disponible: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA non disponible, utilisation du CPU (non optimisé)")

    # Authentification HuggingFace
    if args.hf_token:
        try:
            login(token=args.hf_token)
            print("✅ Authentification HuggingFace réussie")
            try:
                user = whoami()
                print(f"   Connecté en tant que: {user['name']}")
            except:
                pass
        except Exception as e:
            print(f"⚠️  Échec de l'authentification HuggingFace: {e}")
    else:
        print("ℹ️  Pas de token HuggingFace trouvé. Utilisation des limites anonymes.")

    # Vérification de la mémoire GPU
    if args.device.startswith("cuda"):
        if not check_gpu_memory(args.quantize):
            print(f"\n❌ Arrêt: {MODEL_NAME} nécessite plus de mémoire GPU que disponible.")
            print("   Essayez avec --quantize 4bit ou --device cpu")
            return

    # Vérification de l'espace disque
    if not args.download_only:
        if not check_disk_space():
            print("❌ Espace disque insuffisant pour télécharger le modèle")
            return

    print(f"\n🚀 Chargement de {MODEL_NAME}")
    print(f"📍 Appareil: {args.device}")
    print(f"📁 Répertoire des modèles: {get_cache_dir()}")
    if args.quantize:
        print(f"🔢 Quantification: {args.quantize}")
    print()

    # Chargement du modèle et du tokenizer
    try:
        model, tokenizer = load_model_and_tokenizer(args.device, args.quantize)
        
        if args.download_only:
            print("✅ Téléchargement terminé. Vous pouvez maintenant utiliser le modèle localement.")
            return
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        print("\nSolutions possibles:")
        print("1. Vérifiez votre connexion internet.")
        print("2. Utilisez --hf-token pour un téléchargement plus rapide.")
        print("3. Essayez --device cpu si votre GPU n'a pas assez de mémoire.")
        return

    # Boucle de conversation
    print("💬 Bienvenue dans le chat avec Ministral-3-3B-Base-2512!")
    print("Tapez 'quit', 'exit' ou 'q' pour quitter.\n")

    messages = []
    while True:
        try:
            user_input = input("Vous: ")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Au revoir!")
                break
            
            if not user_input.strip():
                continue
            
            messages.append({"role": "user", "content": user_input})
            
            print("Ministral-3: ", end="", flush=True)
            assistant_response = generate_response(
                model, tokenizer, messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature
            )
            
            if assistant_response:
                print(assistant_response)
                messages.append({"role": "assistant", "content": assistant_response})
            else:
                print("❌ Échec de la génération de la réponse")
                break
            
        except KeyboardInterrupt:
            print("\n👋 Au revoir!")
            break
        except Exception as e:
            print(f"\n❌ Erreur: {e}")
            break


if __name__ == "__main__":
    main()
