#!/usr/bin/env python3
"""
CLI simple pour discuter avec un modèle Mistral léger en utilisant CUDA.

Utilisation:
    python chat.py [--model MODELE] [--device DEVICE] [--hf-token TOKEN]

Exemples:
    python chat.py
    python chat.py --model mistralai/Ministral-3-3B-Instruct-2512
    python chat.py --device cuda:0
    python chat.py --hf-token votre_token_huggingface
"""

import argparse
import os
import sys
import time
import torch
import warnings
import requests
from huggingface_hub import login, whoami, constants, snapshot_download, HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pathlib import Path

# Ignorer les avertissements de dépréciation de huggingface_hub
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")


def get_cache_dir():
    """Retourne le répertoire de cache HuggingFace"""
    return Path(constants.HF_HUB_CACHE)


def check_disk_space(model_name, required_space_gb=14):
    """Vérifie l'espace disque disponible"""
    import shutil
    cache_dir = get_cache_dir()
    
    # Vérifier l'espace total
    total, used, free = shutil.disk_usage(cache_dir)
    free_gb = free / (1024**3)
    
    if free_gb < required_space_gb:
        print(f"⚠️  Espace disque insuffisant: {free_gb:.1f} Go disponibles, {required_space_gb} Go requis")
        return False
    return True


def clear_cache_if_needed():
    """Nettoie le cache si nécessaire"""
    cache_dir = get_cache_dir()
    
    # Vérifier si le cache existe et est trop volumineux
    if cache_dir.exists():
        total_size = sum(f.stat().st_size for f in cache_dir.glob('**/*') if f.is_file())
        total_gb = total_size / (1024**3)
        
        if total_gb > 50:  # Plus de 50 Go de cache
            print(f"ℹ️  Cache HuggingFace: {total_gb:.1f} Go")
            print("   Pour nettoyer: huggingface-cli delete-cache")


def get_model_memory_requirements():
    """Retourne un dictionnaire des exigences mémoire pour tous les modèles"""
    return {
        # Modèles très légers (Ministral)
        "mistralai/Ministral-3-3B-Instruct-2512": 3.5,
        "mistralai/Ministral-3-3B-Base-2512": 3.5,
        "mistralai/Ministral-8B-Instruct-2410": 8,
        
        # Modèles standards
        "mistralai/Mistral-7B-v0.1": 14,
        "mistralai/Mistral-7B-Instruct-v0.1": 14,
        "mistralai/Mistral-7B-Instruct-v0.2": 14,
        "mistralai/Mistral-7B-Instruct-v0.3": 14,
        
        # Modèles Mixtes (Mixture of Experts)
        "mistralai/Mixtral-8x7B-v0.1": 48,
        "mistralai/Mixtral-8x7B-Instruct-v0.1": 48,
        "mistralai/Mixtral-8x22B-v0.1": 120,
    }


def check_gpu_memory(model_name):
    """Vérifie si le GPU a assez de mémoire pour le modèle"""
    model_memory_requirements = get_model_memory_requirements()
    
    if torch.cuda.is_available():
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        available_memory_gb = total_memory_gb - (torch.cuda.memory_allocated(0) + torch.cuda.memory_reserved(0)) / (1024**3)
        
        print(f"📊 Mémoire GPU: {total_memory_gb:.2f} Go total, {available_memory_gb:.2f} Go disponible")
        
        # Vérifier si le modèle est connu, sinon estimer à 14 Go (pour les modèles Mistral-7B par défaut)
        required_memory = model_memory_requirements.get(model_name, 14)
        
        if available_memory_gb < required_memory:
            print(f"⚠️  Mémoire GPU insuffisante pour {model_name} (nécessite ~{required_memory} Go)")
            print("   Modèles compatibles avec votre GPU:")
            for light_model, light_req in model_memory_requirements.items():
                if light_req <= available_memory_gb:
                    print(f"   ✅ {light_model} (~{light_req} Go)")
            return False
        return True
    return True  # Pas de GPU, on utilise le CPU


def check_internet_connection():
    """Vérifie la connexion internet"""
    try:
        # Tester la connexion à HuggingFace
        response = requests.get("https://huggingface.co", timeout=10)
        if response.status_code == 200:
            return True
        
        # Tester la connexion à l'API
        api = HfApi()
        api.whoami(token=os.getenv("HF_TOKEN"))
        return True
    except requests.exceptions.Timeout:
        print("⚠️  Timeout de connexion. Vérifiez votre connexion internet.")
    except requests.exceptions.ConnectionError:
        print("⚠️  Erreur de connexion. Vérifiez votre connexion réseau.")
    except Exception as e:
        print(f"⚠️  Erreur de connexion: {e}")
    return False


def download_model_with_retry(model_name, dtype, device_map, max_retries=5):
    """Télécharge le modèle avec gestion des erreurs et reprise"""
    from transformers import AutoModelForCausalLM
    
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            print(f"📥 Tentative {attempt + 1}/{max_retries} de téléchargement de {model_name}")
            
            # Vérifier la connexion avant de commencer
            if not check_internet_connection():
                if attempt < max_retries - 1:
                    wait_time = 60  # Attendre 1 minute avant de réessayer
                    print(f"⏳ Attente de {wait_time} secondes avant réessai...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception("Impossible de se connecter à HuggingFace Hub")
            
            # Utiliser snapshot_download, puis charger le modèle localement
            local_dir = get_cache_dir() / "models" / model_name.replace("/", "--")
            local_dir.mkdir(parents=True, exist_ok=True)
            
            # Télécharger avec la nouvelle API
            print("   Connexion à HuggingFace Hub...")
            snapshot_download(
                repo_id=model_name,
                local_dir=str(local_dir),
                token=os.getenv("HF_TOKEN") or None,
                # Options pour améliorer la stabilité
                max_workers=4,  # Réduire le nombre de workers pour éviter les timeouts
            )
            
            # Vérifier que les fichiers ont été téléchargés
            model_files = list(local_dir.glob("**/*.bin")) + list(local_dir.glob("**/*.safetensors"))
            if not model_files:
                raise Exception("Aucun fichier de modèle trouvé après téléchargement")
            
            print(f"   ✅ {len(model_files)} fichiers téléchargés")
            
            # Charger depuis le répertoire local
            model = AutoModelForCausalLM.from_pretrained(
                str(local_dir),
                dtype=dtype,
                device_map=device_map,
                local_files_only=True
            )
            return model
            
        except Exception as e:
            last_exception = e
            print(f"❌ Échec tentative {attempt + 1}: {str(e)}")
            
            # Attendre avant de réessayer avec un délai exponentiel
            if attempt < max_retries - 1:
                wait_time = 60 * (attempt + 1)  # 60s, 120s, 180s...
                print(f"⏳ Attente de {wait_time} secondes avant réessai...")
                time.sleep(wait_time)
    
    raise last_exception


def suggest_lighter_models():
    """Suggère des modèles plus légers"""
    print("\n💡 Modèles Mistral plus légers recommandés:")
    print("-" * 50)
    
    light_models = [
        ("mistralai/Ministral-3-3B-Instruct-2512", "3B paramètres", "~3.5 Go VRAM", "Recommandé pour 4 Go"),
        ("mistralai/Ministral-3-3B-Base-2512", "3B paramètres", "~3.5 Go VRAM", "Version de base"),
        ("mistralai/Mistral-7B-v0.1", "7B paramètres", "~14 Go VRAM", "Nécessite plus de mémoire"),
    ]
    
    for model, params, vram, desc in light_models:
        print(f"  • {model}")
        print(f"    - {params} - {vram} - {desc}")
    
    print("\n💡 Autres options:")
    print("  • Utilisez --device cpu (plus lent mais pas de limite de mémoire)")
    print("  • Essayez des modèles plus petits si disponibles")


def main():
    parser = argparse.ArgumentParser(description="CLI pour discuter avec un modèle Mistral")
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Ministral-3-3B-Instruct-2512",  # Modèle par défaut plus léger
        help="Nom du modèle Mistral à utiliser (par défaut: mistralai/Ministral-3-3B-Instruct-2512)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Appareil à utiliser (cuda, cpu, etc.)"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="Token HuggingFace pour des téléchargements plus rapides (optionnel)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Nombre maximum de nouveaux tokens à générer"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Température pour la génération"
    )
    parser.add_argument(
        "--local-model",
        action="store_true",
        help="Utiliser uniquement les fichiers locaux (pas de téléchargement)"
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Télécharger le modèle et quitter (pour vérifier la connexion)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Nombre maximum de tentatives de téléchargement (par défaut: 5)"
    )
    args = parser.parse_args()

    # Vérification de CUDA
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("⚠️  CUDA n'est pas disponible. Utilisation du CPU.")
        args.device = "cpu"
    
    if torch.cuda.is_available():
        print(f"✅ CUDA disponible: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA non disponible, utilisation du CPU")

    # Authentification HuggingFace si token fourni
    if args.hf_token:
        try:
            login(token=args.hf_token)
            print("✅ Authentification HuggingFace réussie")
            # Vérifier que le token est valide
            try:
                user = whoami()
                print(f"   Connecté en tant que: {user['name']}")
            except:
                pass
        except Exception as e:
            print(f"⚠️  Échec de l'authentification HuggingFace: {e}")
    else:
        print("ℹ️  Pas de token HuggingFace trouvé. Utilisation des limites anonymes.")
        print("   Pour de meilleurs performances, utilisez --hf-token ou exportez HF_TOKEN")

    # Vérification de la mémoire GPU
    if args.device.startswith("cuda"):
        if not check_gpu_memory(args.model):
            suggest_lighter_models()
            print("\n❌ Arrêt: Modèle trop grand pour votre GPU")
            return

    # Vérification de l'espace disque
    if not args.local_model:
        # Estimer l'espace disque nécessaire en fonction du modèle
        model_memory_requirements = get_model_memory_requirements()
        required_space = model_memory_requirements.get(args.model, 14)
        
        if not check_disk_space(args.model, required_space_gb=required_space):
            print("❌ Espace disque insuffisant pour télécharger le modèle")
            return
        
        clear_cache_if_needed()

    print(f"\n🔍 Chargement du modèle: {args.model}")
    print(f"📍 Appareil: {args.device}\n")

    # Chargement du tokenizer et du modèle
    try:
        # Charger le tokenizer d'abord (plus petit)
        print("📥 Téléchargement du tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            use_auth_token=True if args.hf_token else None
        )
        print("✅ Tokenizer chargé avec succès!")
        
        # Déterminer le dtype et device_map
        dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
        device_map = args.device if args.device.startswith("cuda") else "cpu"
        
        # Charger le modèle
        print("📥 Téléchargement du modèle (cela peut prendre du temps)...")
        
        if args.local_model:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                dtype=dtype,
                device_map=device_map,
                local_files_only=True
            )
        else:
            model = download_model_with_retry(
                args.model, dtype, device_map, max_retries=args.max_retries
            )
        
        if args.device.startswith("cuda"):
            model = model.to(args.device)
        
        print("✅ Modèle chargé avec succès!\n")
        
        # Si on ne fait que le téléchargement
        if args.download_only:
            print("✅ Téléchargement terminé. Vous pouvez maintenant utiliser --local-model")
            return
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Erreur lors du chargement du modèle: {error_msg}")
        
        # Détection spécifique des erreurs
        if "not a local folder and is not a valid model identifier" in error_msg:
            print("\n⚠️  Modèle non trouvé sur HuggingFace!")
            print("   Vérifiez le nom du modèle.")
            suggest_lighter_models()
        elif "CUDA out of memory" in error_msg or "out of memory" in error_msg.lower():
            print("\n⚠️  Problème de mémoire détecté!")
            suggest_lighter_models()
        elif "Connection" in error_msg or "Timeout" in error_msg or "RST" in error_msg:
            print("\n⚠️  Problème de connexion réseau détecté!")
            print("   Essayez ces solutions:")
            print("   1. Vérifiez votre connexion internet")
            print("   2. Essayez avec un VPN si vous êtes derrière un firewall")
            print("   3. Augmentez --max-retries (ex: --max-retries 10)")
            print("   4. Utilisez --download-only pour télécharger d'abord")
        
        print("\nSolutions possibles:")
        print("1. Vérifiez le nom du modèle (ex: mistralai/Ministral-3-3B-Instruct-2512)")
        print("2. Essayez un modèle plus léger")
        print("3. Utilisez --device cpu (plus lent mais pas de limite de mémoire)")
        print("4. Vérifiez votre connexion internet")
        return

    # Création du pipeline de chat
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if args.device.startswith("cuda") else -1,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=True,
        repetition_penalty=1.1
    )

    print("💬 Bienvenue dans le chat Mistral!")
    print("Tapez 'quit', 'exit' ou 'q' pour quitter.\n")

    # Boucle de conversation
    messages = []
    while True:
        try:
            user_input = input("Vous: ")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Au revoir!")
                break
            
            if not user_input.strip():
                continue
            
            # Ajout du message utilisateur
            messages.append({"role": "user", "content": user_input})
            
            # Préparation du prompt
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Génération de la réponse
            print("Mistral: ", end="", flush=True)
            outputs = pipe(
                prompt,
                return_full_text=False,
                streamer=None
            )
            
            # Extraction et affichage de la réponse
            assistant_response = outputs[0]['generated_text']
            print(assistant_response)
            
            # Ajout de la réponse à l'historique
            messages.append({"role": "assistant", "content": assistant_response})
            
        except KeyboardInterrupt:
            print("\n👋 Au revoir!")
            break
        except Exception as e:
            print(f"\n❌ Erreur: {e}")
            print("Vérifiez que vous avez assez de mémoire GPU ou essayez un modèle plus petit.")
            break


if __name__ == "__main__":
    main()
