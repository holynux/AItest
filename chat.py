#!/usr/bin/env python3
"""
CLI simple pour discuter avec un modèle Mistral léger en utilisant CUDA.

Utilisation:
    python chat.py [--model MODELE] [--device DEVICE] [--hf-token TOKEN]

Exemples:
    python chat.py
    python chat.py --model mistralai/Mistral-7B-v0.1
    python chat.py --device cuda:0
    python chat.py --hf-token votre_token_huggingface
"""

import argparse
import os
import sys
import time
import torch
from huggingface_hub import login, whoami, constants
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pathlib import Path


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


def download_model_with_retry(model_name, dtype, device_map, max_retries=3):
    """Télécharge le modèle avec gestion des erreurs et reprise"""
    from transformers import AutoModelForCausalLM
    
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            print(f"📥 Tentative {attempt + 1}/{max_retries} de téléchargement de {model_name}")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                device_map=device_map,
                resume_download=True,  # Reprise du téléchargement
                force_download=False,  # Utiliser le cache si disponible
                local_files_only=False,
                use_auth_token=True if os.getenv("HF_TOKEN") else None
            )
            return model
            
        except Exception as e:
            last_exception = e
            print(f"❌ Échec tentative {attempt + 1}: {str(e)}")
            
            # Attendre avant de réessayer
            if attempt < max_retries - 1:
                wait_time = 30 * (attempt + 1)  # 30s, 60s, 90s...
                print(f"⏳ Attente de {wait_time} secondes avant réessai...")
                time.sleep(wait_time)
    
    raise last_exception


def main():
    parser = argparse.ArgumentParser(description="CLI pour discuter avec un modèle Mistral")
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Nom du modèle Mistral à utiliser (par défaut: mistralai/Mistral-7B-v0.1)"
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

    # Vérification de l'espace disque
    if not args.local_model:
        if not check_disk_space(args.model):
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
            use_auth_token=True if args.hf_token else None,
            resume_download=True
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
                args.model, dtype, device_map
            )
        
        if args.device.startswith("cuda"):
            model = model.to(args.device)
        
        print("✅ Modèle chargé avec succès!\n")
        
        # Si on ne fait que le téléchargement
        if args.download_only:
            print("✅ Téléchargement terminé. Vous pouvez maintenant utiliser --local-model")
            return
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        print("\nSolutions possibles:")
        print("1. Vérifiez votre connexion internet")
        print("2. Essayez avec un token HuggingFace valide (--hf-token)")
        print("3. Utilisez --local-model si le modèle est déjà téléchargé")
        print("4. Essayez un modèle plus petit")
        print("5. Vérifiez l'espace disque disponible")
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
