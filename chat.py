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
    python chat.py --model mistralai/Ministral-3-3B-Base-2512 --quantize 4bit
"""

import argparse
import os
import sys
import time
import torch
import warnings
import requests
from huggingface_hub import login, whoami, constants, snapshot_download, HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, MistralForCausalLM, AutoModel
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


def get_model_class(model_name):
    """Retourne la classe de modèle appropriée en fonction du nom"""
    # Pour les modèles Ministral-3, utiliser AutoModel (plus générique)
    if "Ministral-3" in model_name:
        return AutoModel  # ✅ AutoModel supporte Mistral3Config
    # Pour les autres modèles Mistral
    elif "Mistral" in model_name or "Mixtral" in model_name:
        return MistralForCausalLM
    else:
        return AutoModelForCausalLM


def check_gpu_memory(model_name, quantization=None):
    """Vérifie si le GPU a assez de mémoire pour le modèle"""
    model_memory_requirements = get_model_memory_requirements()
    
    if torch.cuda.is_available():
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        available_memory_gb = total_memory_gb - (torch.cuda.memory_allocated(0) + torch.cuda.memory_reserved(0)) / (1024**3)
        
        print(f"📊 Mémoire GPU: {total_memory_gb:.2f} Go total, {available_memory_gb:.2f} Go disponible")
        
        # Vérifier si le modèle est connu, sinon estimer à 14 Go (pour les modèles Mistral-7B par défaut)
        required_memory = model_memory_requirements.get(model_name, 14)
        
        # Réduire les exigences si quantification
        if quantization == "4bit":
            required_memory *= 0.4  # 4-bit utilise ~40% de la mémoire
        elif quantization == "8bit":
            required_memory *= 0.6  # 8-bit utilise ~60% de la mémoire
        
        if available_memory_gb < required_memory:
            print(f"⚠️  Mémoire GPU insuffisante pour {model_name} (nécessite ~{required_memory:.1f} Go avec {quantization or 'pas de'} quantification)")
            print("   Modèles compatibles avec votre GPU:")
            for light_model, light_req in model_memory_requirements.items():
                adjusted_req = light_req * (0.4 if quantization == "4bit" else 0.6 if quantization == "8bit" else 1.0)
                if adjusted_req <= available_memory_gb:
                    print(f"   ✅ {light_model} (~{adjusted_req:.1f} Go avec {quantization or 'pas de'} quantification)")
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


def download_model_with_retry(model_name, dtype, device_map, max_retries=5, quantization=None):
    """Télécharge le modèle avec gestion des erreurs et reprise"""
    # Obtenir la classe de modèle appropriée
    model_class = get_model_class(model_name)
    
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
            
            # Charger depuis le répertoire local avec la bonne classe de modèle
            # Pour Ministral-3, il faut utiliser trust_remote_code=True
            is_ministral3 = "Ministral-3" in model_name
            
            # Si quantification demandée, utiliser BitsAndBytesConfig
            if quantization:
                try:
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
                    
                    model = model_class.from_pretrained(
                        str(local_dir),
                        dtype=dtype,
                        device_map=device_map,
                        local_files_only=True,
                        trust_remote_code=is_ministral3,
                        quantization_config=quantization_config
                    )
                except ImportError:
                    print("⚠️  bitsandbytes non installé. Installation...")
                    import subprocess
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "bitsandbytes", "accelerate", "--quiet"])
                    print("✅ bitsandbytes installé. Veuillez réessayer.")
                    sys.exit(1)
            else:
                model = model_class.from_pretrained(
                    str(local_dir),
                    dtype=dtype,
                    device_map=device_map,
                    local_files_only=True,
                    trust_remote_code=is_ministral3
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


def generate_response(model, tokenizer, messages, max_new_tokens=512, temperature=0.7):
    """Génère une réponse en utilisant une approche simple et fiable"""
    try:
        # Formater les messages de manière simple et efficace
        # Pour Ministral-3, utiliser un format basique
        prompt = ""
        for msg in messages:
            if msg['role'] == 'user':
                prompt += f"[INST] {msg['content']} [/INST] "
            else:
                prompt += f"{msg['content']} "
        
        # Ajouter un espace et s'assurer que ça se termine bien
        prompt = prompt.strip() + " "
        
        # Tokenizer le prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Déplacer les inputs sur le bon device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Générer la réponse
        with torch.no_grad():
            if hasattr(model, 'generate'):
                # Méthode standard pour les modèles avec generate
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            else:
                # Pour Mistral3Model qui retourne Mistral3ModelOutputWithPast
                outputs = model(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                # Accéder à last_hidden_state
                outputs = outputs.last_hidden_state
                outputs = torch.argmax(outputs, dim=-1)
        
        # Décoder la réponse
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Nettoyer la réponse
        # Supprimer le prompt de la réponse
        if prompt in response:
            response = response[len(prompt):]
        
        # Supprimer les tokens spéciaux
        tokens_to_remove = ['[INST]', '[/INST]', '<s>', '</s>']
        for token in tokens_to_remove:
            response = response.replace(token, "")
        
        # Nettoyer les espaces multiples
        response = ' '.join(response.split())
        
        return response.strip()
    except Exception as e:
        print(f"❌ Erreur lors de la génération: {e}")
        return None


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
    print("  • Utilisez --quantize 4bit pour réduire la mémoire de ~60%")
    print("  • Utilisez --quantize 8bit pour réduire la mémoire de ~40%")


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
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["4bit", "8bit"],
        default=None,
        help="Utiliser la quantification pour réduire la mémoire (4bit ou 8bit)"
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
        if not check_gpu_memory(args.model, args.quantize):
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
    print(f"📍 Appareil: {args.device}")
    if args.quantize:
        print(f"🔢 Quantification: {args.quantize}")
    print()

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
            # Obtenir la classe de modèle appropriée
            model_class = get_model_class(args.model)
            is_ministral3 = "Ministral-3" in args.model
            
            # Si quantification demandée, utiliser BitsAndBytesConfig
            if args.quantize:
                try:
                    from transformers import BitsAndBytesConfig
                    
                    if args.quantize == "4bit":
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=dtype,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=False
                        )
                    elif args.quantize == "8bit":
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True
                        )
                    
                    model = model_class.from_pretrained(
                        args.model,
                        dtype=dtype,
                        device_map=device_map,
                        local_files_only=True,
                        trust_remote_code=is_ministral3,
                        quantization_config=quantization_config
                    )
                except ImportError:
                    print("⚠️  bitsandbytes non installé. Installation...")
                    import subprocess
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "bitsandbytes", "accelerate", "--quiet"])
                    print("✅ bitsandbytes installé. Veuillez réessayer.")
                    sys.exit(1)
            else:
                model = model_class.from_pretrained(
                    args.model,
                    dtype=dtype,
                    device_map=device_map,
                    local_files_only=True,
                    trust_remote_code=is_ministral3
                )
        else:
            model = download_model_with_retry(
                args.model, dtype, device_map, max_retries=args.max_retries, quantization=args.quantize
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
            print("   Essayez avec --quantize 4bit ou --device cpu")
            suggest_lighter_models()
        elif "Unrecognized configuration class" in error_msg:
            print("\n⚠️  Problème de configuration détecté!")
            print("   Solution 1: Mettez à jour transformers: pip install --upgrade transformers")
            print("   Solution 2: Utilisez --local-model avec les fichiers déjà téléchargés")
            print("   Solution 3: Essayez un autre modèle (ex: mistralai/Mistral-7B-v0.1)")
        elif "'dict' object has no attribute 'to_dict'" in error_msg:
            print("\n⚠️  Problème de configuration Ministral-3 détecté!")
            print("   Solution: Mettez à jour transformers: pip install --upgrade transformers")
            print("   Ou utilisez --local-model avec les fichiers déjà téléchargés")
        elif "Connection" in error_msg or "Timeout" in error_msg or "RST" in error_msg:
            print("\n⚠️  Problème de connexion réseau détecté!")
            print("   Essayez ces solutions:")
            print("   1. Vérifiez votre connexion internet")
            print("   2. Essayez avec un VPN si vous êtes derrière un firewall")
            print("   3. Augmentez --max-retries (ex: --max-retries 10)")
            print("   4. Utilisez --download-only pour télécharger d'abord")
        
        print("\nSolutions possibles:")
        print("1. Mettez à jour transformers: pip install --upgrade transformers")
        print("2. Vérifiez le nom du modèle (ex: mistralai/Ministral-3-3B-Instruct-2512)")
        print("3. Essayez un modèle plus léger")
        print("4. Utilisez --device cpu (plus lent mais pas de limite de mémoire)")
        print("5. Utilisez --quantize 4bit pour réduire la mémoire")
        print("6. Vérifiez votre connexion internet")
        return

    # Utiliser la génération directe au lieu du pipeline (pour éviter les bugs de version)
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
            
            # Génération de la réponse (sans pipeline)
            print("Mistral: ", end="", flush=True)
            assistant_response = generate_response(
                model, tokenizer, messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature
            )
            
            if assistant_response:
                print(assistant_response)
                # Ajout de la réponse à l'historique
                messages.append({"role": "assistant", "content": assistant_response})
            else:
                print("❌ Échec de la génération de la réponse")
                break
            
        except KeyboardInterrupt:
            print("\n👋 Au revoir!")
            break
        except Exception as e:
            print(f"\n❌ Erreur: {e}")
            print("Vérifiez que vous avez assez de mémoire GPU ou essayez un modèle plus petit.")
            break


if __name__ == "__main__":
    main()
