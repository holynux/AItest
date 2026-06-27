#!/usr/bin/env python3
"""
CLI simple pour discuter avec un modèle Mistral léger en utilisant CUDA.

Utilisation:
    python chat.py [--model MODELE] [--device DEVICE]

Exemples:
    python chat.py
    python chat.py --model mistralai/Mistral-7B-v0.1
    python chat.py --device cuda:0
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


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
    args = parser.parse_args()

    # Vérification de CUDA
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("⚠️  CUDA n'est pas disponible. Utilisation du CPU.")
        args.device = "cpu"
    
    if torch.cuda.is_available():
        print(f"✅ CUDA disponible: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA non disponible, utilisation du CPU")

    print(f"\n🔍 Chargement du modèle: {args.model}")
    print(f"📍 Appareil: {args.device}\n")

    # Chargement du tokenizer et du modèle
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
            device_map=args.device if args.device.startswith("cuda") else "cpu"
        )
        
        if args.device.startswith("cuda"):
            model = model.to(args.device)
        
        print("✅ Modèle et tokenizer chargés avec succès!\n")
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        print("\nEssayez un modèle plus petit comme 'mistralai/Mistral-7B-v0.1'")
        print("Ou installez les dépendances: pip install -r requirements.txt")
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
