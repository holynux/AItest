# Mistral CLI Chat

Un programme CLI simple pour discuter avec des modèles Mistral en utilisant CUDA.

## Prérequis

- Python 3.8+
- Une carte graphique NVIDIA avec CUDA (recommandé)
- **Mémoire GPU suffisante** pour le modèle choisi (voir tableau ci-dessous)
- Au moins 20-30 Go d'espace disque libre pour le téléchargement et le cache

## 📊 Exigences matérielles

| Modèle | Paramètres | VRAM requise | RAM requise (CPU) | Recommandé pour |
|--------|------------|--------------|-------------------|-----------------|
| **Ministral-3-3B-Instruct-2512** | **3B** | **~3.5 Go** | **~7 Go** | **4 Go GPU** ✅ |
| Ministral-3-3B-Base-2512 | 3B | ~3.5 Go | ~7 Go | 4 Go GPU ✅ |
| Ministral-8B-Instruct-2410 | 8B | ~8 Go | ~16 Go | 10 Go GPU |
| Mistral-7B-v0.1 | 7B | ~14 Go | ~28 Go | 16 Go GPU |
| Mistral-7B-Instruct-v0.1 | 7B | ~14 Go | ~28 Go | 16 Go GPU |
| Mixtral-8x7B-v0.1 | 47B | ~48 Go | ~96 Go | 56 Go GPU |

> ⚠️ **Votre GPU (RTX 2050) a 4 Go de VRAM** - Utilisez **Ministral-3-3B-Instruct-2512** !

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/holynux/AItest.git
cd AItest

# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

### Commande de base *(pour votre RTX 2050 avec 4 Go VRAM)*
```bash
python chat.py --hf-token VOTRE_TOKEN
```

### Avec un modèle spécifique
```bash
# Pour votre configuration (4 Go VRAM) - RECOMMANDÉ
python chat.py --model mistralai/Ministral-3-3B-Instruct-2512 --hf-token VOTRE_TOKEN

# Alternative
python chat.py --model mistralai/Ministral-3-3B-Base-2512 --hf-token VOTRE_TOKEN
```

### Options disponibles
```bash
python chat.py --help
```

Options:
- `--model`: Nom du modèle Mistral (par défaut: mistralai/Ministral-3-3B-Instruct-2512)
- `--device`: Appareil à utiliser (cuda, cpu, cuda:0, etc.)
- `--hf-token`: Token HuggingFace pour des téléchargements plus rapides
- `--max-new-tokens`: Nombre maximum de nouveaux tokens (par défaut: 512)
- `--temperature`: Température pour la génération (par défaut: 0.7)
- `--local-model`: Utiliser uniquement les fichiers locaux (pas de téléchargement)
- `--download-only`: Télécharger le modèle et quitter (pour vérifier la connexion)

## 🎯 Pour votre configuration (RTX 2050 - 4 Go VRAM)

### Modèle recommandé

```bash
# Meilleur choix pour 4 Go VRAM
python chat.py --model mistralai/Ministral-3-3B-Instruct-2512 --hf-token VOTRE_TOKEN
```

### Alternative avec CPU
```bash
# Si vous avez des problèmes avec le GPU
python chat.py --model mistralai/Ministral-3-3B-Instruct-2512 --device cpu --hf-token VOTRE_TOKEN
```

## Authentification HuggingFace

### Méthode 1: Variable d'environnement *(Recommandé)*
```bash
export HF_TOKEN="votre_token_huggingface"
python chat.py
```

### Méthode 2: Argument CLI
```bash
python chat.py --hf-token votre_token_huggingface
```

### Méthode 3: Fichier .env
Créez un fichier `.env` dans le projet:
```
HF_TOKEN=votre_token_huggingface
```

> **Note**: Vous pouvez obtenir votre token sur [HuggingFace Settings](https://huggingface.co/settings/tokens)

## Résolution des problèmes

### 🔴 "not a valid model identifier"

Cela signifie que le nom du modèle est incorrect. **Utilisez ces noms valides :**

```bash
# ✅ Modèles valides pour votre GPU
python chat.py --model mistralai/Ministral-3-3B-Instruct-2512
python chat.py --model mistralai/Ministral-3-3B-Base-2512

# ❌ Modèles INVALIDES (n'existent pas)
python chat.py --model mistralai/TinyMistral-248M  # ❌ N'existe pas
python chat.py --model mistralai/Mistral-2.1B        # ❌ N'existe pas
```

### 🔴 "CUDA out of memory"

Votre GPU n'a pas assez de mémoire. Solutions:

```bash
# 1. Utiliser un modèle plus léger (déjà fait avec Ministral-3-3B)
python chat.py --model mistralai/Ministral-3-3B-Instruct-2512

# 2. Utiliser le CPU
python chat.py --model mistralai/Ministral-3-3B-Instruct-2512 --device cpu

# 3. Réduire la taille du contexte
python chat.py --model mistralai/Ministral-3-3B-Instruct-2512 --max-new-tokens 256
```

### 🟡 Téléchargement lent ou interrompu

```bash
# Télécharger d'abord le modèle
python chat.py --model mistralai/Ministral-3-3B-Instruct-2512 --download-only --hf-token VOTRE_TOKEN

# Puis l'utiliser en local
python chat.py --model mistralai/Ministral-3-3B-Instruct-2512 --local-model
```

### 📁 Emplacement du cache

Par défaut, HuggingFace stocke les modèles dans:
- Linux: `~/.cache/huggingface/hub/`

### 🔧 Optimiser PyTorch pour la mémoire

```bash
# Activer les segments expansibles pour éviter la fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Puis lancer le chat
python chat.py --model mistralai/Ministral-3-3B-Instruct-2512 --hf-token VOTRE_TOKEN
```

## Exemple de conversation réussi

```
$ python chat.py --model mistralai/Ministral-3-3B-Instruct-2512 --hf-token VOTRE_TOKEN
✅ CUDA disponible: NVIDIA GeForce RTX 2050
✅ Authentification HuggingFace réussie
   Connecté en tant que: hollynux
📊 Mémoire GPU: 4.00 Go total, 3.80 Go disponible

📥 Téléchargement du tokenizer...
✅ Tokenizer chargé avec succès!
📥 Téléchargement du modèle (cela peut prendre du temps)...
✅ Modèle chargé avec succès!

💬 Bienvenue dans le chat Mistral!
Tapez 'quit', 'exit' ou 'q' pour quitter.

Vous: Bonjour!
Mistral: Bonjour ! Comment puis-je vous aider aujourd'hui ?

Vous: quit
👋 Au revoir!
```

## Liste complète des modèles Mistral valides

### Modèles légers (compatibles 4 Go VRAM)
- `mistralai/Ministral-3-3B-Instruct-2512` ✅ **Recommandé**
- `mistralai/Ministral-3-3B-Base-2512`

### Modèles standards (nécessitent plus de VRAM)
- `mistralai/Mistral-7B-v0.1` (14 Go VRAM)
- `mistralai/Mistral-7B-Instruct-v0.1` (14 Go VRAM)
- `mistralai/Mistral-7B-Instruct-v0.2` (14 Go VRAM)
- `mistralai/Mistral-7B-Instruct-v0.3` (14 Go VRAM)

### Modèles Mixtes (nécessitent beaucoup de VRAM)
- `mistralai/Mixtral-8x7B-v0.1` (48 Go VRAM)
- `mistralai/Mixtral-8x7B-Instruct-v0.1` (48 Go VRAM)

## Contribution

Les contributions sont les bienvenues! Ouvrez une issue ou une pull request.
