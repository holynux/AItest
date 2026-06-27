# Mistral CLI Chat

Un programme CLI simple pour discuter avec des modèles Mistral en utilisant CUDA.

## Prérequis

- Python 3.8+
- Une carte graphique NVIDIA avec CUDA (recommandé)
- **Mémoire GPU suffisante** pour le modèle choisi (voir tableau ci-dessous)
- Au moins 20-30 Go d'espace disque libre pour le téléchargement et le cache

## 📊 Exigences matérielles

| Modèle | Paramètres | VRAM requise | RAM requise (CPU) |
|--------|------------|--------------|-------------------|
| TinyMistral-248M | 248M | ~1.5 Go | ~3 Go |
| Mistral-1.3B | 1.3B | ~3 Go | ~6 Go |
| Mistral-2.1B | 2.1B | ~4 Go | ~8 Go |
| Mistral-3.1B | 3.1B | ~5 Go | ~10 Go |
| **Mistral-7B** | **7B** | **~14 Go** | **~28 Go** |
| Mistral-7B-Instruct | 7B | ~14 Go | ~28 Go |
| Mixtral-8x7B | 47B | ~48 Go | ~96 Go |

> ⚠️ **Votre GPU a 4 Go de VRAM** - Mistral-7B ne fonctionnera pas. Utilisez un modèle plus léger!

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

### Commande de base
```bash
python chat.py
```

### Avec un modèle spécifique
```bash
# Pour un GPU avec 4 Go de VRAM
python chat.py --model mistralai/Mistral-2.1B

# Pour un GPU avec 6 Go de VRAM
python chat.py --model mistralai/Mistral-3.1B

# Pour un GPU avec 16 Go+ de VRAM
python chat.py --model mistralai/Mistral-7B-v0.1
```

### Options disponibles
```bash
python chat.py --help
```

Options:
- `--model`: Nom du modèle Mistral (par défaut: mistralai/Mistral-7B-v0.1)
- `--device`: Appareil à utiliser (cuda, cpu, cuda:0, etc.)
- `--hf-token`: Token HuggingFace pour des téléchargements plus rapides (optionnel)
- `--max-new-tokens`: Nombre maximum de nouveaux tokens (par défaut: 512)
- `--temperature`: Température pour la génération (par défaut: 0.7)
- `--local-model`: Utiliser uniquement les fichiers locaux (pas de téléchargement)
- `--download-only`: Télécharger le modèle et quitter (pour vérifier la connexion)
- `--quantize`: Utiliser la quantification (4bit ou 8bit) pour réduire la mémoire

## 🎯 Pour votre configuration (4 Go VRAM)

### Modèles recommandés

```bash
# Meilleur choix pour 4 Go VRAM
python chat.py --model mistralai/Mistral-2.1B --device cuda

# Alternative un peu plus légère
python chat.py --model mistralai/Mistral-1.3B --device cuda

# Très léger mais moins performant
python chat.py --model mistralai/TinyMistral-248M --device cuda

# Utilisation CPU (plus lent mais pas de limite de mémoire)
python chat.py --model mistralai/Mistral-7B-v0.1 --device cpu
```

### Avec quantification (expérimental)

```bash
# Réduit la mémoire de ~50%
python chat.py --model mistralai/Mistral-3.1B --quantize 4bit --device cuda
```

## Authentification HuggingFace (Recommandé)

Pour éviter les avertissements et bénéficier de meilleurs taux de téléchargement:

### Méthode 1: Variable d'environnement
```bash
export HF_TOKEN="votre_token_huggingface"
python chat.py --model mistralai/Mistral-2.1B
```

### Méthode 2: Argument CLI
```bash
python chat.py --hf-token votre_token_huggingface --model mistralai/Mistral-2.1B
```

### Méthode 3: Fichier .env
Créez un fichier `.env` dans le projet:
```
HF_TOKEN=votre_token_huggingface
```

> **Note**: Vous pouvez obtenir votre token sur [HuggingFace Settings](https://huggingface.co/settings/tokens)

## Résolution des problèmes de mémoire

### 🔴 "CUDA out of memory"

C'est le problème que vous avez rencontré. Solutions:

#### 1. **Utiliser un modèle plus léger** (Recommandé)
```bash
# Pour 4 Go VRAM
python chat.py --model mistralai/Mistral-2.1B

# Pour 3 Go VRAM
python chat.py --model mistralai/Mistral-1.3B

# Pour 2 Go VRAM
python chat.py --model mistralai/TinyMistral-248M
```

#### 2. **Utiliser le CPU**
```bash
python chat.py --model mistralai/Mistral-7B-v0.1 --device cpu
```
→ Plus lent mais pas de limite de mémoire

#### 3. **Réduire la taille du contexte**
```bash
python chat.py --model mistralai/Mistral-2.1B --max-new-tokens 128
```

#### 4. **Utiliser la quantification** (expérimental)
```bash
python chat.py --model mistralai/Mistral-3.1B --quantize 4bit --device cuda
```

#### 5. **Libérer de la mémoire GPU**
```bash
# Vérifier l'utilisation GPU
nvidia-smi

# Tuer les processus utilisant le GPU
# Trouvez le PID avec nvidia-smi, puis:
kill -9 <PID>
```

### 📁 Emplacement du cache

Par défaut, HuggingFace stocke les modèles dans:
- Linux: `~/.cache/huggingface/hub/`
- Windows: `C:\Users\<user>\.cache\huggingface\hub\`
- Mac: `~/Library/Caches/huggingface/hub/`

### 🔧 Options avancées

#### Optimiser PyTorch pour la mémoire
```bash
# Activer les segments expansibles pour éviter la fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Limiter la mémoire réservée
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

#### Téléchargement avec wget/curl
Si les méthodes ci-dessus échouent, vous pouvez télécharger manuellement:

1. Trouvez l'URL du modèle sur [HuggingFace](https://huggingface.co/mistralai)
2. Téléchargez les fichiers avec `wget` ou `curl`
3. Placez-les dans le bon répertoire de cache

## Modèles recommandés par configuration

### 🟢 2-4 Go VRAM
- `mistralai/TinyMistral-248M` - 248M paramètres
- `mistralai/Mistral-1.3B` - 1.3B paramètres
- `mistralai/Mistral-2.1B` - 2.1B paramètres

### 🟡 6-8 Go VRAM
- `mistralai/Mistral-3.1B` - 3.1B paramètres
- `mistralai/Mistral-7B-v0.1` - 7B paramètres (serré)

### 🔵 14-16 Go+ VRAM
- `mistralai/Mistral-7B-v0.1` - 7B paramètres
- `mistralai/Mistral-7B-Instruct-v0.1` - 7B paramètres optimisé

### 🔴 24 Go+ VRAM
- `mistralai/Mixtral-8x7B-v0.1` - 47B paramètres (8 experts)

## Exemple de conversation

```
$ python chat.py --model mistralai/Mistral-2.1B --device cuda
✅ CUDA disponible: NVIDIA GeForce GTX 1650
📊 Mémoire GPU: 4.00 Go total, 3.80 Go disponible
✅ Authentification HuggingFace réussie
   Connecté en tant que: votre_utilisateur

📥 Téléchargement du tokenizer...
✅ Tokenizer chargé avec succès!
📥 Téléchargement du modèle (cela peut prendre du temps)...
✅ Modèle chargé avec succès!

💬 Bienvenue dans le chat Mistral!
Tapez 'quit', 'exit' ou 'q' pour quitter.

Vous: Bonjour, comment ça va?
Mistral: Bonjour! Je suis une IA, donc je n'ai pas de sentiments, mais je suis là pour vous aider. Comment puis-je vous aider aujourd'hui?

Vous: quit
👋 Au revoir!
```

## Résolution des problèmes

### Erreur de mémoire
Si vous obtenez des erreurs de mémoire:
- Essayez un modèle plus léger
- Réduisez `--max-new-tokens`
- Utilisez `--device cpu` (plus lent)
- Utilisez `--quantize 4bit`

### CUDA non disponible
- Installez les pilotes NVIDIA
- Installez CUDA Toolkit
- Vérifiez avec `nvidia-smi`

### Problèmes réseau
- Vérifiez votre connexion internet
- Essayez avec un VPN si nécessaire
- Configurez un proxy si vous êtes derrière un firewall

## Contribution

Les contributions sont les bienvenues! Ouvrez une issue ou une pull request.
