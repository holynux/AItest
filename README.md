# Mistral CLI Chat

Un programme CLI simple pour discuter avec des modèles Mistral en utilisant CUDA.

## Prérequis

- Python 3.8+
- Une carte graphique NVIDIA avec CUDA (recommandé)
- Au moins 14 Go de VRAM pour Mistral-7B
- **20-30 Go d'espace disque libre** pour le téléchargement et le cache

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

## Authentification HuggingFace (Recommandé)

Pour éviter les avertissements et bénéficier de meilleurs taux de téléchargement:

### Méthode 1: Variable d'environnement
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

## Résolution des problèmes de téléchargement

### 🔄 Téléchargement interrompu à 7 Go

C'est un problème courant. Voici les solutions:

#### 1. **Utiliser un token HuggingFace valide**
```bash
python chat.py --hf-token votre_token --model mistralai/Mistral-7B-v0.1
```

#### 2. **Télécharger d'abord le modèle séparément**
```bash
# Télécharger uniquement le modèle
python chat.py --hf-token votre_token --model mistralai/Mistral-7B-v0.1 --download-only

# Puis l'utiliser en local
python chat.py --model mistralai/Mistral-7B-v0.1 --local-model
```

#### 3. **Utiliser huggingface-cli**
```bash
# Installer huggingface_hub
pip install huggingface-hub

# Télécharger avec reprise
huggingface-cli download mistralai/Mistral-7B-v0.1 --resume-download --local-dir ./models
```

#### 4. **Vérifier l'espace disque**
```bash
# Vérifier l'espace disponible
df -h

# Nettoyer le cache HuggingFace si nécessaire
huggingface-cli delete-cache
```

#### 5. **Utiliser un miroir ou un cache local**
```bash
# Configurer un cache local personnalisé
export HF_HOME=/chemin/vers/votre/cache
export TRANSFORMERS_CACHE=/chemin/vers/votre/cache
```

### 📁 Emplacement du cache

Par défaut, HuggingFace stocke les modèles dans:
- Linux: `~/.cache/huggingface/hub/`
- Windows: `C:\Users\<user>\.cache\huggingface\hub\`
- Mac: `~/Library/Caches/huggingface/hub/`

### 🔧 Options avancées

#### Téléchargement avec wget/curl
Si les méthodes ci-dessus échouent, vous pouvez télécharger manuellement:

1. Trouvez l'URL du modèle sur [HuggingFace](https://huggingface.co/mistralai/Mistral-7B-v0.1)
2. Téléchargez les fichiers avec `wget` ou `curl`
3. Placez-les dans le bon répertoire de cache

#### Utiliser un proxy
Si vous êtes derrière un proxy:
```bash
export HTTP_PROXY=http://votre_proxy:port
export HTTPS_PROXY=http://votre_proxy:port
```

## Modèles recommandés

Pour une utilisation légère:
- `mistralai/Mistral-7B-v0.1` - 7B paramètres (nécessite ~14 Go VRAM)
- `mistralai/Mistral-7B-Instruct-v0.1` - Version optimisée pour les instructions

## Exemple de conversation

```
$ python chat.py --hf-token votre_token
✅ CUDA disponible: NVIDIA GeForce RTX 4090
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
- Essayez un modèle plus petit
- Réduisez `--max-new-tokens`
- Utilisez `--device cpu` (plus lent)

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
