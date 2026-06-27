# Mistral CLI Chat

Un programme CLI simple pour discuter avec des modèles Mistral en utilisant CUDA.

## Prérequis

- Python 3.8+
- Une carte graphique NVIDIA avec CUDA (recommandé)
- Au moins 14 Go de VRAM pour Mistral-7B

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

## Modèles recommandés

Pour une utilisation légère:
- `mistralai/Mistral-7B-v0.1` - 7B paramètres (nécessite ~14 Go VRAM)
- `mistralai/Mistral-7B-Instruct-v0.1` - Version optimisée pour les instructions

## Exemple de conversation

```
$ python chat.py
✅ CUDA disponible: NVIDIA GeForce RTX 4090
ℹ️  Pas de token HuggingFace trouvé. Utilisation des limites anonymes.
   Pour de meilleurs performances, utilisez --hf-token ou exportez HF_TOKEN

🔍 Chargement du modèle: mistralai/Mistral-7B-v0.1
📍 Appareil: cuda

✅ Modèle et tokenizer chargés avec succès!

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

### Téléchargement du modèle
Le premier lancement téléchargera le modèle (plusieurs Go).

### Avertissement HuggingFace Hub
Si vous voyez `Warning: You are sending unauthenticated requests to the HF Hub`:
- Ajoutez votre token HuggingFace comme décrit ci-dessus
- Cela améliorera les performances de téléchargement

## Contribution

Les contributions sont les bienvenues! Ouvrez une issue ou une pull request.
