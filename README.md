# Détecteur d'hallucinations LLM

> Comment empêcher une IA d'halluciner avec assurance quand elle ne sait pas ?

Démo Streamlit qui analyse les **logprobs** d'un LLM local pour détecter les
réponses peu fiables avant qu'elles n'atteignent l'utilisateur. Si la confiance
du modèle tombe sous un seuil défini, la réponse est bloquée et remplacée par un
message de transfert vers un opérateur humain.

Cible : **santé, legaltech, finance, assurance** — tous les secteurs où une IA
qui invente avec assurance n'est pas une option.

## Le problème

Un LLM ne dit jamais "je ne sais pas". Quand il manque d'information, il
hallucine avec la même assurance que lorsqu'il connaît la réponse. Exemple
réel avec `phi4-mini:latest` sur la question *"Quelle sera la population de la France en 2028 ?" * :

> La population de la France en 2028 est estimée à environ 67,5 millions.

Faux : Pourtant le modèle l'affirme sans broncher.

## L'approche

Plutôt que de demander au LLM s'il est sûr de lui (il dira toujours oui), on
inspecte directement les **probabilités de chaque token généré** :

1. On demande au modèle de retourner ses `logprobs` (top 3 alternatives par token).
2. On reconstruit le texte et on l'analyse avec spaCy (`fr_core_news_md`).
3. On filtre les tokens correspondant à des **mots grammaticalement critiques** :
   noms (`NOUN`), noms propres (`PROPN`), nombres (`NUM`), verbes (`VERB`).
4. Le **maillon faible** = la probabilité la plus basse parmi ces mots critiques.
5. Si maillon faible < seuil (par défaut **70 %**) → la réponse est bloquée.

Le filtre POS est crucial : la ponctuation, les articles et autres mots
fonctionnels ont souvent une confiance basse sans que ce soit problématique.
On ne veut bloquer que sur du contenu factuel.

## Lancement rapide avec Docker

C'est la méthode recommandée pour tester l'app sans toucher à un
environnement Python.

### Prérequis

1. **Docker Desktop** doit être installé et lancé sur ta machine :
   - Windows / macOS / Linux → [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
2. **Ollama** doit être installé sur ta machine (le LLM tourne en local pour
   garder le contrôle des données) :
   - [https://ollama.com/download](https://ollama.com/download)

### Étape 1 — Récupérer le modèle LLM

Dans un terminal, télécharge `phi4-mini` (~2.5 GB, à faire une fois) et
laisse Ollama tourner en arrière-plan :

```bash
ollama pull phi4-mini:latest
ollama serve
```

> Sur macOS et Windows, l'app Ollama installée lance `ollama serve`
> automatiquement au démarrage — la commande n'est nécessaire que sur Linux
> ou si le service n'est pas déjà actif.

### Étape 2 — Construire l'image Docker

Dans un **deuxième terminal**, à la racine du projet :

```bash
docker build -t logprobs-demo .
```

Le build force la plateforme `linux/amd64` au niveau du `Dockerfile`,
l'image produite tourne donc nativement sur Windows et Linux x86_64 (et via
émulation sur Mac Apple Silicon).

### Étape 3 — Lancer le conteneur

**Sur macOS et Windows (Docker Desktop)** :

```bash
docker run --rm -p 8501:8501 \
  -e DEFAULT_MODEL=phi4-mini:latest \
  --name logprobs-demo logprobs-demo
```

**Sur Linux** — il faut ajouter un flag pour que le conteneur résolve
`host.docker.internal` vers la machine hôte :

```bash
docker run --rm -p 8501:8501 \
  --add-host=host.docker.internal:host-gateway \
  -e DEFAULT_MODEL=phi4-mini:latest \
  --name logprobs-demo logprobs-demo
```

### Étape 4 — Ouvrir l'app

Rendez-vous sur **[http://localhost:8501](http://localhost:8501)**

Pour arrêter le conteneur : `Ctrl+C` dans le terminal, ou
`docker stop logprobs-demo` depuis un autre.

### Variables d'environnement supportées

| Variable           | Défaut (Docker)                        | Rôle                                    |
| ------------------ | -------------------------------------- | --------------------------------------- |
| `OLLAMA_BASE_URL`  | `http://host.docker.internal:11434/v1` | URL du serveur Ollama                   |
| `DEFAULT_MODEL`    | `phi4-mini:latest`                     | Modèle Ollama proposé dans la sidebar   |

---

## Installation locale (sans Docker)

### Prérequis

- Python 3.10+
- [Ollama](https://ollama.com/) installé localement
- Un modèle Ollama (par défaut `phi4-mini:latest` — petit donc hallucine beaucoup,
  parfait pour la démo)

### Installation

```bash
git clone <repo-url>
cd logprobs
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download fr_core_news_md
ollama pull phi4-mini:latest
```

### Lancement

Dans un terminal, démarrer le serveur Ollama :

```bash
ollama serve
```

Dans un autre terminal, lancer l'app :

```bash
streamlit run app.py
```

L'interface est accessible sur http://localhost:8501.

## Utilisation

1. Ajuster le seuil de confiance dans la sidebar (défaut : 70 %).
2. Saisir une question — idéalement piège (statistique précise, fait pointu).
3. Cliquer sur **Analyser**.
4. L'app affiche :
   - La réponse du modèle (ou le message de sécurité si seuil non respecté)
   - Le diagnostic : maillon faible, seuil, nombre de mots critiques
   - Un scatter plot des top-3 probabilités par token
   - Le détail des mots critiques avec leur catégorie grammaticale

## Architecture

Découpage modulaire (cf. standards `CLAUDE.md`) :

```
.
├── app.py                   # UI Streamlit (orchestration uniquement)
├── config.py                # constantes (URL, modèle, seuil, message)
├── llm_client.py            # client Ollama (API compat OpenAI)
├── confidence_analyzer.py   # logique métier : POS + maillon faible
├── plots.py                 # visualisations matplotlib
├── Dockerfile               # image de production (linux/amd64)
├── .dockerignore            # exclusions du build context
├── requirements.txt         # dépendances Python
└── logprobs.ipynb           # notebook d'exploration initial
```

Les modules `confidence_analyzer` et `plots` n'ont aucune dépendance vis-à-vis
de Streamlit : ils sont réutilisables (API REST, batch jobs, tests unitaires).

## Limites connues

- Le filtre POS dépend de la qualité du tagger spaCy ; sur du texte mal formé
  produit par un petit modèle, les tags peuvent être imprécis (mais ça reste
  exploitable car les *mauvais* tokens sont précisément ceux qui rendent la
  phrase incohérente).
- Les modèles fermés (Anthropic, OpenAI) exposent les logprobs avec des
  restrictions variables ; le code utilise l'API compat OpenAI d'Ollama mais
  doit être adapté pour d'autres backends.
- Le seuil optimal dépend du modèle et du domaine — à calibrer sur un jeu
  d'évaluation propre au cas d'usage.

## Pistes d'extension

- Empaqueter `confidence_analyzer` dans une API FastAPI pour intégration prod.
- Calibrer automatiquement le seuil sur un dataset annoté (vraies vs. fausses
  réponses).
- Ajouter un mode "streaming" pour bloquer la réponse dès qu'un token critique
  passe sous le seuil, sans attendre la fin de la génération.
- Tester d'autres modèles Ollama pour comparer les profils de confiance.
