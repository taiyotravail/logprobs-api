# Détecteur d'hallucinations LLM

> Comment empêcher une IA d'halluciner avec assurance quand elle ne sait pas ?

API REST qui analyse les **logprobs** d'un LLM (`phi4-mini` via Ollama) pour
détecter les réponses peu fiables avant qu'elles n'atteignent l'utilisateur. Si
la confiance du modèle tombe sous un seuil défini, la réponse est bloquée et
remplacée par un message de transfert vers un opérateur humain.

L'API est pensée pour être intégrée à un site web et déployée gratuitement sur
Google Cloud Run : voir [Déploiement](#déploiement-gratuit-google-cloud-run).

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

## API REST

`api.py` expose le détecteur en HTTP. L'image Docker de l'API (`Dockerfile`)
est **autonome** : elle embarque Ollama, `phi4-mini` et le modèle spaCy. Aucun
service externe n'est nécessaire.

### Endpoints

| Méthode | Chemin     | Rôle                                                              |
| ------- | ---------- | ----------------------------------------------------------------- |
| `GET`   | `/health`  | Sonde de vie. À appeler au chargement du site pour réveiller le serveur. |
| `POST`  | `/analyze` | Pose la question, renvoie la réponse et la décision de blocage.   |
| `GET`   | `/docs`    | Documentation interactive (Swagger).                              |

Requête :

```json
{ "question": "Quelle est la capitale de la France ?", "threshold": 70 }
```

`threshold` est optionnel (défaut : 70). La question fait au maximum
500 caractères.

Réponse (extrait) :

```json
{
  "answer": "Paris est la capitale de la France.",
  "blocked": false,
  "raw_answer": "Paris est la capitale de la France.",
  "threshold": 70,
  "weakest_probability": 86.25,
  "weakest_token": "Paris",
  "critical_tokens": [{ "token": "Paris", "word": "Paris", "pos": "PROPN", "probability": 86.25 }],
  "tokens": [{ "token": "Paris", "probability": 86.25, "alternatives": [{ "token": "Paris", "probability": 86.25 }] }],
  "model": "phi4-mini:latest"
}
```

- `answer` est le texte à afficher : la réponse du modèle, ou le message de
  transfert humain si elle est bloquée.
- `raw_answer` contient la réponse brute, même bloquée : c'est ce qui permet
  la démo « protection on/off ». Avec `EXPOSE_BLOCKED_ANSWERS=false`, une
  réponse bloquée n'est plus envoyée du tout (`raw_answer` vaut `null`,
  `tokens` et `critical_tokens` sont vides).
- `tokens` permet de tracer le graphique des probabilités côté site.

Codes d'erreur : `422` requête invalide, `429` quota du jour atteint, `503`
modèle surchargé (réessayer), `502` réponse du modèle inexploitable.

### Lancer l'API en local

Prérequis : Python 3.10+ et [Ollama](https://ollama.com/download) installé et
lancé, avec le modèle téléchargé (~2,5 Go, une seule fois) :

```bash
ollama pull phi4-mini:latest
```

Puis, à la racine du projet :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download fr_core_news_md
uvicorn api:app --port 8000 --no-access-log
```

Ouvrir ensuite http://localhost:8000/docs pour tester avec Swagger.

Avec Docker, l'image est autonome : elle embarque Ollama et le modèle, soit
~4 Go.

```bash
docker build -t logprobs-api .
docker run --rm -p 8080:8080 logprobs-api
```

### Variables d'environnement de l'API

| Variable               | Défaut           | Rôle                                                      |
| ---------------------- | ---------------- | --------------------------------------------------------- |
| `API_ALLOWED_ORIGINS`  | `*`              | Origines autorisées (CORS), séparées par des virgules.    |
| `DAILY_ANALYSIS_LIMIT` | `100`            | Analyses max par jour, tous visiteurs confondus (0 = illimité). |
| `EXPOSE_BLOCKED_ANSWERS` | `true`         | Renvoyer le contenu d'une réponse bloquée (démo). `false` en production. |
| `MAX_QUESTION_LENGTH`  | `500`            | Longueur max d'une question (caractères).                 |
| `MAX_ANSWER_TOKENS`    | `150`            | Longueur max d'une réponse (tokens).                      |
| `LLM_TIMEOUT_SECONDS`  | `120`            | Délai max d'une inférence.                                |
| `DEFAULT_MODEL`        | `phi4-mini:latest` | Modèle Ollama utilisé.                                  |
| `OLLAMA_BASE_URL`      | `http://localhost:11434/v1` | URL d'Ollama (API compatible OpenAI).          |

### Confidentialité (RGPD)

- Les questions et réponses ne sont **jamais** écrites dans les logs. Seuls
  la durée, la décision de blocage et le type d'erreur le sont.
- Le log d'accès d'uvicorn, qui contient l'IP des visiteurs, est désactivé.
- Ollama tourne dans le conteneur et n'écoute qu'en local (`127.0.0.1`) : les
  questions ne sortent jamais du serveur.
- Un test automatisé vérifie qu'aucune question ne fuit dans les logs.

## Déploiement gratuit (Google Cloud Run)

Cloud Run ne facture que le temps de traitement des requêtes et **s'éteint
quand personne ne l'utilise**. Au premier appel après une période
d'inactivité (jusqu'à ~15 min), le serveur redémarre. Ce réveil devrait
prendre environ 1 à 2 min (estimation, à mesurer après le premier
déploiement). Les appels suivants sont rapides.

L'offre gratuite couvre chaque mois 180 000 vCPU-secondes et 360 000 Gio-secondes.
Avec 4 vCPU et 8 Gio, cela représente environ 12 h de calcul par mois. Cela
suffit largement pour une démo, mais chaque réveil consomme aussi environ une
minute. Le quota `DAILY_ANALYSIS_LIMIT` plafonne le nombre d'analyses, ce qui
limite la consommation sans la garantir à 100 %.

> ⚠️ Google exige un **compte de facturation** (carte bancaire) même pour
> l'offre gratuite. Le stockage de l'image (~4 Go dans Artifact Registry)
> dépasse les 0,5 Go gratuits et coûte quelques dizaines de centimes par
> mois. Créez une **alerte budgétaire** (Facturation → Budgets et alertes)
> pour être prévenu au moindre euro.

### Étapes (tout dans l'interface web, aucune ligne de commande)

Tout se fait depuis la console Google Cloud
(https://console.cloud.google.com). Google construit l'image directement à
partir du dépôt GitHub : rien n'est installé ni construit sur votre machine.
Les libellés peuvent varier légèrement selon la version de la console.

1. **Créer un projet** : sélecteur de projet en haut de la page →
   **Nouveau projet** → nom `logprobs-api` → **Créer**. Vérifiez ensuite que
   ce projet est bien sélectionné en haut de la page.
2. **Lier la facturation** : menu ☰ → **Facturation** → **Associer un compte
   de facturation** (en créer un si besoin, avec votre carte bancaire). En
   profiter pour créer l'alerte budgétaire : **Budgets et alertes** →
   **Créer un budget**.
3. **Ouvrir Cloud Run** : menu ☰ → **Cloud Run** → **Déployer un conteneur**
   → **Service**. Si la console propose d'activer des API (Cloud Run, Cloud
   Build, Artifact Registry), acceptez.
4. **Brancher le dépôt GitHub** : choisir **Déployer en continu à partir
   d'un dépôt (source ou fonction)** → **Configurer avec Cloud Build**.
   - Fournisseur : **GitHub** → **S'authentifier** et autoriser l'application
     Google Cloud Build sur le dépôt `logprobs-api`.
   - Dépôt : `taiyotravail/logprobs-api` → **Suivant**.
   - Branche : `^main$`.
   - Type de compilation : **Dockerfile**, emplacement `/Dockerfile`.
   - **Enregistrer**.
5. **Paramètres du service** :
   - Nom : `logprobs-api`.
   - Région : `europe-west1 (Belgique)` : les données restent dans l'UE.
   - Authentification : **Autoriser l'accès public** (anciennement
     « Autoriser les appels non authentifiés »), sinon le site ne pourra pas
     appeler l'API.
   - Facturation : **Basée sur les requêtes** (et **pas** « Basée sur les
     instances » / « CPU toujours alloué » : le CPU serait facturé en
     permanence).
   - Scaling du service : nombre minimal d'instances **0** (le serveur
     s'éteint quand il n'est pas utilisé, au prix du temps de réveil),
     nombre maximal **1** (plafonne la consommation).
6. Déplier **Conteneur(s), volumes, réseau, sécurité** :
   - Onglet **Paramètres** :
     - Port du conteneur : `8080`.
     - Mémoire : **8 Gio** — Processeur : **4**.
     - Délai avant expiration de la requête : `300` secondes.
     - Nombre maximal de requêtes simultanées par instance : `5`.
     - Cocher **Boost du processeur au démarrage** (accélère le réveil).
   - Onglet **Variables et secrets** → **Ajouter une variable**, trois fois :

     | Nom | Valeur |
     | --- | --- |
     | `API_ALLOWED_ORIGINS` | `https://mon-site.fr,https://www.mon-site.fr` |
     | `DAILY_ANALYSIS_LIMIT` | `100` |
     | `EXPOSE_BLOCKED_ANSWERS` | `true` |

     `EXPOSE_BLOCKED_ANSWERS=true` sert à la démo « protection on/off ». Pour
     un usage réel, passez-le à `false` : la réponse bloquée n'est alors plus
     envoyée au navigateur.
7. Cliquer sur **Créer**. Le premier build télécharge le modèle : comptez
   10 à 15 min. Pour suivre sa progression : menu ☰ → **Cloud Build** →
   **Historique**.
8. Une fois le build terminé, l'URL du service s'affiche en haut de la page
   Cloud Run (`https://logprobs-api-….run.app`). Pour tester, ouvrez
   `https://logprobs-api-….run.app/health` dans le navigateur : la page doit
   répondre en JSON (1 à 2 min d'attente si le serveur dormait).

**Redéployer** : chaque `git push` sur la branche `main` reconstruit et
redéploie automatiquement le service. Pour modifier un paramètre (mémoire,
variables…) : Cloud Run → `logprobs-api` → **Modifier et déployer une
nouvelle révision**.

**Nettoyer les anciennes images** : chaque build stocke une nouvelle image de
~4 Go, et les anciennes restent facturées. Menu ☰ → **Artifact Registry** →
dépôt `cloud-run-source-deploy` : supprimez les anciennes versions à la main,
ou **Modifier le dépôt** → **Règles de nettoyage** pour ne garder que la plus
récente ([documentation](https://cloud.google.com/artifact-registry/docs/repositories/cleanup-policy)).

### Intégration dans le site

```js
const API_URL = "https://logprobs-api-….run.app";

// Dès le chargement de la page : réveille le serveur pendant que le
// visiteur tape sa question.
fetch(`${API_URL}/health`).catch(() => {});

async function analyser(question, seuil = 70) {
  const response = await fetch(`${API_URL}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, threshold: seuil }),
  });
  if (response.status === 429) throw new Error("Quota du jour atteint, réessayez demain.");
  if (!response.ok) throw new Error(`Erreur ${response.status}`);
  return response.json(); // { answer, blocked, raw_answer, tokens, ... }
}
```

Si le serveur dormait, la première réponse peut prendre 1 à 2 min. Affichez
un message du type « Réveil du serveur… » tant que la requête est en cours.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Les tests n'ont besoin ni d'Ollama ni du modèle spaCy : le LLM et
l'étiquetage grammatical sont simulés.

## Architecture

Découpage modulaire (cf. standards `CLAUDE.md`) :

```
.
├── api.py                   # API REST FastAPI (/health, /analyze)
├── detection_service.py     # orchestration LLM → POS → décision
├── confidence_analyzer.py   # logique métier : POS + maillon faible
├── llm_client.py            # client Ollama (API compat OpenAI)
├── usage_quota.py           # quota quotidien d'analyses (garde-fou de coût)
├── config.py                # constantes (URL, modèle, seuil, message, limites)
├── Dockerfile               # image autonome : Ollama + phi4-mini + API
├── start-api.sh             # démarrage du conteneur (Ollama puis uvicorn)
├── .dockerignore            # exclusions du build context
├── requirements.txt         # dépendances de l'API
├── requirements-dev.txt     # dépendances de test
├── pytest.ini               # configuration de pytest
└── tests/                   # tests unitaires
```

Les modules `confidence_analyzer` et `detection_service` ne dépendent pas de
FastAPI : ils sont réutilisables (batch, tests unitaires, autre framework web).

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

- Calibrer automatiquement le seuil sur un dataset annoté (vraies vs. fausses
  réponses).
- Ajouter un mode "streaming" pour bloquer la réponse dès qu'un token critique
  passe sous le seuil, sans attendre la fin de la génération.
- Tester d'autres modèles Ollama pour comparer les profils de confiance.
