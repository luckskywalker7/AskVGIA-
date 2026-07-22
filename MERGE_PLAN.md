# MERGE_PLAN.md — Fusion « nouveau design » × « logique réseau fonctionnelle »

> **Statut : ANALYSE UNIQUEMENT.** Aucun fichier n'a été modifié. Ce document est le
> plan à valider avant exécution. Contraintes respectées : on ne touche à aucun CSS,
> on ne touche pas à l'auth (login/signup), rien n'est encore codé.

---

## 0. Cartographie des fichiers

### Dashboard EXISTANT & FONCTIONNEL — `static/app/`
Servi par FastAPI (`api/main.py`) :
- Page : `GET /app/index.html` → `static/app/index.html`
- Assets : `GET /app-assets/js/*.js` et `/app-assets/css/*.css` (montés depuis `static/app/`)

| Fichier | Rôle | Lignes |
|---|---|---|
| `static/app/index.html` | Page dashboard « Ask » + tout le CSS inline spécifique | 544 |
| `static/app/js/api.js` | **LE cœur réseau + toute la logique applicative** (fetch, streaming, rendu, wiring DOM) | 1570 |
| `static/app/js/main.js` | UI pure : tabs, thème, dropdown user, drag&drop upload, toggle pricing, nav settings | 226 |
| `static/app/js/auth.js` | Auth (login/session) — **hors périmètre** | 202 |
| `static/app/css/style.css`, `pages.css` | Styles (design « basique » à abandonner) | — |

> ⚠️ Important : `api.js` n'est **pas** qu'une couche API. Il contient aussi tout le
> câblage DOM (`loadVideoList`, `selectVideo`, `sendQuestion`, `renderChunkPanel`,
> `loadTranscript`, `loadGraphViz`, `DOMContentLoaded`…). C'est le fichier à porter.

### Nouveau design — `project/`
Statique / mockup, aucun appel réseau :

| Fichier | Rôle | Lignes |
|---|---|---|
| `project/dashboard.html` | Structure HTML + contenu **factice codé en dur** | 196 |
| `project/dashboard.css` | **NE PAS TOUCHER** — visuel final | — |
| `project/dashboard.js` | UI pure : sidebar, top-tabs, video-tabs, player custom, thème, i18n FR/EN | 191 |
| `project/{login,signup,index,pricing,documentation}.html` | Autres pages — hors périmètre pour l'instant | — |

---

## 1. Ce que fait le dashboard EXISTANT (appels réseau)

Toutes les requêtes utilisent des **chemins relatifs** (`fetch('/videos')`) — ça marche
parce que le front est servi par le même serveur FastAPI (même origine). C'est le point
clé à centraliser (voir §6).

### 1.1 Endpoints réellement appelés par `api.js` / `main.js`

| Action UI | Méthode + endpoint | Réponse (forme) | Fonction JS |
|---|---|---|---|
| Lister les vidéos | `GET /videos` | `{ videos: [...], count }` | `API.listVideos()` → `loadVideoList()` |
| Détail vidéo (résumé, keywords, chapitres, url) | `GET /video/{id}` | voir §1.2 | `API.getVideo(id)` / `loadTranscript()` |
| Lecture vidéo (streaming) | `GET /video/{id}/stream` | flux vidéo (Range `206`) | `<video src>` via `file_url` |
| Chapitres riches | `GET /video/{id}/chapters` | `{ chapters: [...] }` | `loadTranscript()` (onglet Chapters) |
| Supprimer une vidéo | `DELETE /video/{id}` | `{...}` | `API.deleteVideo(id)` |
| Upload fichier | `POST /upload` (multipart, champ `file` [+`language`]) | `{ video_id }` | `API.uploadFile()` (XHR + progress) |
| (Ré)indexer | `POST /ingest` `{ video_id }` | `{...}` | `API.ingest(id)` |
| Statut d'indexation | `GET /ingestion/status/{id}` | `{ status, progress_pct, total_chunks }` | `startPolling()` |
| Poser une question (non-stream) | `POST /ask` `{ video_id, question, confirmed:true, top_k, vision }` | `{ answer, chunks/sources }` | `API.ask()` (fallback) |
| Poser une question (STREAM) | `POST /ask/stream` (SSE) | voir §1.3 | `API.askStream()` → `sendQuestion()` |
| Recherche brute (timeline) | `POST /search` `{ video_id, query, top_k }` | `{ results: [...] }` | `loadTranscript()` timeline |
| Graphe de connaissances | `GET /graph/video/{id}` | `{ nodes:[...], edges:[...] }` | `loadGraphViz()` (d3.js) |
| Frame image (vignette/preuve) | `GET /vision/frame/{chunk_id}` | image | `renderChunkPanel()`, thumbnails |
| 1ʳᵉ frame (thumbnail vidéo) | `GET /vision/frames/{id}?limit=1` | `{ frames:[...] }` | `_loadThumbnailUrl()` |
| Historique (global serveur) | `GET/POST/DELETE /history`, `DELETE /history/{id}` | `{ entries:[...] }` | `HistoryStore` |
| Santé | `GET /health` | `{...}` | `API.health()` |

> Auth (`/auth/*`) et pages Graph avancées (`/graph/ask`, `/graph/local`…) existent
> côté backend mais **ne sont pas** câblées dans ce dashboard → on les ignore.

### 1.2 Forme JSON de `GET /videos` et `GET /video/{id}` (source : `api/main.py:2157` & `:2218`)

```jsonc
// item de /videos.videos[]  et  /video/{id}
{
  "video_id": "20c6..._Title",
  "filename": "Title.mp4",
  "title": "Titre lisible (LLM ou nom de fichier)",
  "has_file": true,
  "total_chunks": 128,                 // 0 = pas encore indexée
  "file_url": "/video/{id}/stream",    // null si pas de fichier  → src du player
  "thumbnail_url": "/vision/frame/...",// null si pas de frame
  "summary": "…",                      // (métadonnées LLM)
  "keywords": ["…"], "topics": ["…"],
  "language": "en", "has_visual": true,
  "duration_seconds": 1476,
  "chapters": [ … ]                    // uniquement sur /video/{id}
}
```

### 1.3 Format du streaming SSE `POST /ask/stream` (source : `api/main.py:2660`)

Le serveur émet une séquence d'événements SSE ; `API.askStream()` (générateur async)
les parse :

```
event: meta
data: {"language_hint":"en", ...}                     ← métadonnées (ignorables au 1er merge)

data: {"token":"Le ", "done":false}                    ← tokens de réponse (streaming)
data: {"token":"sujet ", "done":false}
...
data: {"done":true, "citations":[ {start_time, ...} ]} ← fin + citations (timestamps sourcés)

event: results
data: [ {chunk_id, chunk_type, start_time, text, display_text, score, ...}, ... ]  ← chunks audio+visuels
```

Logique de rendu associée (dans `api.js`) :
- `token`/`text`/`answer` → prose accumulée (`fullText`) affichée dans la bulle réponse.
- événement `results` → `renderChunkPanel()` : panneau « Referenced segments » (frames
  visuelles cliquables via `/vision/frame/{chunk_id}` + segments audio), tri, badges
  `CITED`/`SOURCE`.
- événement `done.citations` → `_filterChunksByCitations()` re-filtre les chunks affichés.
- `_linkifyTimestamps()` transforme les `MM:SS` de la prose en liens cliquables →
  `seekVideoTo(sec)` qui fait `document.querySelector('video').currentTime = sec`.
- Chaque réponse est sauvegardée via `HistoryStore.save()`.

### 1.4 Gestion d'état / session déjà en place

- `State` (objet module) : `videos[]`, `currentVideo`, `pollingTimer`, `chaptersCache`,
  `graphCache`, `graphVideoId`.
- Persistance sélection : `localStorage['askvgia_selected_video']` (sticky) +
  `sessionStorage['selectedVideo']` (one-shot) + `?v=` URL param → résolus dans
  `DOMContentLoaded`.
- Thème : `main.js` bascule la classe `body.light-mode`.
  ⚠️ Le nouveau design utilise `document.documentElement.dataset.theme` + `localStorage['vgia_theme']` — **systèmes différents** (voir §4).
- Sidebar collapse : `localStorage['sidebarCollapsed']`.

---

## 2. Ce que contient le NOUVEAU design (`project/dashboard.*`)

### 2.1 Zones qui DOIVENT recevoir des données dynamiques (actuellement en dur)

| Zone (sélecteur DOM actuel) | Contenu factice actuel | Donnée réelle attendue |
|---|---|---|
| Lecteur `<video id="dashVideo">` + `<source src=sintel>` | trailer Sintel + poster Pexels | `file_url` de la vidéo sélectionnée |
| Header colonne : `h2[data-i18n="select_video"]`, `a[data-i18n="all_videos"]`, `.upload-btn` | textes statiques | titre vidéo courante / lien / bouton upload |
| **(inexistant)** sélecteur/liste de vidéos | — | **AUCUN moyen de choisir une vidéo** (cf. §5, bloquant) |
| `#panel-transcript .transcript-list` (`.transcript-row` > `.t` + `.txt`) | 6 lignes FR/EN codées en dur | résumé + segments réels (`/video/{id}`, `/search`) |
| `#panel-timeline .tab-placeholder` | « bientôt » | segments speech triés (`/search top_k=50`) |
| `#panel-chapters .chapter` (`.idx`+`.t`+`.time`) | 5 chapitres en dur | `/video/{id}/chapters` |
| `#panel-graph .tab-placeholder` | « en cours de génération » | `/graph/video/{id}` rendu d3 |
| Zone Ask : `#askInput` + `.ask-send` | textarea préremplie, **bouton sans handler** | envoi question → `/ask/stream` |
| **(inexistant)** fil de conversation / bulle réponse | — | streaming réponse + panneau chunks/citations |
| `.top-tab[data-tab=videos/upload/history]` | bascule visuelle seulement | pages Vidéos/Upload/Historique |

### 2.2 JS déjà présent dans `dashboard.js` — **À GARDER TEL QUEL**

- Sidebar toggle + backdrop (`#sidebarToggle`, `#sideBackdrop`).
- Sidebar nav active (`.side-item[data-nav]`).
- Top-tabs actif + crumb (`.top-tab[data-tab]`).
- Video-tabs (bascule panneaux `#panel-*` via `[data-vtab]`).
- **Lecteur custom** : play/pause (`#playBtn`), progress (`#vProgress/#vFilled`),
  temps (`#vTime/#vDur`), seek au clic, clic sur `.transcript-row` → seek. (`fmt()`)
- `setAsk(q)` (remplit `#askInput` depuis les `.try-item`).
- Thème `data-theme` sur `<html>` + `localStorage['vgia_theme']`.
- **i18n FR/EN** (`dict`, `applyLang`, `localStorage['vgia_lang']`).

> ⚠️ Ces blocs référencent des IDs (`#dashVideo`, `#playBtn`, `#vFilled`…) qui doivent
> continuer d'exister. Le portage devra **réutiliser `#dashVideo`** comme élément vidéo
> (et non recréer `#video-el`).

---

## 3. MAPPING PRÉCIS — endpoint existant → cible dans le nouveau design

Pour chaque brique réseau de l'ancien `api.js`, où la brancher dans le nouveau
`dashboard.js` / `dashboard.html`. **Convention** : on crée un nouveau module réseau
réutilisable (`API`, identique à l'ancien) + des fonctions de rendu adaptées aux
sélecteurs du nouveau design.

| # | Fonction/endpoint ancien | Sélecteur(s) DOM cible (nouveau) | Fonction `dashboard.js` à créer/modifier |
|---|---|---|---|
| 1 | `API` (fetch get/post/delete + tous les wrappers) | — | **COPIER tel quel** l'objet `API` (+`HistoryStore`, `State`, helpers `esc/fmt/cleanVideoLabel`) en haut de `dashboard.js` |
| 2 | `GET /videos` → `loadVideoList()` | **à ajouter** : sélecteur/liste de vidéos (cf. §5-A) | `loadVideoList()` : remplir le nouveau sélecteur + `renderMyVideosList()` pour l'onglet Videos |
| 3 | `selectVideo(v)` | `h2[data-i18n=select_video]` (titre), `#dashVideo` | `selectVideo()` adaptée : set titre, `loadVideoPlayer`, `loadTranscript`, `loadGraphViz` |
| 4 | `GET /video/{id}/stream` → `loadVideoPlayer()` | `#dashVideo` + son `<source>` | `loadVideoPlayer(v)` : `dashVideo.src=v.file_url; dashVideo.load()` (retirer le `<source>` Sintel) |
| 5 | `GET /video/{id}` (résumé+keywords) → onglet Transcript | `#panel-transcript .transcript-list` | `loadTranscript()` : remplacer les `.transcript-row` en dur par le résumé/segments réels |
| 6 | `POST /search` (timeline) | `#panel-timeline` | `loadTimeline()` : injecter segments triés (remplace `.tab-placeholder`) |
| 7 | `GET /video/{id}/chapters` | `#panel-chapters` | `loadChapters()` : générer les `.chapter` (`.idx/.t/.time`) réels + clic → seek |
| 8 | `GET /graph/video/{id}` → `loadGraphViz()` (d3) | `#panel-graph` | Injecter conteneur `#graph-viz-container` **dans** `#panel-graph` + charger d3 ; réutiliser `renderGraphViz()` |
| 9 | `POST /ask/stream` → `sendQuestion()` + `renderChunkPanel()` | `#askInput`, `.ask-send`, **+ fil de conversation à ajouter** | `sendQuestion()` : wirer `.ask-send` (clic + Ctrl/⌘+Entrée), streamer dans une bulle, appeler `renderChunkPanel()` |
| 10 | `renderChunkPanel()` / `seekVideoTo()` | bulle réponse + `#dashVideo` | **COPIER** ; adapter `seekVideoTo` pour cibler `#dashVideo` (déjà générique `querySelector('video')`) |
| 11 | `POST /upload` + `POST /ingest` → `handleDashboardUpload()` | `.upload-btn`, top-tab/nav `upload` | `handleUpload()` : câbler bouton + input fichier caché ; barre de progression (cf. §5-B) |
| 12 | `GET /ingestion/status/{id}` → `startPolling()` | zone statut (à ajouter) | `startPolling()` réutilisable |
| 13 | `DELETE /video/{id}` → `confirmDelete()` | bouton supprimer (à ajouter, onglet Videos) | `confirmDelete()` réutilisable |
| 14 | `HistoryStore` (`/history`) | top-tab/nav `history` (à peupler) | `renderHistory()` |
| 15 | `GET /vision/frame/{chunk_id}` | `<img>` dans `renderChunkPanel` | déjà inclus dans #10 |
| 16 | Init sélection (`?v=`, sticky, session) | — | Adapter le bloc `DOMContentLoaded` (le nouveau `dashboard.js` n'a pas de `DOMContentLoaded` global — fusionner) |

---

## 4. Points de friction / conflits identifiés (à trancher à l'exécution)

1. **Thème : deux systèmes incompatibles.**
   Ancien = `body.light-mode` ; nouveau = `<html data-theme>`. → **Garder le système du
   nouveau design** (`data-theme` + `vgia_theme`). Ne rien porter du thème de `main.js`.
   Le CSS du nouveau design pilote le thème via `data-theme` → conforme à « ne pas
   toucher au CSS ».

2. **i18n vs contenu injecté dynamiquement.**
   `applyLang()` ne traduit que les nœuds `[data-i18n]` **statiques**. Les transcripts,
   chapitres, réponses injectés par JS ne passeront pas par le dico (et n'ont pas à l'être :
   ce sont des données réelles). À noter : quand on remplace les `.transcript-row`/`.chapter`
   en dur par du contenu réel, on **retire aussi leurs `data-i18n`** (sinon un changement de
   langue réécraserait la donnée réelle par le texte factice du dico). **Action** : purger
   les clés `tr_*`, `ch_*`, `tl_soon`, `gr_soon` du `dict` une fois ces zones dynamisées.

3. **Un seul `<video>` doit exister.** Réutiliser `#dashVideo` (le player custom du
   nouveau design en dépend). Ne pas introduire `#video-el` de l'ancien.

4. **Chemins relatifs vs origine.** Voir §6 (centralisation `API_BASE_URL`).

5. **Ordre de chargement des scripts.** Le nouveau `dashboard.html` charge un seul
   `dashboard.js` en fin de body. Si on garde `d3` (graph), il faudra ajouter la balise
   `<script src=d3>` **avant** `dashboard.js` (comme dans l'ancien index.html).

---

## 5. Éléments du NOUVEAU design SANS logique existante (à décider ensemble)

Ces éléments existent visuellement mais n'ont pas d'équivalent branché dans l'ancien
dashboard — à spécifier avant de coder :

- **A. Absence totale de sélecteur / liste de vidéos dans la vue Ask.** 🔴 *Bloquant.*
  L'ancien dashboard repose sur `<select id="video-select">`. Le nouveau n'a qu'un titre
  « Sélectionnez une vidéo » et un lien « Toutes les vidéos ». **Décision requise** :
  (a) ajouter un `<select>` discret dans `.col-head`, ou (b) faire de l'onglet « Videos »
  (top-tab/sidebar) une vraie grille cliquable qui sélectionne puis bascule sur « Ask ».
  → *Recommandation : (b), plus proche de l'intention visuelle ; (a) en repli rapide.*
- **B. Barre de progression d'upload.** Le nouveau design n'a ni carte de statut d'upload
  ni barre de progression → à concevoir (l'ancien avait `#upload-status-card`).
- **C. Contrôles player custom avancés** (rewind/forward 10 s, volume, captions,
  fullscreen) : boutons présents mais sans handler. Non essentiels au merge réseau —
  à câbler plus tard (optionnel).
- **D. Top-tabs comme SPA** (ask/videos/upload/history dans une seule page) vs ancien =
  4 pages HTML séparées. **Décision** : tout mettre dans `dashboard.html` (SPA) ou garder
  des pages séparées ? → impacte l'ampleur du portage (History/Upload/Videos/Settings).
- **E. Carte « Upgrade to PRO », cloche de notifications, menu user** : purement visuels,
  pas de backend correspondant utilisé aujourd'hui → laisser statiques.

## 5 bis. Éléments de l'ANCIEN dashboard SANS équivalent visuel dans le nouveau

À recréer en HTML (sans toucher au CSS : réutiliser les classes existantes du nouveau
design, ou des styles inline comme le fait déjà l'ancien `api.js`) :

- Fil de conversation / bulles Q&R streamées (`#chat-thread`, `.ask-answer`, `.chat-question`).
- Indicateur de chargement « Searching… » (`#ask-loading`).
- Panneau « Referenced segments » (chunks audio + frames visuelles) — généré par
  `renderChunkPanel()` en JS, donc pas de HTML à créer, **mais** dépend de variables CSS
  (`--text-secondary`, etc.) : vérifier qu'elles existent dans `dashboard.css`.
- Barre de statut (`#status-bar`) + carte statut upload.
- Bouton **Re-index** et bouton **Supprimer** la vidéo.
- Grille **My Videos**, page **Upload**, page **History**, **Recommandations**.
- Conteneurs graphe : `#graph-viz-container`, `#graph-viz-legend`, `#graph-viz-tooltip`.

---

## 6. Centralisation de la config (`API_BASE_URL`)

Aujourd'hui l'ancien code fait `fetch('/videos')` (relatif). Pour préparer le déploiement
(front éventuellement servi ailleurs que le backend), introduire **un seul point de
config** en tête de `dashboard.js` :

```js
// ─── CONFIG (unique endroit à changer pour le déploiement) ───
const API_BASE_URL = ""; // "" = même origine (dev local). Ex prod: "https://api.mondomaine.com"
```

Puis modifier les 3 helpers `API.get/post/delete` (+ `uploadFile` XHR, + `askStream` fetch,
+ `HistoryStore`, + les URLs d'images `/vision/frame/...`) pour préfixer `API_BASE_URL`.
→ Un seul fichier, une seule constante à changer. (Alternative : exposer
`window.API_BASE_URL` depuis un `<script>` dans `dashboard.html` pour config sans toucher au JS.)

> Remarque déploiement : si `API_BASE_URL` pointe vers une autre origine, le backend
> autorise déjà le CORS (`allow_origins=["*"]`, `api/main.py:117`). Le streaming vidéo
> (`/video/{id}/stream`) supporte les requêtes Range (`206`) → OK cross-origin.

---

## 7. Plan d'exécution étape par étape (proposé)

> Principe directeur : **0 modification CSS**, réutiliser `#dashVideo` et tous les IDs du
> nouveau design, porter la logique de `api.js` telle quelle quand c'est possible.

**Étape 0 — Préparation**
- Décider §5-A (sélecteur vidéo) et §5-D (SPA vs pages) — *questions ouvertes*.
- Sauvegarde/versioning de `project/` avant édition.

**Étape 1 — Socle réseau (aucun visuel)**
- Copier dans `dashboard.js` : la constante `API_BASE_URL` (§6), l'objet `API`,
  `HistoryStore`, `State`, et les helpers `esc/fmt/cleanVideoLabel/setStatus`.
- Préfixer toutes les URLs par `API_BASE_URL`.
- ✅ Vérif : `await API.listVideos()` en console renvoie les vraies vidéos.

**Étape 2 — Liste & sélection de vidéos** (dépend de §5-A)
- Implémenter `loadVideoList()` + `selectVideo()` ciblant les sélecteurs du nouveau design.
- Restaurer la persistance de sélection (`?v=`, localStorage sticky, sessionStorage).
- ✅ Vérif : sélectionner une vidéo met à jour le titre.

**Étape 3 — Lecteur vidéo**
- `loadVideoPlayer(v)` → `dashVideo.src = API_BASE_URL + v.file_url; dashVideo.load()`.
- Retirer `<source src=sintel>` du HTML (garder la balise `#dashVideo`).
- ✅ Vérif : la vraie vidéo se lit ; le player custom (play/seek/temps) fonctionne toujours.

**Étape 4 — Onglets Transcript / Timeline / Chapters**
- `loadTranscript()` (résumé+keywords), `loadTimeline()` (`/search`), `loadChapters()`
  (`/video/{id}/chapters`), en remplaçant le contenu en dur des `#panel-*`.
- Retirer les `data-i18n` des zones dynamisées + purger les clés correspondantes du `dict` (§4.2).
- ✅ Vérif : les 3 onglets affichent les vraies données ; clic timestamp → seek `#dashVideo`.

**Étape 5 — Q&A streaming (cœur)**
- Ajouter en HTML un fil de conversation + loader (classes du nouveau design ou inline).
- Porter `sendQuestion()` + `renderChunkPanel()` + `_filterChunksByCitations()` +
  `_linkifyTimestamps()` + `seekVideoTo()`.
- Câbler `.ask-send` (clic) et `#askInput` (Ctrl/⌘+Entrée) ; garder `setAsk()` pour `.try-item`.
- ✅ Vérif : question → réponse streamée + panneau chunks + citations cliquables + `HistoryStore.save`.

**Étape 6 — Graphe de connaissances**
- Injecter `<script d3>` avant `dashboard.js` ; insérer `#graph-viz-container/legend/tooltip`
  dans `#panel-graph` ; brancher `loadGraphViz()` + rendu au moment où l'onglet Graph devient visible.
- ✅ Vérif : le graphe s'affiche et est interactif.

**Étape 7 — Upload / Re-index / Delete / Polling** (dépend §5-B, §5-D)
- Câbler upload (`handleUpload`), `startPolling`, `confirmDelete`, bouton Re-index.

**Étape 8 — Onglets Videos / Upload / History / Settings** (dépend §5-D)
- Si SPA : peupler ces vues ; sinon lier vers pages séparées.

**Étape 9 — Nettoyage & config finale**
- Vérifier que le nouveau `dashboard.html` est servable par FastAPI (route `/app` ou copie
  dans `static/app/`), ou décider du mode de service.
- Recontrôler : thème, i18n, sidebar, tabs = intacts ; **CSS non modifié**.

---

## 8. Questions ouvertes à valider avant de coder

1. **Sélecteur de vidéos** (§5-A) : `<select>` dans l'en-tête, ou grille cliquable via
   l'onglet « Videos » ? *(bloquant)*
2. **Architecture** (§5-D) : tout en SPA dans `dashboard.html`, ou conserver des pages
   séparées Upload/History/Videos/Settings comme l'ancien ?
3. **Service du fichier** : le nouveau `project/dashboard.html` sera-t-il déplacé sous
   `static/app/` (servi par FastAPI, chemins relatifs OK) ou servi séparément (→ nécessite
   `API_BASE_URL` non vide) ?
4. **Onglet Videos/Upload/History** : à brancher maintenant ou dans une phase ultérieure
   (comme l'auth Supabase) ?
