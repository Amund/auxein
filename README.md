# Auxein, « croître » en grec

**Version officielle : 0.1.0**

## Description

En une phrase, **Auxein est une mémoire vectorielle qui se construit elle-même pendant qu’elle observe**.

Le moteur reçoit un flux de vecteurs et entretient une population d’identités locales — les Cellules — qui représentent les régularités rencontrées. Ces identités ne sont pas fixes : elles se déplacent, se divisent, meurent et peuvent faire apparaître de nouveaux niveaux de représentation. La hiérarchie n’est donc ni choisie à l’avance, ni obtenue après une phase d’entraînement séparée ; elle est l’état courant d’un processus continu.

Une des caractéristiques du projet est la distinction entre deux formes de croissance :

* **la croissance horizontale** affine une représentation locale. Lorsqu’une Cellule contient durablement deux tendances géométriques incompatibles, elle peut se diviser en deux identités plus précises ;
* **la croissance verticale** extrait une régularité commune à plusieurs identités locales. Lorsqu’une même distinction réapparaît à travers des Cellules différentes, elle peut devenir la matière d’une couche supérieure.

Autrement dit, une couche inférieure apprend surtout *ce qui est semblable localement*, tandis qu’une couche supérieure peut apprendre *ce qui varie de la même manière dans plusieurs contextes*. C’est dans ce second mécanisme qu’**Auxein** ressemble le plus à une machine d’abstraction plutôt qu’à un simple algorithme de clustering.

### Une image mentale

On peut imaginer **Auxein** comme un petit écosystème de territoires perceptifs.

Chaque Cellule possède un centre, reconnaît une partie du flux proche de ce centre et mémorise la structure de ce qu’elle reçoit. Si son territoire contient deux courants réellement distincts, elle se divise. Si des territoires différents produisent une même forme de variation résiduelle, un nouvel étage peut apparaître pour représenter cette relation commune. Les structures ont toutefois un coût permanent : elles doivent tenir dans un budget de maintenance, sinon les moins défendables disparaissent.

Cette métaphore biologique n’est pas seulement décorative. La mitose, le bourgeonnement, la mort et les couches correspondent à des opérations mathématiques précises. L’« économie » ne récompense pas une tâche extérieure : elle empêche simplement la représentation de croître sans borne et oblige le système à arbitrer entre précision locale, abstraction et coût de persistance.

### À quoi cela ressemble, et ce que ce n’est pas

**Auxein** se situe à l’intersection de plusieurs familles connues : quantification vectorielle en ligne, clustering adaptatif, systèmes auto-organisés à topologie croissante, mémoire hiérarchique et apprentissage de représentations non supervisé. Il évoque aussi un système développemental : sa forme finale n’est pas codée dans son architecture initiale, mais résulte de son histoire.

Il s’en distingue toutefois sur des points importants :

* aucun nombre de groupes ou de couches n’est fixé ;
* aucune matrice de poids n’est apprise ;
* il n’existe ni époques, ni lots, ni phase d’inférence séparée ;
* aucune étiquette, fonction de perte ou cible extérieure ne décide de ce qui mérite d’exister ;
* une transformation doit justifier un gain géométrique, puis être payable dans le budget courant.

**Auxein** n’est donc pas un classifieur ou un prédicteur prêt à l’emploi. C’est plutôt un **substrat de représentation adaptative** : il organise un flux avant qu’un autre système ne cherche éventuellement à prédire, décider, expliquer ou contrôler.

### Usages plausibles

Le moteur paraît particulièrement adapté aux situations où la distribution change avec le temps et où l’on veut conserver une structure interprétable sans réentraîner périodiquement un modèle complet.

Des usages naturels sont :

* **compression structurelle d’un flux** : remplacer de nombreuses observations par des identités persistantes, puis représenter les relations récurrentes entre ces identités ;
* **mémoire de régimes** : suivre les états récurrents d’un système, leurs transitions et l’apparition de nouvelles organisations ;
* **prétraitement adaptatif** : fournir à un classifieur, un prédicteur ou un contrôleur une représentation plus compacte et déjà structurée ;
* **exploration non supervisée** : observer quelles catégories et quelles abstractions émergent sans imposer au moteur un vocabulaire humain ;
* **agents continus ou systèmes embarqués** : entretenir une mémoire bornée de l’expérience sous une contrainte explicite de ressources.

Les domaines possibles incluent la télémétrie, les signaux de capteurs, les trajectoires d’agents, les séries multivariées et, plus généralement, tout flux vectoriel non stationnaire. La sémantique des Cellules n’est cependant pas garantie : le moteur découvre une géométrie utile selon ses propres lois, pas nécessairement les catégories qu’un humain attend.

### La question de recherche portée par le projet

Auxein explore la question suivante :

> Jusqu’où peut-on faire émerger une représentation hiérarchique, continue et bornée en ressources à partir de la seule géométrie d’un flux, sans objectif externe ni architecture figée ?

Le projet ne répond pas à cette question par une métaphore vague. Il propose une mécanique complète — mémoire quadratique, reconnaissance locale, bifurcation, concordance entre identités, capital géométrique et maintenance — définie de manière déterministe et testable.

## Contenu du dépôt

Le projet contient :

- la spécification normative dans [`spec/auxein.md`](spec/auxein.md) ;
- l’implémentation Python de référence dans [`auxein.py`](auxein.py) ;
- un laboratoire expérimental reproductible dans [`lab.py`](lab.py) ;
- un benchmark dans [`benchmark.py`](benchmark.py) ;
- une suite de tests dans [`test.py`](test.py) ;
- un socle de preuves Lean dans [`lean/`](lean/).

## Prérequis

### Moteur Python

- Python **3.10 ou plus récent** ;
- aucune dépendance Python externe : le moteur et les outils utilisent uniquement la bibliothèque standard.

### Vérification Lean facultative

- `elan` ;
- Lean **4.32.2** ;
- mathlib **4.32.2**, téléchargée par Lake.

## Démarrage rapide

Depuis la racine du projet :

```bash
python test.py
```

Puis lancer l’expérience minimale :

```bash
python lab.py experiments/smoke.json --check-invariants
```

Pour obtenir le résultat complet en JSON :

```bash
python lab.py experiments/smoke.json --check-invariants --json
```

## Utilisation directe en Python

L’API directe distingue deux représentations du même budget :

- `budget` exprime une capacité ergonomique en cellules terminales équivalentes ;
- `budget_units` exprime l’empreinte brute exacte sous forme d’un entier.

Il faut fournir exactement l’un des deux. Le moteur réalise lui-même la conversion selon la dimension, le format persistant et le modèle de maintenance.

```python
from auxein import Auxein

network = Auxein.empty(
    dimension=2,
    memory=50,
    budget=8,
    eta=1.0,
    scalar="f64",
)

stream = ([-2.0, 0.0], [2.0, 0.0]) * 20

for point in stream:
    report = network.step(point, detailed_report=False)

print(network.summary())
```

`eta` est le multiplicateur du taux d’apprentissage. Il appartient à l’intervalle `[0, 1]`, vaut `1` par défaut et peut être modifié pendant l’exécution. Le coefficient statistique effectivement appliqué est `eta * alpha`, où `alpha` reste déterminé par `memory`. À `eta = 0`, les mémoires et les transformations volontaires sont gelées ; les destructions forcées nécessaires à la solvabilité restent actives.

Pour imposer directement une empreinte exacte :

```python
network = Auxein.empty(
    dimension=2,
    memory=50,
    budget_units=1248,
)
```

Le résumé distingue lui aussi les deux niveaux :

```python
{
    "steps_seen": 40,
    "dimension": 2,
    "scalar": "f64",
    "memory": 50.0,
    "layer_count": 1,
    "cells_per_layer": [8],
    "capital_per_layer": [...],
    "maintenance_units": 1248,
    "budget": "8",
    "budget_units": 1248,
    "budget_margin_units": 0,
    "is_solvent": True,
    "chi": ...,
    "alpha": ...,
    "eta": 1.0,
    "effective_alpha": ...,
    "root_bud_mass": 0.0,
}
```

### Réseau vide ou pré-initialisé

L’initialisation canonique utilise `Auxein.empty()`. Le réseau apprend alors son incarnation initiale par son bourgeon racine.

Pour les tests ou l’import d’une topologie explicitement fondée :

```python
network = Auxein.from_seed(
    [0.0, 0.0],
    memory=50,
    budget=8,
    eta=1.0,
    scalar="f64",
)
```

La propriété en lecture seule `network.memory` expose la demi-vie statistique en nombre de présentations. Les propriétés `network.budget` et `network.budget_units` sont modifiables. La première applique la conversion ergonomique canonique ; la seconde attend un entier exact. `network.budget_margin_units` expose la marge structurelle courante, et `network.is_solvent` indique si la topologie tient dans le budget exact.

### Rapport d’un pas

`Auxein.step()` retourne un `StepReport` contenant notamment :

- `step_index` ;
- `maintenance_charged_units` et `maintenance_units` ;
- `budget_units` et `remaining_step_budget_units` ;
- les transformations réalisées ;
- les rapports de couches ;
- le gain et la concordance verticaux.

Pour éviter les scans purement diagnostiques :

```python
report = network.step(point, detailed_report=False)
```

Les transformations et l’état final restent identiques ; seuls les rapports détaillés sont omis. Dans ce mode, `layer_reports` est vide et les diagnostics verticaux valent `None`.

## Sérialisation stricte

Le moteur expose un état causal complet compatible JSON.

```python
import json
from auxein import Auxein

state = network.to_state_dict()

with open("state.json", "w", encoding="utf-8") as handle:
    json.dump(state, handle, ensure_ascii=False, allow_nan=False)

with open("state.json", encoding="utf-8") as handle:
    restored = Auxein.from_state_dict(
        json.load(handle),
        budget_units=network.budget_units,
    )
```

Le chargement rejette notamment :

- les clés manquantes ou inconnues ;
- les versions incompatibles ;
- les valeurs non finies ;
- les conversions numériques silencieuses entre `f32` et `f64` ;
- les noyaux quadratiques non canoniques ;
- les identités dupliquées ou les compteurs périmés ;
- les états sérialisés pendant un arbitrage éphémère.

Le budget et le modèle de maintenance ne font pas partie de l’état causal sérialisé : ils doivent être fournis explicitement au rechargement. `eta`, qui influe sur la trajectoire future du moteur, appartient en revanche à l’état sérialisé.

## Laboratoire expérimental

[`lab.py`](lab.py) exécute des expériences JSON déterministes composées de phases. Le monde produit une vérité extérieure, mais le moteur Auxein ne la reçoit jamais directement.

Expériences fournies :

- [`experiments/smoke.json`](experiments/smoke.json) : vérification rapide sur un flux alternant ;
- [`experiments/abstraction.json`](experiments/abstraction.json) : émergence d’abstractions ;
- [`experiments/extinction.json`](experiments/extinction.json) : croissance, extinction budgétaire et renaissance.

Commandes utiles :

```bash
# Exécution lisible avec validation des invariants
python lab.py experiments/abstraction.json --check-invariants

# Un objet JSONL par essai
python lab.py experiments/abstraction.json \
  --check-invariants \
  --output results.jsonl

# Sauvegarde de l’état final d’un essai unique
python lab.py experiments/smoke.json \
  --check-invariants \
  --save state.json
```

La définition d’expérience accepte notamment :

- la dimension ;
- le format persistant `f32` ou `f64` ;
- la demi-vie statistique `memory` ;
- un budget ergonomique en cellules équivalentes ;
- une graine déterministe ;
- plusieurs phases et mondes ;
- des observations et sondes.

## Benchmark

Le benchmark mesure le moteur de référence sans dépendance externe :

```bash
python benchmark.py \
  --steps 1000 \
  --warmup 50 \
  --dimension 2 \
  --memory 50 \
  --scalar f64 \
  --budget 100 \
  --stream alternating
```

Options principales :

```text
--stream gaussian|alternating|drifting
--budget CELLS          budget ergonomique en cellules équivalentes
--budget-units UNITS    budget brut exact
--eta RATE              multiplicateur d’apprentissage dans [0, 1]
--load FILE             charger un état
--save FILE             sauvegarder l’état final
--window STEPS          afficher des fenêtres de débit
--check-invariants      valider toute la hiérarchie à chaque phase causale
--json                  produire un résultat structuré
```

Le moteur de référence favorise la littéralité et le déterminisme. En régime saturé, les recherches de conservation peuvent légitimement atteindre \(O(N^2D)\). Une dérive vers \(O(N^3D)\) indique généralement un recalcul imbriqué accidentel.

## Vérification Lean

Depuis le sous-répertoire `lean/` :

```bash
cd lean
lake build --wfail
```

Organisation des preuves :

```text
lean/Auxein/
├── Geometry.lean       identités d’espace préhilbertien réel
├── Kernel.lean         noyau quadratique et recentrage
├── Admissibility.lean  invariant WQ ≥ ‖S‖² et préservation
├── Routing.lean        routage latent
├── Split.lean          gain de séparation
├── Concordance.lean    concordance entre identités
├── Topology.lean       matérialisation et invariants du split
└── Solvency.lean       troncature, budget et terminaison
```


## Modèle numérique

Auxein sépare trois domaines :

| Domaine | Représentation |
|---|---|
| Géométrie persistante | `f32` ou `f64` |
| Calcul intermédiaire Python | binary64 (`float`) |
| Maintenance et budget | entiers exacts |

À chaque frontière causale de mutation, les réels persistants sont projetés dans le format choisi. Les trajectoires `f32` et `f64` sont donc déterministes et distinctes, sans prétendre émuler chaque instruction matérielle binary32.

Règle centrale : **fermer les zéros structurellement démontrés, jamais une valeur seulement petite**.

Les garde-fous couvrent notamment :

- `NaN`, `±∞`, dépassements et sous-flux ;
- annulation dans les différences quadratiques ;
- cohérence des moments après projection ;
- exactitude des égalités de seuil et des départages ;
- atomicité face aux entrées quadratiquement non représentables ;
- fermeture canonique des branches sous-fluées en `f32`.

## Limites connues

Les principales limites numériques et de validation concernent :

- le traitement canonique de `-0.0` ;
- les échelles nulles ou subnormales ;
- le départage d’offres de destruction exactement égales ;
- l’accumulation `f32` avec de nombreux propriétaires ;
- la couverture des frontières numériques par des tests déterministes ;
- les performances en régime saturé.

## Notes sur l'auteur

Je ne suis ni mathématicien, ni spécialiste en réseaux neuronaux, ni développeur Python chevronné. Mais comme tous les humains, j’ai des intuitions et j’utilise les outils à ma disposition pour les concrétiser. Les principaux éléments mis à l’épreuve pour ce projet sont :

- Deepseek-V4-Preview (flash et pro) ;
- ChatGPT-5.6-Sol ;
- ma cafetière Senseo ;
- la patience de mon épouse.

## Licence

Copyright © 2026 Dimitri Avenel.

Ce projet est distribué sous la licence [GNU General Public License v3.0 uniquement](LICENSE).
