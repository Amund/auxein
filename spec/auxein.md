# Auxein, « croître » en grec

Modèle de quantification vectoriel autorégulé, piloté par les données. Le moteur gère lui-même sa topologie, sa croissance horizontale et verticale, et la représentation de ses entrées, sans seuils fixes ni paramètres externes. Tout est proportionnel à la déviation de l’état interne.

- **géométrie de l’entrée comme matière première**
- **absence de seuils arbitraires** : les transformations reposent sur des gains géométriques exacts
- **état auto-référentiel** : rayons, reconnaissances, valeurs et distinctions sont dérivés du système lui-même
- **topologie autonome** : morts et naissances de cellules et de couches appartiennent au moteur
- **flux continu** : aucune époque, aucun lot, aucune séparation entraînement/inférence

**Auxein n'a pas d'objectif**, mais des capacités. C’est une machine à abstraction, une machine de croissance alimentée par un flux, dont les seules raisons doivent émerger de sa géométrie interne et de ses ressources.

La meilleure définition de son usage serait : construire et entretenir une représentation hiérarchique en ligne à partir des régularités récurrentes d’un flux. Quelques usages possibles :

- compression structurelle adaptative : remplacer beaucoup d’observations par quelques identités, puis quelques relations communes entre ces identités
- détection de régimes et de transitions : servir de mémoire structurée d’un système non stationnaire
- découverte de motifs transférables
- en amont d'un système de classification, de prédiction, de contrôle ou d'explication, pour simplifier ou fluidifier sa tâche

Quelques notions vérifiées empiriquement :

- l’abstraction provient de la partie reconnue qui n’est pas absorbée par l’identité locale
- ce qui devient abstrait dépend de ce que l’étage inférieur choisit d’absorber
- la profondeur factorise une relation commune entre représentations locales différentes
- la récursion fonctionnelle profonde existe

## 1. Constitution

- **la géométrie pousse, l'économie autorise**
- l'économie n'est pas basée sur la géométrie ; la géométrie **est** l'économie

Le principe de croissance et d’arbitrage est :

- les croissances horizontale et verticale sont **économiquement équivalentes** ; seule la géométrie décide de l'orientation
- La géométrie est l’échelle commune de valeur des transformations.
- La maintenance est l’échelle de solvabilité.
- Aucune conversion par ratio ou coefficient n’est introduite entre ces deux grandeurs.

La constitution hiérarchique est :

- une cellule ne connaît que son propre état
- une couche connaît son état et ses cellules
- le réseau connaît son état, son bourgeon racine et ses couches

La fondation économique est :

- la géométrie du moteur est complète sans économie ; sans économie, Auxein peut croître sans borne
- l'économie est le frein nécessaire au fonctionnement en espace et ressources finis
- il n'y a pas d'épargne, pas de spéculation ; le budget global est neuf à chaque tour
- la mort est une perte sèche ; elle réduit seulement la maintenance future

## 2. Horloge statistique commune

La fondation géométrique expose :

\[
D\in\mathbb N^*,
\qquad
0<T_{mem}<\infty.
\]

La réalisation numérique de référence expose en outre un format persistant :

\[
\boxed{p\in\{\mathrm{f32},\mathrm{f64}\}.}
\]

Ce format n’est ni une preuve géométrique, ni une horloge, ni un seuil. Il fixe la résolution des réels persistants et leur empreinte dans le modèle de maintenance de référence.

La rétention statistique universelle est :

\[
\boxed{\chi=2^{-1/T_{mem}}}
\]

avec :

\[
\chi^{T_{mem}}=\frac12.
\]

Pour l’exécution numérique, poser aussi :

\[
\boxed{
\alpha
=
1-\chi
=
-\operatorname{expm1}\!\left(-\frac{\ln 2}{T_{mem}}\right).
}
\]

Cette écriture évite l’annulation de \(1-\chi\) pour les grandes demi-vies.

Le moteur expose en outre un multiplicateur d’apprentissage :

\[
\boxed{\eta\in[0,1],\qquad \alpha_\eta=\eta\alpha.}
\]

Toutes les mises à jour statistiques sont évaluées sous la forme stable :

\[
\boxed{X\leftarrow X+\alpha_\eta(X_{cible}-X).}
\]

À \(\eta=1\), on retrouve exactement l’EMA canonique définie par \(T_{mem}\). Réduire \(\eta\) ralentit conjointement l’injection et l’oubli sans modifier la demi-vie déclarée ; à \(\eta=0\), aucune mémoire n’est modifiée. Toutes les mémoires EMA utilisent le même \(\alpha_\eta\), sauf si une nécessité distincte est démontrée.

L'horizon intrinsèque dérivé reste :

\[
\Theta_{mem}=\frac1{1-\chi}=\frac1\alpha.
\]

Pour \(\eta>0\), l’horizon effectif d’exécution vaut \(1/\alpha_\eta\) ; pour \(\eta=0\), il est infini. Aucun de ces horizons ne possède de signification économique automatique et aucun ne multiplie un coût topologique.

## 3. Noyau quadratique universel

Pour une pertinence \(r\ge0\) et un vecteur \(x\in\mathbb R^D\), un noyau :

\[
H=(W,S,Q)
\]

suit :

\[
\boxed{W\leftarrow W+\alpha_\eta(r-W)}
\]

\[
\boxed{S\leftarrow S+\alpha_\eta(rx-S)}
\]

\[
\boxed{Q\leftarrow Q+\alpha_\eta(r\|x\|^2-Q).}
\]

L'invariant est :

\[
\boxed{WQ\ge\|S\|^2.}
\]

Pour \(W>0\) :

\[
\mu=\frac SW,
\]

\[
P_{move}=\frac{\|S\|^2}{W},
\qquad
P_{struct}=Q-\frac{\|S\|^2}{W}\ge0,
\]

et :

\[
Q=P_{move}+P_{struct}.
\]

### 3.1. Recentrage exact

Après déplacement normalisé \(\Delta\) du centre du noyau :

\[
S'=S-W\Delta,
\]

\[
Q'=Q-2\Delta\cdot S+W\|\Delta\|^2.
\]

Alors :

\[
\boxed{P'_{struct}=P_{struct}.}
\]

### 3.2. Oubli sans injection

Pour \(r=0\) :

\[
W'=(1-\alpha_\eta)W,
\qquad
S'=(1-\alpha_\eta)S,
\qquad
Q'=(1-\alpha_\eta)Q.
\]


### 3.3. Projection dans le format persistant

Les expressions intermédiaires de la référence Python sont évaluées en `float` binary64. À chaque frontière causale de mutation, tout réel destiné à persister est projeté dans le format choisi \(p\).

\[
\boxed{
\operatorname{persist}_p(x)
=
\text{arrondi IEEE-754 de }x\text{ vers }p.
}
\]

La projection s’applique uniformément à tous les réels persistants : masses, premiers moments, seconds moments, centres, formes propriétaires et masses de concordance. Elle ne s’applique pas sélectivement aux seuls « poids ».

Cette convention rend les trajectoires `f32` et `f64` déterministes et réellement distinctes, sans prétendre émuler chaque instruction d’un processeur matériel binary32.

Les fermetures numériques de l’invariant \(WQ\ge\|S\|^2\) et les égalités de géométrie dégénérée sont dérivées de la résolution du format choisi. Elles ne constituent jamais un epsilon sémantique ni un seuil de transformation.

## 4. Géométrie propre d'une couche

Une couche reçoit uniformément :

\[
(I,r_{in}),
\qquad
r_{in}\ge0.
\]

Aucune couche ne possède de règle d’entrée, d’amorçage, de survie ou de mort particulière. Le réseau construit le couple reçu par la première couche existante.

Chaque couche entretient :

\[
H_R=(W_R,S_R,Q_R)
\]

sur son flux reçu.

Pour \(W_R>0\) :

\[
M_R=\frac{S_R}{W_R},
\]

\[
\boxed{
R_{geo}^2
=
\frac{Q_R}{W_R}-\|M_R\|^2.
}
\]

\(R_{geo}\) est l'unité géométrique interne dérivée du flux. Elle ne dépend ni du nombre de Cellules ni de leur compétition.

### 4.1. Convention dégénérée

Pour une distance physique \(d\), si \(R_{geo}>0\) :

\[
d_{norm}=\frac d{R_{geo}}.
\]

Si \(R_{geo}=0\) :

\[
\boxed{
d_{norm}
=
\begin{cases}
0,&d=0,\\
+\infty,&d>0.
\end{cases}
}
\]

Aucun \(\varepsilon\) comportemental n'est introduit.

### 4.2. Interface racine du réseau

Le réseau possède en permanence un bourgeon racine :

\[
\boxed{B_{\varnothing}=(W_{\varnothing},S_{\varnothing},Q_{\varnothing}).}
\]

Ce bourgeon :

- n’est pas une couche ;
- ne possède aucune Cellule ni identité source ;
- n’effectue aucune concordance inter-identitaire ;
- utilise le noyau quadratique universel ;
- constitue l’organe minimal par lequel le réseau peut recommencer à construire une hiérarchie.

L’ensemble des couches est :

\[
\boxed{\mathcal L=\varnothing}
\qquad\text{ou}\qquad
\boxed{\mathcal L=\{0,1,\ldots,L_{max}\}.}
\]

Si \(\mathcal L=\varnothing\), l’entrée extérieure \(x\) est présentée au bourgeon racine avec \(r=1\). Si des couches existent, le réseau présente :

\[
\boxed{(I_0,r_{in,0})=(x,1)}
\]

à la couche \(0\). Cette convention appartient au réseau ; elle ne modifie aucune loi interne de la couche.

#### 4.2.1. Incarnation de la première couche

Lorsque :

\[
\mathcal L=\varnothing,
\qquad
W_{\varnothing}>0,
\]

et que l’économie autorise l’empreinte persistante résultante, le réseau matérialise une couche \(0\) avec une seule Cellule fondatrice.

Poser :

\[
C_0=\frac{S_{\varnothing}}{W_{\varnothing}},
\]

\[
A_0=W_{\varnothing},
\qquad
E_0=0,
\qquad
G_0=Q_{\varnothing}-\frac{\|S_{\varnothing}\|^2}{W_{\varnothing}}.
\]

La Cellule reçoit une identité neuve et un split latent neutre :

\[
\boxed{H_0^+=H_0^-=\frac12(A_0,0,G_0).}
\]

La géométrie propre de la nouvelle couche est une copie du noyau racine accumulé. Le bourgeon racine est ensuite consommé et remis à zéro. La présentation ayant causé cette naissance n’est pas rejouée dans la couche créée, qui ne produit aucune demande avant la présentation suivante.

Cette incarnation ne participe à aucun marché de valeur géométrique : lorsqu’aucune couche n’existe, aucune transformation concurrente n’existe. Elle est exécutée dès que sa masse est positive et que l’économie ne lui oppose pas son veto.

#### 4.2.2. Warmup sans topologie

Tant que la naissance n’est pas soutenable, le bourgeon racine continue d’accumuler la géométrie extérieure sous le même \(\alpha_\eta\). Ce warmup est dépourvu de couches et de Cellules, mais il n’est pas dépourvu d’empreinte persistante :

\[
\boxed{M_{min}=M_{network}+M_{B_{\varnothing}}.}
\]

Si \(B_{units}<M_{min}\), Auxein est inexécutable.

## 5. Identité cellulaire

Chaque Cellule possède une identité opaque et persistante :

\[
\boxed{\iota_i.}
\]

L'identité permet seulement :

- de tester l'égalité de deux références ;
- de retrouver une mémoire attachée à une Cellule ;
- de suivre une source malgré la dérive de son prototype ;
- de déclarer une naissance lors d'une mitose.

Elle ne possède aucune géométrie. Sont interdites :

\[
\iota_i-\iota_j,
\qquad
\|\iota_i-\iota_j\|,
\qquad
\frac{\iota_i+\iota_j}{2}.
\]

Une permutation bijective des identités ne modifie aucun comportement géométrique.

L'identité garantit la continuité biographique, pas l'immuabilité sémantique.

---

## 6. Reconnaissance absolue et écriture exclusive

Pour une Cellule \(i\) de prototype courant \(C_i\), si \(R_{geo}>0\) :

\[
\boxed{e_i=\frac{I-C_i}{R_{geo}}.}
\]

La reconnaissance absolue est :

\[
\boxed{a_i=e^{-\|e_i\|^2}.}
\]

Si \(R_{geo}=0\), appliquer la convention de la section 4.1 : reconnaissance \(1\) en cas d'égalité exacte, \(0\) sinon.

Chaque \(a_i\) est strictement local : aucune autre Cellule n'intervient dans sa valeur.

### 6.1. Sélection unique

S'il existe une Cellule avec \(a_i>0\), l'indice actif est :

\[
\boxed{k\in\operatorname*{argmax}_i a_i}
\]

avec un ordre déterministe immuable en cas d'égalité mathématique exacte.

Cet ordre ferme les états dégénérés ; il n'a aucune signification géométrique ou économique.

### 6.2. Pertinence d'écriture

\[
\boxed{
r_i^{learn}
=
\begin{cases}
r_{in}a_i,&i=k,\\
0,&i\ne k.
\end{cases}
}
\]

Lorsqu'un gagnant existe :

\[
\sum_i r_i^{learn}=r_{in}a_k\le r_{in}.
\]

La masse non écrite :

\[
r_{void}=r_{in}(1-a_k)
\]

reste diagnostique. Elle n'est ni redistribuée ni transmise comme résidu.

La sélection décide **qui écrit** ; la reconnaissance décide **avec quelle autorité**.

## 7. Mémoire locale, bifurcation latente et mouvement

Une Cellule entretient deux noyaux quadratiques ordonnés :

\[
H_i^+=(A_i^+,E_i^+,G_i^+),
\qquad
H_i^-=(A_i^-,E_i^-,G_i^-).
\]

Le noyau parent est dérivé par somme :

\[
A_i=A_i^++A_i^-,
\qquad
E_i=E_i^++E_i^-,
\qquad
G_i=G_i^++G_i^-.
\]

Les symboles \(+\) et \(-\) sont des conventions historiques persistantes. Les histoires ne sont jamais permutées après coup.

Un split neutre d'un noyau \((A,E,G)\) est :

\[
\boxed{
A^+=A^-=\frac A2,
\quad
E^+=E^-=\frac E2,
\quad
G^+=G^-=\frac G2.
}
\]

Il porte un gain de séparation nul.

### 7.1. Routage interne de la preuve

Toutes les quantités de routage sont lues avant l'injection courante.

Pour \(A>0\) :

\[
\mu=\frac EA.
\]

Lorsque les deux masses sont positives :

\[
\mu_+=\frac{E^+}{A^+},
\qquad
\mu_-=\frac{E^-}{A^-},
\qquad
b=\mu_+-\mu_-.
\]

Poser :

\[
s=e_k-\mu.
\]

La Cellule reste l'unique propriétaire externe :

\[
\boxed{r^++r^-=r_k^{learn}.}
\]

Conventions :

- si \(r_k^{learn}=0\), alors \(r^+=r^-=0\) ;
- si le noyau est vide, la première preuve est répartie neutralement ;
- si \(b=0\) et \(s\ne0\), la première preuve structurelle non nulle amorce la branche \(+\) ;
- si \(b=0\) et \(s=0\), la preuve est partagée également ;
- si \(b\ne0\) :

\[
\boxed{
(r^+,r^-)
=
\begin{cases}
(r_k^{learn},0),&b\cdot s>0,\\
(0,r_k^{learn}),&b\cdot s<0,\\
(r_k^{learn}/2,r_k^{learn}/2),&b\cdot s=0.
\end{cases}
}
\]

Chaque branche suit ensuite le noyau universel avec \((e_k,r^\pm)\). Par sommation, le parent suit exactement \((e_k,r_k^{learn})\).

### 7.2. Déplacement

Après injection, pour \(A_i>0\) :

\[
\boxed{h_i=\frac{E_i}{A_i}.}
\]

Pour \(A_i=0\) :

\[
h_i=0.
\]

Le prototype se déplace :

\[
\boxed{C_i'=C_i+R_{geo}h_i.}
\]

Les deux histoires sont recentrées par le même \(h_i\). Le parent vérifie alors :

\[
E_i'=0,
\]

\[
G_i'=G_i-\frac{\|E_i\|^2}{A_i}=P_{struct,i}.
\]

L'axe latent est invariant au déplacement commun :

\[
(\mu_+'-\mu_-')=(\mu_+-\mu_-).
\]

Aucun coefficient de vitesse indépendant n'est introduit ; l'inertie provient de l'EMA.

---

## 8. Croissance horizontale

### 8.1. Gain exact de séparation

Pour les deux branches de masse positive :

\[
P_\pm=G^\pm-\frac{\|E^\pm\|^2}{A^\pm}.
\]

Le gain exact de matérialisation de la bifurcation est :

\[
\boxed{J_{split}=P_{struct}-(P_++P_-)}
\]

et :

\[
\boxed{
J_{split}
=
\frac{A^+A^-}{A}\|\mu_+-\mu_-\|^2
\ge0.
}
\]

La Cellule produit une **demande géométrique de mitose** si et seulement si :

\[
\boxed{J_{split}>0.}
\]

Cette demande n'implique aucune exécution sans autorisation économique.

### 8.2. Matérialisation

Définir :

\[
\mu=\frac EA,
\qquad
\delta_+=\mu_+-\mu,
\qquad
\delta_-=\mu_--\mu.
\]

Alors :

\[
A^+\delta_+ + A^-\delta_-=0.
\]

Les deux prototypes matérialisés sont :

\[
\boxed{C_+=C+R_{geo}\delta_+}
\]

\[
\boxed{C_-=C+R_{geo}\delta_-.}
\]

Chaque branche devient le noyau parent de sa Cellule, recentré autour du nouveau prototype :

\[
A_{child,\pm}=A^\pm,
\]

\[
E_{child,\pm}=E^\pm-A^\pm\delta_\pm=0,
\]

\[
G_{child,\pm}
=
G^\pm-2\delta_\pm\cdot E^\pm+A^\pm\|\delta_\pm\|^2
=
P_\pm.
\]

La mitose conserve la masse et réduit la puissance structurelle totale de :

\[
\boxed{J_{split}.}
\]

### 8.3. Mère et fille

La branche \(+\) est, par convention constitutionnelle, la continuation de la mère :

\[
\boxed{\iota_+=\iota_{mother}.}
\]

La branche \(-\) reçoit une identité neuve :

\[
\boxed{\iota_-=\operatorname{newIdentity}().}
\]

Cette asymétrie est exclusivement identitaire. Elle n'intervient dans aucune distance ni aucun gain.

Les deux Cellules initialisent leur prochaine bifurcation par un split neutre de leur noyau hérité. Ainsi :

\[
\boxed{J_{split,+}=J_{split,-}=0.}
\]

Une preuve historique ne peut justifier deux mitoses successives.

La mémoire verticale attachée à la mère reste attachée à son identité. La fille commence avec une mémoire verticale de propriétaire nulle. Aucune preuve verticale n'est copiée.

## 9. Émission verticale

La sortie verticale n'est ni l'entrée brute, ni le prototype \(C_k\), ni l'erreur instantanée \(e_k\), ni l'identité seule.

Avant l'injection courante, lorsque le noyau parent et les deux branches possèdent une masse positive, définir les formes internes de la Cellule active :

\[
\boxed{\delta_k^+=\mu_k^+-\mu_k,}
\qquad
\boxed{\delta_k^-=\mu_k^--\mu_k.}
\]

Elles représentent les deux variantes locales que la Cellule sait déjà distinguer relativement à son propre centre.

Si aucune forme interne n'est encore définie, adopter la convention exacte :

\[
\boxed{\delta_k^+=\delta_k^-=0.}
\]

La Cellule peut alors transmettre une preuve de masse positive, mais aucun contenu géométrique inventé.

### 9.1. Branche reconnue

La branche verticale \(\sigma_k\in\{+,-\}\) est celle désignée par le même test signé que le routage interne pré-injection.

En cas d'égalité exacte, la branche \(+\) est retenue par convention déterministe immuable pour conserver une émission unique.

Le contenu transmis est :

\[
\boxed{Y_L=\delta_k^{\sigma_k}.}
\]

Sa pertinence est :

\[
\boxed{r_{\uparrow,L}=r_k^{learn}=r_{in,L}a_k.}
\]

L'émission complète est :

\[
\boxed{
\mathcal E_L
=
(\iota_k,Y_L,r_{\uparrow,L}).
}
\]

L'identité indique la source ; elle n'entre pas dans la géométrie du récepteur.

### 9.2. Ordre causal

Pour chaque présentation :

1. lire les états pré-injection ;
2. calculer les reconnaissances et sélectionner \(k\) ;
3. déterminer \(\sigma_k\) et émettre \(Y_L\) ;
4. injecter la preuve dans \(H_k^\pm\) ;
5. déplacer et recentrer la Cellule.

La Cellule affirme d'abord ce qu'elle savait ; elle apprend ensuite l'observation courante.

### 9.3. Invariance au mouvement

Après déplacement commun \(\Delta\) de la Cellule :

\[
\mu' = \mu-\Delta,
\qquad
\mu_\pm'=\mu_\pm-\Delta,
\]

et donc :

\[
\boxed{\delta_\pm'=\delta_\pm.}
\]

La dérive de \(C_k\) ne change pas le contenu transmis.

### 9.4. Récepteur

Si une couche supérieure existe, elle reçoit :

\[
I_{L+1}=Y_L,
\qquad
r_{in,L+1}=r_{\uparrow,L}
\]

et applique exactement la même géométrie horizontale.

L'identité inférieure accompagne la transmission comme métadonnée de source mais n'est ni une coordonnée d'entrée ni une identité de Cellule supérieure.

Si aucune couche supérieure n'existe, l'émission alimente le bourgeon terminal.

---

## 10. Bourgeon vertical terminal

Seule la couche terminale possède un bourgeon vertical. Il est distinct du bourgeon racine du réseau : le premier teste une abstraction inter-identitaire, le second permet l’incarnation initiale du flux extérieur.

Le bourgeon est une hypothèse statistique de future couche :

- il n'est pas une couche ;
- il ne possède aucune Cellule matérialisée ;
- il ne transmet rien plus haut ;
- il entretient une bifurcation latente ;
- il disparaît lorsqu'il est consommé.

Il reçoit :

\[
(Y,r,\iota)
=
(Y_L,r_{\uparrow,L},\iota_k).
\]

### 10.1. Noyaux latents

Le bourgeon entretient :

\[
H_B^+=(A_B^+,E_B^+,G_B^+),
\qquad
H_B^-=(A_B^-,E_B^-,G_B^-).
\]

Son parent est :

\[
A_B=A_B^++A_B^-,
\qquad
E_B=E_B^++E_B^-,
\qquad
G_B=G_B^++G_B^-.
\]

La première preuve positive est répartie neutralement. Ensuite, le bourgeon applique la même loi signée que la section 7.1, en remplaçant l'erreur cellulaire par \(Y\).

On obtient :

\[
\boxed{r_B^++r_B^-=r.}
\]

Les deux noyaux sont mis à jour par le noyau universel.

### 10.2. Gain du bourgeon

Pour les deux branches de masse positive :

\[
\nu_B^+=\frac{E_B^+}{A_B^+},
\qquad
\nu_B^-=\frac{E_B^-}{A_B^-}.
\]

Le gain quadratique exact de la coupure latente est :

\[
\boxed{
J_B
=
\frac{A_B^+A_B^-}{A_B}
\|\nu_B^+-\nu_B^-\|^2
\ge0.
}
\]

\(J_B>0\) établit qu'une future couche pourrait matérialiser une séparation dans le flux montant. Il n'établit pas encore que cette séparation est transversale à plusieurs identités.

## 11. Concordance verticale entre identités

L'identité n'intervient qu'après le routage géométrique autonome du bourgeon.

Pour chaque identité source \(i\), le bourgeon entretient deux premiers moments pondérés :

\[
(\Lambda_i^+,F_i^+),
\qquad
(\Lambda_i^-,F_i^-).
\]

Si l'émission courante provient de \(i\), les mêmes parts \(r_B^\pm\) que celles du bourgeon alimentent :

\[
\Lambda_i^\pm
\leftarrow
\Lambda_i^\pm+\alpha_\eta(r_B^\pm-\Lambda_i^\pm),
\]

\[
F_i^\pm
\leftarrow
F_i^\pm+\alpha_\eta(r_B^\pm Y-F_i^\pm).
\]

Les autres identités oublient sans injection.

Avec une initialisation nulle et le même \(\alpha_\eta\) :

\[
\boxed{\sum_i\Lambda_i^\pm=A_B^\pm}
\]

et :

\[
\boxed{\sum_iF_i^\pm=E_B^\pm.}
\]

### 11.1. Distinction propre à un propriétaire

Si \(\Lambda_i^+>0\) et \(\Lambda_i^->0\), définir :

\[
m_i^+=\frac{F_i^+}{\Lambda_i^+},
\qquad
m_i^-=\frac{F_i^-}{\Lambda_i^-},
\]

\[
\boxed{d_i=m_i^+-m_i^-}
\]

et :

\[
\boxed{
w_i
=
\frac{\Lambda_i^+\Lambda_i^-}
{\Lambda_i^++\Lambda_i^-}.
}
\]

Sinon :

\[
w_i=0,
\qquad
d_i=0.
\]

La puissance de séparation réellement portée par le propriétaire est :

\[
\boxed{j_i=w_i\|d_i\|^2.}
\]

La simple visite des deux branches ne suffit pas : une source constante ou une distinction infinitésimale porte une puissance nulle ou infinitésimale.

### 11.2. Concordance croisée sans matrice

Poser :

\[
W=\sum_iw_i,
\]

\[
V=\sum_iw_id_i,
\]

\[
U=\sum_iw_i^2\|d_i\|^2.
\]

Pour \(W>0\), définir :

\[
\boxed{
P_\uparrow
=
\frac{\|V\|^2-U}{W}.
}
\]

Pour \(W=0\) :

\[
\boxed{P_\uparrow=0.}
\]

Équivalent explicatif :

\[
\boxed{
P_\uparrow
=
\frac{
2\sum_{i<j}w_iw_j\,d_i\cdot d_j
}{\sum_iw_i}.
}
\]

Aucun couple n'est stocké ; la forme \((W,V,U)\) se calcule en une passe.

\(P_\uparrow\) est une concordance signée :

- \(P_\uparrow>0\) : distinctions concordantes entre plusieurs identités ;
- \(P_\uparrow=0\) : aucune répétition croisée nette ;
- \(P_\uparrow<0\) : distinctions contradictoires.

Il n'est jamais rectifié dans l'état par \(\max(P_\uparrow,0)\).

Il vérifie :

\[
\boxed{
P_\uparrow
\le
\sum_iw_i\|d_i\|^2
=
\sum_i j_i.
}
\]

La concordance ne fabrique donc aucune puissance géométrique.

### 11.3. Cas structurants

- une seule identité : \(P_\uparrow=0\) ;
- une distinction et une source constante : \(P_\uparrow=0\) ;
- distinctions orthogonales : \(P_\uparrow=0\) ;
- distinctions opposées : \(P_\uparrow<0\) ;
- distinctions identiques portées par au moins deux identités : \(P_\uparrow>0\) ;
- inversion globale des labels \(+\) et \(-\) : \(P_\uparrow\) inchangé.

## 12. Croissance verticale

Le bourgeon produit une **demande géométrique de naissance verticale** si et seulement si :

\[
\boxed{J_B>0}
\]

et :

\[
\boxed{P_\uparrow>0.}
\]

Les deux conditions ont des fonctions distinctes :

- \(J_B\) prouve qu'une séparation matérialisable existe dans le flux montant ;
- \(P_\uparrow\) prouve que cette séparation est portée de manière concordante par plusieurs identités.

Aucun produit arbitraire \(J_BP_\uparrow\) n'est introduit.

La demande n'implique aucune naissance sans autorisation économique.

### 12.1. Matérialisation d'une nouvelle couche

Lorsque la naissance est autorisée, les deux branches du bourgeon deviennent les deux Cellules fondatrices de la nouvelle couche.

Le noyau géométrique historique de la nouvelle couche est :

\[
\boxed{H_{R,L+1}=H_B^++H_B^-.}
\]

Les centres fondateurs sont :

\[
\boxed{C_{L+1,+}=\nu_B^+}
\]

\[
\boxed{C_{L+1,-}=\nu_B^-.}
\]

Chaque branche est recentrée autour de son centre :

\[
E_{founder,\pm}=0,
\]

\[
G_{founder,\pm}
=
G_B^\pm-
\frac{\|E_B^\pm\|^2}{A_B^\pm}.
\]

Les deux fondatrices reçoivent des identités neuves propres à la nouvelle couche. Chacune initialise sa bifurcation interne par un split neutre.

La preuve du bourgeon est consommée :

- \(H_B^\pm\) sont transférés puis détruits comme états de bourgeon ;
- les statistiques \((\Lambda_i^\pm,F_i^\pm)\) sont détruites ;
- l'ancienne couche cesse d'être terminale ;
- un bourgeon neuf et nul est créé au-dessus de la nouvelle couche.

Une même preuve ne peut créer deux couches.

### 12.2. Récursion

Une fois née, toute couche applique le même contrat :

\[
\text{reconnaissance}
\rightarrow
\text{WTA}
\rightarrow
\text{EMA locale}
\rightarrow
\text{forme interne}
\rightarrow
\text{émission verticale}.
\]

Il n'existe aucun type spécial de couche profonde.

## 13. Effets topologiques sur la preuve verticale

### 13.1. Mitose d'une source terminale

Lors d'une mitose horizontale :

- la mère conserve son identité et sa mémoire de propriétaire déjà accumulée ;
- la fille reçoit une identité neuve ;
- la fille commence avec \(\Lambda^\pm=0\) et \(F^\pm=0\) ;
- aucune preuve verticale historique n'est copiée ;
- le bourgeon existant n'est pas réinitialisé par principe.

L'ancienne preuve de la mère décrit correctement sa biographie passée et décroît sous le même \(\alpha_\eta\). La fille doit produire réellement une distinction avant de participer à la concordance verticale.

### 13.2. Changement d'échelle du repère inférieur

Les formes \(Y=\delta^\sigma\) vivent dans les coordonnées normalisées de la couche inférieure.

Si son échelle passe de \(R\) à \(R'\), poser :

\[
\alpha=\frac R{R'}.
\]

Les histoires internes des Cellules de cette couche sont d’abord transportées dans la nouvelle unité :

\[
\boxed{E_i^\pm\leftarrow\alpha E_i^\pm,
\qquad
G_i^\pm\leftarrow\alpha^2G_i^\pm.}
\]

Le flux vertical et les états déjà appris au-dessus sont transportés par la même dilatation commune :

\[
C\leftarrow\alpha C,
\qquad
S\leftarrow\alpha S,
\qquad
Q\leftarrow\alpha^2Q.
\]

Ce transport est un changement de repère, pas un apprentissage. Si \(R=0\) et \(R'>0\), les anciens moments normalisés nécessairement nuls restent nuls et l’observation courante est injectée dans la nouvelle unité. Sous l’EMA exacte, un rayon strictement positif ne redevient pas exactement nul en temps fini ; un tel état constitue une violation d’invariant numérique, non un cas comportemental.

## 14. Décroissance topologique — mort cellulaire

La mort est un opérateur topologique fondamental :

\[
\boxed{\operatorname{death}_L(i):1\longrightarrow0.}
\]

Elle est exécutée par la couche qui contient la Cellule. Le réseau ne connaît ni l’identité de la victime, ni son prototype, ni son histoire.

### 14.1. Effet local

La mort de la Cellule \(i\) détruit irréversiblement :

\[
(C_i,\iota_i,H_i^+,H_i^-)
\]

et tout état cellulaire qui lui est exclusivement attaché.

Elle ne provoque :

- aucun apprentissage ;
- aucun déplacement d’une Cellule survivante ;
- aucune redistribution de son histoire ;
- aucune fusion implicite ;
- aucun remboursement économique.

\[
\boxed{\text{Ce qui n’est plus représenté est réellement perdu.}}
\]

La maintenance de la Cellule supprimée cesse à la présentation suivante.

### 14.2. Valeur géométrique conservatrice d’une Cellule

La couche juge la perte de ses propres constituants.

Pour une victime \(i\) et une survivante \(j\), définir le déplacement normalisé :

\[
\boxed{h_{ij}=\frac{C_j-C_i}{R_{geo}}}
\]

avec la convention dégénérée de la section 4.1.

Si la preuve quadratique de \(i\) était représentée par le seul substitut \(j\), la variation de puissance serait :

\[
\boxed{
\Delta_{i\to j}
=
A_i\|h_{ij}\|^2-2E_i\cdot h_{ij}.
}
\]

La valeur de conservation cellulaire est :

\[
\boxed{
K_{i\mid L}
=
\left[
\min_{j\ne i}\Delta_{i\to j}
\right]_+.
}
\]

Après le déplacement et le recentrage complets de la section 7.2, \(E_i=0\), donc :

\[
\boxed{
K_{i\mid L}
=
A_i\min_{j\ne i}\|h_{ij}\|^2.
}
\]

Cette quantité est :

- calculée uniquement par la couche ;
- exprimée dans la même unité que \(J_{split}\) ;
- un contrefactuel de conservation, non une redistribution effective ;
- conservatrice, car elle exige qu’une seule survivante porte toute la preuve de la victime.

### 14.3. Dernière Cellule d’une couche

Une couche ne décide pas seule de tuer sa dernière Cellule active. Cette transformation ferait disparaître un constituant immédiat du réseau et relève donc de la juridiction du réseau.

La couche peut seulement exposer une demande agrégée de disparition. Elle ne révèle aucune identité cellulaire. Pour toute couche existante, y compris \(L=0\) :

\[
\boxed{N_{cell,L}=0\Longrightarrow\operatorname{truncate}(L).}
\]

avec :

\[
\boxed{\operatorname{truncate}(L)=\text{supprimer toutes les couches }k\ge L.}
\]

En particulier :

\[
\boxed{\operatorname{truncate}(0)\Longrightarrow\mathcal L=\varnothing.}
\]

Le réseau et son bourgeon racine persistent ; aucune couche ne bénéficie d’une exception de survie.

### 14.4. Effet sur un bourgeon terminal

La mort d’une Cellule terminale supprime une identité source sans fournir les seconds moments par propriétaire nécessaires à une soustraction exacte du bourgeon.

Par nécessité informationnelle, une mort cellulaire dans la couche terminale invalide donc la preuve verticale candidate :

\[
\boxed{H_B^+=H_B^-=(0,0,0)}
\]

et toutes les statistiques :

\[
\boxed{(\Lambda_i^\pm,F_i^\pm)=0.}
\]

## 15. Capital géométrique réalisé

La géométrie commune sert de numéraire aux transformations topologiques.

### 15.1. Capital d’une couche

Pour chaque Cellule de masse \(A_i>0\), poser :

\[
z_i=\frac{C_i-M_R}{R_{geo}},
\qquad
\mu_i=\frac{E_i}{A_i},
\qquad
\boxed{p_i=z_i+\mu_i.}
\]

Sous la convention dégénérée, le capital est nul tant qu’aucune différenciation normalisée n’est définie.

Poser :

\[
A_\Sigma=\sum_iA_i,
\qquad
\bar p_L=\frac{\sum_iA_ip_i}{A_\Sigma}.
\]

Le capital géométrique réalisé de la couche est :

\[
\boxed{
\Gamma_L
=
\sum_iA_i\|p_i-\bar p_L\|^2.
}
\]

Forme agrégée :

\[
\boxed{
\Gamma_L
=
\sum_iA_i\|p_i\|^2
-
\frac{\left\|\sum_iA_ip_i\right\|^2}{A_\Sigma}.
}
\]

Le centroïde effectif \(p_i\) est invariant au partage de jauge entre déplacement du prototype et résidu moyen :

\[
z_i\mapsto z_i+\Delta,
\qquad
\mu_i\mapsto\mu_i-\Delta
\Longrightarrow
p_i\mapsto p_i.
\]

Après recentrage complet, \(E_i=0\), donc \(p_i=z_i\).

### 15.2. Mitose

Une mitose conserve la masse et le barycentre du parent. Elle augmente exactement le capital de sa couche de :

\[
\boxed{
\Gamma_L'-\Gamma_L=J_{split}.
}
\]

Ainsi, \(J_{split}\) est le gain géométrique réalisé d’une mitose.

### 15.3. Naissance verticale

Les deux fondatrices d’une nouvelle couche sont matérialisées aux moyennes du bourgeon.

Dans l’espace normalisé de la couche mère, la valeur géométrique de la séparation créée est exactement :

\[
\boxed{
J_B
=
\frac{A_B^+A_B^-}{A_B}
\|\nu_B^+-\nu_B^-\|^2.
}
\]

Après matérialisation, la nouvelle couche calcule son propre capital \(\Gamma_{L+1}\) dans son repère normalisé courant. En général :

\[
\boxed{\Gamma_{L+1}^{birth}\ne J_B}
\]

car la nouvelle couche dérive sa propre échelle \(R_{geo,L+1}\). L’égalité ne vaut que dans la jauge particulière où cette nouvelle échelle vaut \(1\).

Ce n’est pas une rupture de comparabilité : \(J_B\) et \(\Gamma_{L+1}\) ont la même unité géométrique standardisée, mais l’un évalue l’acte de naissance dans le repère du parent et l’autre évalue l’état réalisé dans le repère propre du nouveau-né.

\(P_\uparrow>0\) reste l’admissibilité transversale ; \(J_B\) est la valeur géométrique de la naissance demandée.

### 15.4. Mort cellulaire

En général :

\[
\Gamma_L-\Gamma_{L\setminus i}
\]

n’est pas la perte de représentation causée par la mort de \(i\), car ce calcul retire également la preuve détruite du domaine mesuré.

La valeur de conservation d’une Cellule reste donc \(K_{i\mid L}\), et non la variation marginale de \(\Gamma_L\).

### 15.5. Capital exposé au réseau

Une couche expose seulement :

\[
\boxed{(\Gamma_L,M_L,\text{demandes agrégées}).}
\]

Le réseau ne reçoit aucun état cellulaire.

Le capital géométrique réalisé exposé au réseau est :

\[
\boxed{
\Gamma_{\mathcal N}
=
\sum_{L=0}^{L_{max}}\Gamma_L.
}
\]

\(\Gamma_{\mathcal N}\) décrit l’état réalisé ; il n’est pas un potentiel unique dont chaque demande serait nécessairement la différence exacte. En particulier, une naissance est jugée par \(J_B\) dans le repère de sa couche mère, puis la nouvelle couche expose son propre \(\Gamma\) dans son repère courant.

Cette additivité exprime le choix constitutionnel suivant :

\[
\boxed{\text{Chaque niveau d’abstraction matérialisé constitue une connaissance supplémentaire.}}
\]

La pertinence montante est issue du même flux racine, les distances sont normalisées par chaque couche et toutes les mémoires utilisent le même \(\alpha_\eta\). Les \(\Gamma_L\), \(J_{split}\), \(J_B\) et \(K_{i\mid L}\) sont donc commensurables.

### 15.6. Suppression d’une couche et troncature

La suppression d’une couche \(L\) impose la continuité de la hiérarchie :

\[
\boxed{
\operatorname{truncate}(L)
=
\text{supprimer toutes les couches }k\ge L.
}
\]

La perte géométrique exposée au réseau est :

\[
\boxed{
K_{\ge L}
=
\sum_{k=L}^{L_{max}}\Gamma_k.
}
\]

Le réseau calcule cette somme uniquement à partir des valeurs de ses constituants immédiats. Il n’inspecte aucune Cellule.

La suppression d’un suffixe est définie pour tout \(L\ge0\). Pour \(L=0\), elle détruit toute la hiérarchie et sa perte vaut le capital total réalisé. Lorsque \(\mathcal L=\varnothing\), le capital géométrique réalisé du réseau est nul ; le bourgeon racine contient une preuve de future incarnation, non un capital de couche matérialisé.

## 16. Économie de maintenance et transformations

### 16.1. Empreinte persistante

Soit \(M(\mathcal A)\) l’empreinte de maintenance de l’état persistant \(\mathcal A\).

Elle est la somme des coûts positifs et finis des objets réellement entretenus :

- réseau ;
- bourgeon racine ;
- couches ;
- Cellules ;
- bourgeon terminal ;
- statistiques persistantes associées.

Les fonctions physiques de coût restent abstraites. Elles dépendent de la représentation effective, pas d’un facteur temporel arbitraire.

Dans la réalisation de référence, toute empreinte et tout budget brut appartiennent à \(\mathbb N\) et sont calculés par arithmétique entière exacte :

\[
\boxed{
M_{units}(\mathcal A)\in\mathbb N,
\qquad
B_{units}\in\mathbb N.
}
\]

L’unité de référence est l’octet logique. Un réel persistant coûte quatre unités en `f32` et huit unités en `f64`; les identités et compteurs possèdent des coûts entiers propres. Cette convention évite qu’un réseau très grand perde la capacité de distinguer l’ajout d’un objet parce que son budget dépasserait la résolution entière d’un flottant.

La solvabilité exige :

\[
\boxed{M(\mathcal A)\le B_{units}.}
\]


### 16.1.1. Budget ergonomique en cellules équivalentes

Le budget brut entier reste la quantité exécutée par le moteur. L’interface utilisateur peut toutefois l’exprimer en cellules équivalentes, afin d’abstraire la résolution numérique et de conserver une intuition démographique stable.

Pour le modèle de référence, définir le paquet cellulaire terminal :

\[
\boxed{
U_{cell}(D,p)
=
M_{cell}(D,p)
+
M_{owner}(D,p).
}
\]

Il représente une Cellule et son enregistrement propriétaire dans le bourgeon terminal. Définir également :

\[
M_{root}=M_{network}+M_{root\ bud},
\]

\[
M_{shell}=M_{layer}+M_{bud\ base}.
\]

Pour un budget ergonomique \(B\ge0\) :

\[
\boxed{
B_{units}(B)=
\begin{cases}
M_{root},&B=0,\\
M_{root}+M_{shell}+\lfloor B\,U_{cell}\rfloor,&B>0.
\end{cases}
}
\]

L’arrondi est dirigé vers le bas : l’interface ne crée jamais davantage d’empreinte que celle demandée. Le même \(B\) produit des budgets bruts différents en `f32` et `f64`, mais conserve une capacité abstraite comparable.

Dans l’API de référence, cette capacité ergonomique est exposée sous le nom `budget`, tandis que la quantité entière effectivement exécutée est exposée sous le nom `budget_units`. Exactement l’une de ces deux valeurs est fournie à la construction ou au rechargement ; toute l’économie interne opère exclusivement sur `budget_units`.

### 16.2. Coût d’une création

Pour une transformation élémentaire de création \(q\) :

\[
\boxed{
 c_q
 =
 \left[
 M(\mathcal A_q')-M(\mathcal A)
 \right]_+.
}
\]

Ainsi :

- l’incarnation racine paie l’empreinte persistante ajoutée par la première couche, sa Cellule fondatrice et son bourgeon vertical ;
- une mitose paie l’empreinte persistante ajoutée par une Cellule ;
- une naissance verticale paie la différence exacte entre l’état avec bourgeon et l’état avec nouvelle couche, fondatrices et nouveau bourgeon ;

### 16.3. Mort et perte sèche

Pour une mort ou une troncature :

\[
\boxed{c_{death}=c_{truncate}=0.}
\]

Une destruction :

- ne rembourse aucun coût passé ;
- ne finance aucune création au même pas ;
- réduit seulement la maintenance à partir de la présentation suivante.

### 16.4. Juridictions économiques

La Cellule expose à sa couche :

\[
\boxed{J_{split,i}.}
\]

La couche :

- compare les demandes de ses Cellules ;
- calcule les \(K_{i\mid L}\) ;
- choisit en interne les mitoses et les victimes ;
- expose au réseau uniquement une demande agrégée et son coût ;
- expose son capital \(\Gamma_L\) et sa maintenance \(M_L\).

Le réseau :

- entretient le bourgeon racine et lui présente l’entrée lorsque la hiérarchie est vide ;
- matérialise la première couche lorsque l’incarnation est solvable ;
- autorise ou refuse les demandes agrégées de couches ;
- arbitre une naissance verticale ;
- décide de supprimer une couche ou un suffixe ;
- restaure la solvabilité globale à partir d’offres agrégées de contraction ;
- ne connaît jamais l’existence individuelle des Cellules.

Une couche peut exposer une offre de contraction élémentaire :

\[
\boxed{
\mathcal O_L^-
=
(K_L^-,\Delta M_L^-),
\qquad
K_L^-=\min_i K_{i\mid L},
}
\]

avec l’identité de la victime conservée strictement à l’intérieur de la couche.

### 16.5. Réallocation horizontale

Si une mitose \(q\) est bloquée par la capacité et si la mort d’une seule Cellule permettrait de rendre sa création soutenable à la présentation suivante, la couche peut comparer :

\[
\boxed{
R_{qi}=J_{split,q}-K_{i\mid L}.
}
\]

Une mort volontaire de réallocation n’est admissible que si la meilleure marge interne est strictement positive :

\[
\boxed{
\max_{q\ne i}R_{qi}>0.
}
\]

Seule la mort est exécutée. La mitose doit être redemandée et réévaluée à la présentation suivante.

La pénurie ne crée aucune demande géométrique ; elle compare une demande déjà constituée à la perte géométrique d’une capacité existante.

### 16.6. Interdiction de la réallocation verticale par troncature

Dans une hiérarchie linéaire, une demande verticale est portée par le bourgeon de la couche terminale. Toute troncature susceptible de libérer une couche détruit nécessairement cette couche terminale ou son support causal.

\[
\boxed{\text{Une troncature ne peut pas être exécutée pour financer une naissance verticale future.}}
\]

La solvabilité peut toujours imposer une troncature. Mais une preuve détruite ne peut pas être redemandée au pas suivant : le demandeur n’existe plus. Il n’existe donc aucun marché \(J_B-K_{\ge L}\).

### 16.7. Solvabilité forcée

Si :

\[
M(\mathcal A)>B_{units},
\]

la destruction n’est pas un gain : elle est obligatoire.

Chaque parent choisit parmi les destructions de ses constituants immédiats celle qui expose la plus faible perte géométrique admissible. Le réseau compare uniquement :

- les offres agrégées de contraction des couches ;
- les pertes de suppression de couches ou de suffixes.

Les sacrifices sont répétés jusqu’au retour à :

\[
\boxed{M(\mathcal A)\le B_{units}.}
\]

Le réseau peut tronquer depuis n’importe quel niveau existant, y compris \(L=0\). Si la hiérarchie est déjà vide et que \(M_{min}>B_{units}\), aucune destruction supplémentaire n’est définie : Auxein est inexécutable.

Les égalités exactes sont fermées par un ordre déterministe immuable.

### 16.8. Primauté verticale en égalité exacte

Après admissibilité et avant veto économique, si une naissance verticale et une demande horizontale agrégée ont exactement la même valeur positive :

\[
\boxed{J_B=J_{split}>0,}
\]

la naissance verticale est ordonnée en premier.

Cette convention exprime la mission architecturale d’Auxein : produire de l’abstraction lorsque la géométrie ne départage pas les deux formes de croissance.

Elle :

- n’ajoute aucun \(\varepsilon\) ;
- ne multiplie aucune pression ;
- ne modifie aucune inégalité ;
- ne supprime jamais le veto de solvabilité.

### 16.9. Séquencement des créations et des pertes volontaires

Après l’injection, chaque couche expose au plus sa meilleure proposition agrégée dans chaque catégorie pertinente. Le réseau exécute itérativement la meilleure création positive et payable, puis ne recalcule que les propositions affectées par la transformation.

Une proposition non payable ne bloque pas une proposition de valeur inférieure mais payable. Une demande consommée ne réapparaît pas pendant la même présentation. Un objet créé ne demande rien avant la présentation suivante.

La solvabilité forcée peut exécuter autant de destructions que nécessaire avant perception. En revanche :

\[
\boxed{\text{au plus une mort volontaire de réallocation est exécutée par présentation.}}
\]

Cette mort met fin à l’arbitrage topologique du pas. Elle ne rembourse rien et la création espérée doit être redemandée après une nouvelle perception.

### 16.10. Conséquence de croissance sans seuil

La condition géométrique de mitose reste exactement :

\[
\boxed{J_{split}>0.}
\]

Aucune masse minimale, maturité, temporisation ou marge numérique n’est ajoutée. Par conséquent, lorsqu’un budget abondant autorise toute empreinte supplémentaire, une Cellule fraîche peut produire un gain strictement positif après une seule écriture informative et la topologie peut remplir rapidement sa capacité disponible.

\[
\boxed{\text{Un budget abondant autorise la saturation ; ce comportement n’est pas un défaut de spécification.}}
\]

## 17. Ordre causal d’une présentation

Une présentation suit :

1. **Solvabilité de l’état vivant** : calcul de la maintenance existante, morts ou troncatures forcées éventuelles.
2. **Présentation de l’entrée** : sélection du bourgeon racine si la hiérarchie est vide ; sinon propagation dans la topologie survivante.
3. **Lecture pré-injection** : reconnaissance, WTA, routages internes et émissions verticales.
4. **Injection statistique unique** : si \(\eta>0\), mises à jour EMA des couches, Cellules, bourgeon et propriétaires avec \(\alpha_\eta\).
5. **Mouvement et recentrage** : si \(\eta>0\), déplacement complet des Cellules actives.
6. **Construction des demandes et valeurs** : si \(\eta>0\), incarnation racine éventuelle, \(J_{split}\), \(J_B\), \(P_\uparrow\), \(K_{i\mid L}\), \(\Gamma_L\).
7. **Arbitrages locaux** : si \(\eta>0\), chaque couche sélectionne ses demandes, morts ou offres agrégées.
8. **Arbitrage du réseau** : si \(\eta>0\), autorisation des demandes de couches, naissance ou suppression volontaire de couches.
9. **Transformations** : matérialisations et morts volontaires autorisées, sans nouvelle perception ni nouvelle injection.

Résumé :

\[
\boxed{
\text{survivre}
\to
\text{percevoir}
\to
\text{mémoriser}
\to
\text{évaluer}
\to
\text{autoriser}
\to
\text{transformer}.
}
\]

À \(\eta=0\), les étapes 4 à 9 sont neutralisées ; la lecture diagnostique et la solvabilité forcée restent actives. La présentation courante est vue par une seule version de la topologie. Les objets créés après perception commencent leur maintenance et leur activité à la présentation suivante. Les objets morts cessent leur maintenance à la présentation suivante.

## 18. Fermeture

L’interface racine est fermée par :

\[
x
\rightarrow B_{\varnothing}
\rightarrow\text{incarnation solvable}
\rightarrow L_0\text{ avec une fondatrice}.
\]

La géométrie horizontale est fermée par :

\[
I
\rightarrow
a_i
\rightarrow k
\rightarrow r_k^{learn}
\rightarrow H_k^\pm
\rightarrow C_k
\rightarrow J_{split}
\rightarrow\text{mère et fille}.
\]

La géométrie verticale est fermée par :

\[
H_k^\pm
\rightarrow\delta_k^{\sigma_k}
\rightarrow H_B^\pm
\rightarrow J_B
\rightarrow(\Lambda_i^\pm,F_i^\pm)
\rightarrow P_\uparrow
\rightarrow\text{nouvelle couche}.
\]

La décroissance et l’économie géométrique sont fermées par :

\[
\boxed{
\begin{aligned}
\text{mitose} &: +J_{split},\\
\text{mort cellulaire} &: -K_{i\mid L},\\
\text{naissance verticale} &: +J_B,\\
\text{suppression d’un suffixe} &: -K_{\ge L},\\
\text{capital de couche} &: \Gamma_L,\\
\text{solvabilité} &: M(\mathcal A)\le B_{units}.
\end{aligned}
}
\]

Toutes les valeurs géométriques ont l’unité :

\[
\boxed{\text{masse de pertinence}\times\text{distance normalisée}^{2}.}
\]

## 19. Réalisation numérique et sérialisation de référence

### 19.1. Trois domaines numériques

La référence sépare explicitement :

\[
\boxed{
\begin{aligned}
\text{géométrie persistante} &: \mathrm{f32}\text{ ou }\mathrm{f64},\\
\text{calcul intermédiaire} &: \texttt{float Python},\\
\text{économie} &: \mathbb N\text{ exact}.
\end{aligned}
}
\]

Le format persistant et \(\eta\) appartiennent à l’état causal. Le budget et le modèle de maintenance appartiennent à l’environnement d’exécution.

### 19.2. Sérialisation causale stricte

Le moteur expose une représentation JSON-compatible de son état causal complet et ne réalise aucune entrée-sortie de fichier.

Le dictionnaire sérialisé contient au minimum :

- version du schéma et version du modèle ;
- dimension, format persistant, demi-vie et multiplicateur d’apprentissage `eta` ;
- indice de présentation et compteurs d’identités ou de couches ;
- bourgeon racine ;
- couches, Cellules, bourgeons terminaux et enregistrements propriétaires.

Il exclut :

- budget et modèle de maintenance ;
- traces, rapports et télémétrie ;
- flux extérieur et état de son générateur ;
- propositions, jetons et index éphémères.

\[
\boxed{\text{clés manquantes ou inconnues}\Longrightarrow\text{rejet}.}
\]

Le chargement vérifie les versions, dimensions, valeurs finies, identités, compteurs et invariants complets. Une sérialisation n’est autorisée qu’à une frontière de présentation, sans arbitrage local pendant.

Le format numérique d’un état chargé ne peut pas être converti silencieusement. Toute conversion `f64` vers `f32`, ou inversement, est une opération extérieure explicite qui produit un nouvel état validé.

## 20. Note de réalisation non comportementale — pièges d’implémentation

Les lois d’Auxein définissent des résultats mathématiques, non l’obligation de les recalculer depuis zéro à chaque usage. Une implémentation conforme doit préserver exactement la causalité, les égalités déterministes et les valeurs géométriques, mais éviter les recomputations imbriquées accidentelles.

Une règle numérique domine toutes les autres: **fermer les zéros structurellement démontrés; ne jamais fermer une valeur seulement petite**.

La fermeture dépend exclusivement de la résolution du format persistant choisi. Elle ne constitue ni une marge comportementale, ni une maturité, ni un seuil de croissance. Lorsqu’une forme algébrique stable ou positive existe, elle est préférée à une soustraction suivie d’une fermeture.

1. **Reconnaissance et écriture.** Toutes les Cellules évaluent leur reconnaissance ; seule la gagnante calcule son routage interne et écrit.

2. **Agrégats de couche.** Sommes, barycentres, rayons, capitaux et autres dérivés communs sont calculés une fois par version pertinente de l’état puis réutilisés. Une propriété dérivée ne reconstruit pas récursivement des objets agrégés complets lorsqu’une expression directe suffit.

3. **Valeurs de conservation.** La recherche exacte d’un substitut est évaluée au plus une fois par victime et par état pertinent. La répéter pour chaque création, chaque victime puis chaque substitut transforme facilement un coût quadratique légitime en boucle cubique accidentelle. Un index spatial éphémère et exact est permis ; il ne devient ni voisinage persistant, ni relation apprise, ni état canonique.

4. **Concordance croisée.** La forme agrégée \(\bigl(\|V\|^2-U\bigr)/W\) est une identité mathématique, mais sa soustraction flottante peut laisser un résidu de signe arbitraire lorsque les termes s’annulent. Comme le signe strict de \(P_\uparrow\) autorise une naissance verticale, une implémentation ne doit pas créer de preuve à partir de cette annulation numérique. Évaluer de préférence la forme croisée en une passe : avant d’ajouter le propriétaire \(i\), accumuler \(C\leftarrow C+2w_iV\cdot d_i\), puis \(V\leftarrow V+w_id_i\), et poser \(P_\uparrow=C/W\). Cette réalisation reste en \(O(ND)\), ne stocke aucun couple et garantit notamment \(P_\uparrow=0\) lorsqu’un seul propriétaire porte une distinction.

5. **Puissance quadratique résolue.** Les expressions

   \[
   P_{struct}=Q-\frac{\|S\|^2}{W}
   \]

   et

   \[
   R_{geo}^2=\frac{P_{struct}}{W}
   \]

   soustraient des moments persistants arrondis indépendamment. La projection dans `f32` ou `f64` peut imposer un \(Q\) immédiatement supérieur à \(\|S\|^2/W\) afin de préserver l’invariant, même lorsque le flux est un point exact. Ce résidu ne doit devenir ni rayon, ni capital, ni preuve. Une implémentation définit une unique primitive de **différence quadratique résolue**, dérivée de la résolution du format et de l’erreur des opérandes. Cette primitive ferme aussi bien un petit résidu positif qu’un petit résidu négatif lorsqu’ils sont numériquement indécidables. Toute valeur positivement résolue reste comportementale, quelle que soit sa taille absolue.

6. **Unicité de la réalisation numérique.** Une grandeur canonique ne possède pas plusieurs évaluations flottantes concurrentes. En particulier, le rayon est dérivé de la même puissance structurelle résolue que le noyau ; il n’est pas recalculé séparément sous une forme algébriquement équivalente. Deux expressions mathématiquement identiques peuvent sinon classer simultanément le même état comme dégénéré et non dégénéré.

7. **Recentrage stable.** La forme littérale

   \[
   Q'=Q-2\Delta\cdot S+W\|\Delta\|^2
   \]

   est canonique mais peut annuler trois termes proches. Après avoir conservé \(P_{struct}\), une réalisation stable calcule :

   \[
   S'=S-W\Delta,
   \qquad
   Q'=P_{struct}+\frac{\|S'\|^2}{W}.
   \]

   Lorsque \(\Delta=\mu=S/W\) est la propre moyenne du noyau, le résultat structurel est connu exactement :

   \[
   \boxed{S'=0,\qquad Q'=P_{struct}.}
   \]

   Ce zéro est construit directement ; il n’est pas recherché par soustraction composante par composante.

   Une limite demeure inhérente au stockage en moments bruts : après une translation extrême, si le nouveau \(P_{move}\) est si grand que \(P_{struct}\) tombe sous son ulp dans le format persistant, \((W,S,Q)\) ne peut plus représenter simultanément les deux puissances. La fermeture de \(P_{struct}\) à zéro dans ce cas relève de la résolution déclarée du format. Préserver davantage exigerait un autre état persistant, par exemple des moments centraux ; ce serait une modification architecturale et non une correction locale d’implémentation.

8. **Capital de couche.** La forme agrégée

   \[
   \sum_iA_i\|p_i\|^2-
   \frac{\left\|\sum_iA_ip_i\right\|^2}{A_\Sigma}
   \]

   est exacte mais sujette à l’annulation lorsque les représentations réalisées coïncident. Une implémentation évalue de préférence la forme centrée ou une récurrence barycentrique pondérée équivalente, composée uniquement de distances quadratiques non négatives. Des représentations identiques doivent produire exactement \(\Gamma_L=0\).

9. **Conservation après mouvement.** Les offres de mort et les réallocations sont construites après le déplacement et le recentrage complets. À cet instant, \(E_i=0\) canoniquement ; la référence évalue donc directement :

   \[
   \boxed{K_{i\mid L}=A_i\min_{j\ne i}\|h_{ij}\|^2.}
   \]

   Réintroduire le terme général \(-2E_i\cdot h_{ij}\) donnerait une valeur économique à un éventuel résidu flottant du recentrage, alors que ce résidu n’est pas une géométrie canonique post-mouvement.

10. **Assertions de l’invariant quadratique.** Une mise à jour EMA, une translation ou une projection de format peut laisser \(WQ-\|S\|^2\) légèrement négatif par accumulation de plusieurs arrondis. La tolérance d’assertion est bornée par la chaîne d’opérations et les échelles réellement manipulées ; elle n’est pas limitée arbitrairement à un ulp unique. Une violation au-delà de cette borne reste une erreur. Cette tolérance d’assertion ne produit jamais de valeur comportementale positive.

11. **Zéros vectoriels et frontières de phase.** Un premier moment global connu nul après recentrage ne doit pas être interprété comme une nouvelle direction à cause d’un résidu. En revanche, une différence entre deux branches, un produit scalaire de routage ou une distinction entre propriétaires n’est pas fermé uniquement parce qu’il est petit : ces valeurs peuvent représenter une géométrie réelle et déterminer légitimement une phase. Toute fermeture supplémentaire doit être justifiée par une identité exacte, puis soumise aux tests d’équivariance causale.

12. **Transformations et demandes.** Une demande consommée ne réapparaît pas au même pas ; un objet créé ne demande rien avant le suivant. Une transformation ne fait recalculer que les propositions dont les données ont changé.

13. **Assertions et télémétrie.** Les invariants complets restent vérifiables aux frontières causales, après transformation ou en mode diagnostic. Une primitive locale ne déclenche pas récursivement la validation complète du réseau. Les rapports détaillés ne sont pas calculés lorsqu’ils ne sont pas demandés.

14. **Profilage.** Le régime saturé, avec morts de réallocation et recherches de substituts actives, est obligatoire pour détecter les dégénérescences que les petites populations masquent.

Pour une couche de \(N\) Cellules en dimension \(D\) :

\[
\text{perception ordinaire}=O(ND),
\]

\[
\text{toutes les valeurs de conservation}=O(N^2D)
\]

peut être légitime. Une complexité :

\[
\boxed{O(N^3D)}
\]

indique généralement qu’une recherche de substitut ou un agrégat de couche est recalculé à l’intérieur d’une boucle qui le possède déjà.

Les caches et index d’exécution doivent être entièrement reconstruisibles depuis l’état canonique, ne porter aucune mémoire comportementale et ne jamais modifier une décision.
