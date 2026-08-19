# AUXEIN v0.2.0 — Canon mathématique et matériel

**Version : 0.2.0**  
**Statut : canon mathématique et matériel**

\[
\boxed{\text{la géométrie pousse ; l'économie autorise}}
\]

## 0. Contrat

Auxein est un réseau ordonné de `LAYER` autonomes. Chaque `LAYER` applique exactement la même transformation :

```text
présentation
→ concernement par les CELL
→ partage de masse
→ apprentissage local des CELL
→ inconnu vers Σ
→ contexte unique des reconnaissances
→ présentation de la LAYER suivante
```

Une `CELL` représente une connaissance directionnelle acquise. Elle se déclare concernée uniquement par sa propre géométrie. Plusieurs `CELL` peuvent se déclarer concernées simultanément.

Ce qu'aucune `CELL` ne reconnaît reste local à la `LAYER`, alimente une mémoire privée `Σ`, et peut devenir une nouvelle `CELL` lorsqu'il est récurrent.

Ce qui traverse une frontière de couche n'est pas une erreur, une provenance ni une branche par `CELL`. Une `LAYER` compresse toutes les valeurs qu'elle a effectivement reconnues pendant la présentation en **un unique noyau de contexte**. Ce noyau devient, lorsqu'il possède une diversité et une direction non nulles, l'unique présentation de la `LAYER` suivante.

Une présentation extérieure est une observation logique simultanée. Une présentation multi-vecteur affirme donc que ses vecteurs appartiennent au même contexte causal ; la découper en plusieurs appels successifs constitue plusieurs présentations différentes.

Principes normatifs :

1. aucune tâche externe, cible, classe, label ou loss supervisée ;
2. aucune `LAYER` ne lit l'état privé d'une autre ;
3. le seul contrat cognitif inter-couches est une présentation finie positive de noyaux centrés ;
4. toutes les `LAYER` appliquent exactement la même loi ;
5. aucun WTA, `top-k`, choix cognitif par identité ou ordre d'itération ;
6. plusieurs `CELL` peuvent être concernées simultanément par le même atome ;
7. ce qui est inconnu apprend horizontalement dans la `LAYER` courante ;
8. seules les connaissances effectivement reconnues participent au contexte vertical ;
9. une couche émet au plus un noyau de contexte par présentation ;
10. aucune responsabilité d'apprentissage, masse interne de `CELL`, identité ou provenance ne pondère la géométrie du contexte vertical ;
11. aucun seuil numérique arbitraire ni epsilon comportemental ;
12. aucune autorité cognitive de l'âge, d'une provenance ou d'une identité administrative ;
13. une présentation est causalement atomique ;
14. l'ordre des atomes d'une présentation n'a aucune autorité ;
15. un objet créé pendant un pas ne lit ni n'émet pour ce pas ;
16. la géométrie cognitive est définie indépendamment du budget ;
17. toute quantité exactement reconstructible peut rester éphémère ;
18. l'origine `0` n'est pas une connaissance : elle représente l'absence de direction cognitive canonique ;
19. une relation exactement symétrique de centre nul reste silencieuse plutôt que de recevoir un axe arbitraire ;
20. aucune matrice, covariance persistante, axe privilégié ou géométrie de second ordre n'appartient à Auxein.

---

## 1. Présentations et horloge

### 1.1 Présentation extérieure

Soit `D∈N*`. Une présentation extérieure est une liste finie non vide de vecteurs :

\[
\boxed{
\mathcal X=(x_1,\dots,x_n),
\qquad n>0,
\qquad x_s\in\mathbb R^D.
}
\]

Son ordre n'a aucune autorité. `NETWORK` lui associe à l'entrée de `L0` la présentation uniforme :

\[
\boxed{
\mathcal P_0=
\left\{\left(\frac1n,x_s,0\right)\right\}_{s=1}^{n}.
}
\]

La masse totale extérieure vaut donc exactement `1`.

### 1.2 Atome interne

Toute présentation reçue par une `LAYER` est une famille finie positive de noyaux-atome :

\[
\boxed{
\mathcal P=
\{X_s=(r_s,c_s,v_s)\}_{s\in S},
}
\]

avec :

\[
\boxed{
r_s>0,
\qquad c_s\in\mathbb R^D,
\qquad v_s\ge0,
\qquad 0<\sum_s r_s\le1.
}
\]

Interprétation :

- `r_s` : masse causale de l'atome ;
- `c_s` : centre vectoriel présenté ;
- `v_s` : dispersion scalaire interne autour de ce centre.

Une entrée extérieure `x` est donc exactement le cas dégénéré `(r,x,0)`.

Les atomes de géométrie exactement identique `(c,v)` sont coalescés par somme de masse avant tout calcul. Leur ordre et leur découpage artificiel n'ont aucune autorité.

Poser :

\[
\boxed{|\mathcal P|:=\sum_s r_s.}
\]

Une présentation est une unité de contexte causal, pas un batch d'exécution. Regrouper ou séparer deux observations non simultanées appartient à l'application hôte et peut modifier la cognition.

### 1.3 Horloge commune

Soient :

\[
0<T_{mem}<\infty,
\qquad
\eta\in[0,1].
\]

Définir :

\[
\chi=2^{-1/T_{mem}},
\qquad
\alpha=1-\chi,
\qquad
\beta=\eta\alpha,
\qquad
\lambda=1-\beta.
\]

Ainsi :

\[
\boxed{0\le\beta<1,
\qquad 0<\lambda\le1.}
\]

Après cette dérivation, les lois cognitives ne dépendent que de `β` et `λ`. `T_mem`, `η`, `χ` et `α` n'ont aucune autre autorité.

Une `LAYER` n'avance son horloge que lorsqu'elle reçoit une présentation non vide. L'absence de présentation n'est pas une cible nulle et ne provoque aucun oubli.

Toute mémoire apprenante suit la même EMA :

\[
\boxed{X\leftarrow\lambda X+\beta X_{cible}.}
\]

À `eta=0` :

\[
\boxed{\beta=0,
\qquad\lambda=1.}
\]

L'état cognitif et structurel est alors figé. Les `CELL` existantes peuvent encore reconnaître et produire le `readout`, mais aucune mémoire ne change, aucun seed, aucune promotion et aucune nouvelle `LAYER` ne sont créés.

---

## 2. Noyau centré universel

Toute mémoire géométrique locale utilise :

\[
\boxed{H=(W,C,V)}
\]

avec `W>0`, `C∈R^D` et `V≥0`, où pour une famille pondérée de noyaux-atome :

\[
C=\frac1W\sum_s r_sc_s,
\]

\[
\boxed{
V=
\frac1W\sum_s r_s
\left(v_s+\|c_s-C\|^2\right).
}
\]

L'énergie quadratique totale du noyau par rapport à l'origine est :

\[
\boxed{Q_0(H)=W(\|C\|^2+V).}
\]

L'énergie centrale dérivée est :

\[
\boxed{E(H)=WV.}
\]

Un vecteur ponctuel `(x,r)` est exactement le noyau `(r,x,0)`.

### 2.1 Somme de noyaux

Pour `H_1=(W_1,C_1,V_1)` et `H_2=(W_2,C_2,V_2)`, poser `W=W_1+W_2`. Alors :

\[
C=C_1+\frac{W_2}{W}(C_2-C_1),
\]

\[
\boxed{
V=
\frac{W_1V_1+W_2V_2}{W}
+
\frac{W_1W_2}{W^2}\|C_1-C_2\|^2.
}
\]

Cette opération est associative et commutative en arithmétique réelle. Elle est l'unique primitive canonique de fusion de noyaux.

### 2.2 Oubli

L'oubli homothétique est :

\[
\boxed{(W,C,V)\mapsto(\lambda W,C,V).}
\]

Il ne déplace ni le centre ni la dispersion.

### 2.3 EMA d'un noyau

Pour `H=(W,C,V)` et une cible `H_t=(w,c,v)`, poser :

\[
a=\lambda W,
\qquad
b=\beta w,
\qquad
W'=a+b.
\]

Si `W'>0` :

\[
\boxed{C'=C+\frac{b}{W'}(c-C)}
\]

et :

\[
\boxed{
V'=
\frac{aV+bv}{W'}
+
\frac{ab}{W'^2}\|C-c\|^2.
}
\]

Si `b=0`, cette loi se réduit exactement à l'oubli du §2.2.

### 2.4 CONCERN sur un noyau présenté

Soit un noyau mémoire :

\[
H_a=(W_a,C_a,V_a)
\]

et un atome présenté :

\[
X=(r,c,v).
\]

La distance quadratique moyenne du contenu de `X` au centre `C_a` est :

\[
\boxed{D_a(X)=\|c-C_a\|^2+v.}
\]

Son énergie moyenne par rapport à l'origine est :

\[
\boxed{D_0(X)=\|c\|^2+v.}
\]

`H_a` est concerné par `X` si et seulement si :

\[
\boxed{
D_a(X)<D_0(X)
\quad\land\quad
D_a(X)<\|C_a\|^2+V_a.
}
\]

La première inégalité est exactement équivalente à :

\[
\boxed{\|c-C_a\|^2<\|c\|^2.}
\]

Ainsi la dispersion entrante ne crée aucune direction ni aucun gain ; elle intervient seulement dans l'admissibilité géométrique complète.

Toutes les inégalités sont strictes. Une égalité exacte n'accorde aucune autorité.

En particulier, un noyau mémoire de centre `C_a=0` ne concerne aucun atome.

Pour une population finie `\mathcal H`, poser :

\[
\boxed{
I_{\mathcal H}(X)=
\{a:\ H_a\text{ est concerné par }X\}.
}
\]

Pour `a∈I_{\mathcal H}(X)`, définir le gain :

\[
\boxed{
g_a(X)=D_0(X)-D_a(X)
=\|c\|^2-\|c-C_a\|^2>0,
}
\]

puis :

\[
\boxed{q_a(X)=W_ag_a(X).}
\]

Si `I_{\mathcal H}(X)\ne\varnothing`, la responsabilité est :

\[
\boxed{
\theta_a(X)=
r\frac{q_a(X)}{\sum_{b\in I_{\mathcal H}(X)}q_b(X)}.
}
\]

Sinon toutes les responsabilités sont nulles.

Lorsque `I_{\mathcal H}(X)\ne\varnothing` :

\[
\boxed{\sum_a\theta_a(X)=r.}
\]

Toute composante concernée reçoit donc une responsabilité strictement positive. Deux noyaux de géométrie exactement identique `(C,V)` ont ensemble exactement l'autorité du noyau obtenu en additionnant leurs supports.

### 2.5 Cible d'une population

Pour une présentation :

\[
\mathcal P=\{X_s=(r_s,c_s,v_s)\},
\]

chaque noyau préexistant reçoit :

\[
m_a=\sum_s\theta_a(X_s).
\]

Si `m_a>0`, poser :

\[
\boxed{
c_a=\frac{\sum_s\theta_a(X_s)c_s}{m_a}}
\]

et :

\[
\boxed{
v_a=
\frac1{m_a}
\sum_s\theta_a(X_s)
\left(v_s+\|c_s-c_a\|^2\right).
}
\]

Puis appliquer l'EMA du §2.3 avec la cible `(m_a,c_a,v_a)`.

Si `m_a=0`, appliquer seulement l'oubli du §2.2.

Cette primitive ne décide pas ce que signifie l'absence de noyau concerné. Cette décision appartient au rôle de la population qui l'emploie.

---

## 3. CELL

### 3.1 État

Une `CELL i` possède exactement :

\[
\boxed{H_i=(A_i,C_i,V_i),
\qquad A_i>0.}
\]

`C_i` est la valeur directionnelle reconnue par la `CELL`. `V_i` est la dispersion des présentations apprises autour de cette valeur. `A_i` est son support EMA courant.

À toute frontière causale :

\[
\boxed{C_i\ne0.}
\]

### 3.2 CONCERN et ALLOCATE publics

Pour chaque atome présenté :

\[
X_s=(r_s,c_s,v_s),
\]

appliquer la primitive du §2.4 à la population des `CELL` du snapshot perceptif et poser :

\[
\boxed{I_s=I_{\mathcal H_{CELL}}(X_s).}
\]

Si `I_s=\varnothing` :

\[
\boxed{\rho_{Ls}=r_s,
\qquad\rho_{is}=0.}
\]

Si `I_s\ne\varnothing` :

\[
\boxed{\rho_{Ls}=0,
\qquad\rho_{is}=\theta_i(X_s).}
\]

Ainsi :

\[
\boxed{\rho_{Ls}+\sum_i\rho_{is}=r_s.}
\]

L'absence de `CELL` concernée signifie donc « inconnu pour cette `LAYER` ». Aucune classe supplémentaire ni gagnant artificiel n'est introduit.

### 3.3 Reconnaissance

Toute `CELL i` telle que :

\[
\rho_{is}>0
\]

reconnaît, pour cet atome, sa valeur de snapshot :

\[
\boxed{C_i^-.}
\]

La reconnaissance est éphémère. Elle ne modifie pas la géométrie avant la phase d'apprentissage et participe à la fois au `readout` externe et au contexte vertical du §5.

### 3.4 Apprentissage

Après calcul de toutes les responsabilités, les `CELL` préexistantes sont mises à jour exactement une fois par la règle de population du §2.5 avec `\theta_i=\rho_i`.

Les suppressions de centres nuls et les coalescences exactes appartiennent à la normalisation du §4.4 ; elles ne modifient jamais les reconnaissances ni le contexte déjà déterminés depuis le snapshot perceptif.

### 3.5 Persistance

Une `CELL` acquise n'est pas détruite par l'absence d'alimentation. `A_i` peut décroître par oubli ; `C_i,V_i` restent sa connaissance.

À temps fini en arithmétique réelle, une masse positive soumise seulement à `A_i←λA_i` reste positive. Une réalisation numérique doit préserver cette sémantique : un sous-flux numérique ne constitue pas une destruction cognitive.

Une `CELL` ne disparaît que par contraction matérielle obligatoire définie au §7.4.

### 3.6 Valeur géométrique intrinsèque

Pour toute `CELL` persistante :

\[
\boxed{
K_i=
\frac{\|C_i\|^2}{\|C_i\|^2+V_i}.
}
\]

Ainsi :

\[
\boxed{0<K_i\le1.}
\]

`K_i` est entièrement dérivé. Il ne dépend ni de `A_i`, ni du temps, ni d'une fréquence d'utilisation. Il n'intervient dans aucune loi d'apprentissage ou d'allocation ; il mesure seulement la perte intrinsèque d'une destruction forcée de connaissance.

---

## 4. LAYER et Σ

### 4.1 État d'une LAYER

Une `LAYER` possède exactement :

- une population finie de `CELL` ;
- une mémoire privée `Σ_L` contenant les présentations encore inconnues en cours d'apprentissage.

\[
\boxed{
\Sigma_L=\{K_a\}_{a\in A},
\qquad K_a=(W_a,C_a,V_a).
}
\]

Ses noyaux utilisent exactement la même géométrie que les `CELL`. `Σ_L` n'est ni émissive ni une seconde allocation publique : elle ne lit que les atomes qu'aucune `CELL` ne concerne.

### 4.2 DETECT

Pour chaque atome :

\[
X_s=(r_s,c_s,v_s)
\]

ayant :

\[
I_s=\varnothing
\]

et `c_s\ne0`, appliquer la primitive du §2.4 à la population `Σ_L` du snapshot perceptif.

Si au moins un noyau de `Σ_L` est concerné :

\[
\boxed{\tau_{as}=\theta_a(X_s).}
\]

Toute la masse inconnue de cet atome est alors répartie entre les composantes concernées de `Σ_L`.

Si aucun noyau de `Σ_L` n'est concerné et `β>0`, l'atome produit une demande de seed :

\[
\boxed{K_s^{new}=(\beta r_s,c_s,v_s).}
\]

Si `β=0`, aucune demande n'est créée.

Un atome de centre `c_s=0` ne peut concerner aucun noyau par la première inégalité du §2.4. Il fait avancer l'horloge de la `LAYER` puisqu'il appartient à une présentation reçue, mais n'alimente ni `CELL`, ni `Σ_L`, et ne crée aucun contexte vertical.

Après calcul de toutes les responsabilités privées, les noyaux préexistants de `Σ_L` sont mis à jour exactement une fois par la règle du §2.5 avec `\theta_a=\tau_a`.

Un noyau qui n'a reçu aucune responsabilité subit donc uniquement l'oubli. Les seeds restent hors de l'état persistant jusqu'à la transaction matérielle du §7.3.

### 4.3 Récurrence

Après mise à jour, et seulement si `β>0`, un noyau **préexistant** de `Σ_L` est mûr si et seulement si :

\[
\boxed{W_a>\beta
\qquad\land\qquad
C_a\ne0.}
\]

Un seed issu d'une seule présentation vérifie :

\[
W=\beta r\le\beta
\]

et ne peut donc pas être mûr au même pas. Sans nouvelle alimentation, `W_a` est seulement multiplié par `λ≤1` ; le temps seul ne peut jamais créer une `CELL`.

Une composante mûre devient une `CELL` portant exactement le même noyau :

\[
\boxed{H_{new}=K_a.}
\]

La promotion ne crée aucun payload cognitif supplémentaire et n'a aucun coût matériel marginal.

### 4.4 Normalisation de frontière

Après les mises à jour de `CELL` et de `Σ_L`, une `LAYER` est ramenée à une forme canonique unique avant toute croissance matérielle :

1. supprimer tout noyau de centre `C=0` ;
2. coalescer, séparément dans les `CELL` et dans `Σ_L`, les noyaux de géométrie exactement identique `(C,V)` par somme de support ;
3. promouvoir simultanément toutes les composantes mûres issues exclusivement des noyaux de `Σ_L` préexistants au snapshot ;
4. coalescer à nouveau les `CELL` de géométrie exactement identique ;
5. supprimer de `Σ_L` tout noyau qui, considéré comme présentation `(1,C_a,V_a)`, est concerné par au moins une `CELL` courante ;
6. annuler toute demande de seed qui, considérée comme présentation `(1,C_s,V_s)`, est concernée par au moins une `CELL` courante.

La forme normalisée vérifie donc : aucun centre nul, aucun clone exact dans une même population, aucune composante privée déjà couverte par une `CELL`.

Les seeds survivants restent des demandes de croissance et ne deviennent persistants qu'au §7.3. Toute `CELL`, tout seed ou toute `LAYER` créé pendant le pas n'acquiert d'autorité perceptive qu'à la présentation suivante.

---

## 5. Contexte reconnu, readout et récursion

### 5.1 Valeurs reconnues d'un atome

Pour chaque atome présenté `X_s`, définir l'ensemble exact des valeurs reconnues :

\[
\boxed{
R_s=\{C_i^-:\ i\in I_s\}/=,
}
\]

où `/=` quotient les centres vectoriellement exactement identiques.

Poser :

\[
\boxed{n_s=|R_s|.}
\]

Les identités administratives, les supports `A_i`, les dispersions `V_i` des `CELL` et les responsabilités `\rho_{is}` n'appartiennent pas à la géométrie du contexte une fois `R_s` déterminé.

Si `n_s=0`, l'atome ne contribue pas au contexte reconnu.

Si `n_s>0`, chaque valeur `c\in R_s` contribue au contexte par le noyau ponctuel :

\[
\boxed{
\left(\frac{r_s}{n_s},c,0\right).
}
\]

Cette répartition est uniforme entre les valeurs reconnues distinctes d'un même atome. Elle conserve la masse de l'atome sans introduire d'autorité d'identité, de support ou d'ordre.

### 5.2 Noyau de contexte d'une LAYER

Pour toute la présentation reçue par une `LAYER`, fusionner par la loi du §2.1 toutes les contributions du §5.1 :

\[
\boxed{
H_L^{\uparrow}
=
\bigoplus_{s:n_s>0}
\bigoplus_{c\in R_s}
\left(\frac{r_s}{n_s},c,0\right).
}
\]

S'il n'existe aucune reconnaissance, `H_L^{\uparrow}` est absent.

Sinon écrire :

\[
\boxed{
H_L^{\uparrow}
=
(W_L^{\uparrow},C_L^{\uparrow},V_L^{\uparrow}).
}
\]

Sa masse vérifie exactement :

\[
\boxed{
W_L^{\uparrow}
=
\sum_{s:n_s>0}r_s
\le|\mathcal P|.
}
\]

`C_L^{\uparrow}` est le barycentre des connaissances effectivement reconnues pendant la présentation, avec conservation de la masse causale des atomes.

`V_L^{\uparrow}` mesure la dispersion **entre les valeurs reconnues**. Cette dispersion appartient au contexte reconnu lui-même ; elle n'est ni une erreur d'explication, ni une mémoire d'autorité des `CELL`.

Le noyau de contexte dépend donc seulement :

- des masses de la présentation ;
- des ensembles exacts de valeurs reconnues par ses atomes.

Conditionnellement à ces ensembles, il est indépendant des supports EMA des `CELL`, de leurs responsabilités d'apprentissage, de leurs identités et de leur ordre.

### 5.3 Autorité verticale

Une `LAYER` possède un contexte vertical émissible si et seulement si :

\[
\boxed{
H_L^{\uparrow}\text{ existe}
\quad\land\quad
V_L^{\uparrow}>0
\quad\land\quad
C_L^{\uparrow}\ne0.
}
\]

- `V_L^{\uparrow}=0` signifie que toute la reconnaissance de la présentation se réduit à une seule valeur vectorielle distincte ; aucune relation entre connaissances distinctes n'est donc formée ;
- `C_L^{\uparrow}=0` signifie que le contexte ne possède aucune direction vectorielle canonique ; il reste silencieux.

Aucune direction arbitraire n'est construite pour sauver un contexte exactement centré en zéro.

Lorsqu'il est émissible, le contexte de la `LAYER` suivante est exactement la présentation singleton :

\[
\boxed{
\operatorname{input}(L_{k+1})
=
\{H_{L_k}^{\uparrow}\}.
}
\]

Il n'existe donc jamais d'arbre de branches inter-couches : une `LAYER` émet au plus un noyau de contexte par présentation.

### 5.4 Limite de résolution contextuelle

Le contrat vertical conserve exactement le quotient `(W,C,V)` du contexte reconnu. Deux configurations de reconnaissances distinctes produisant exactement le même noyau contextuel sont indiscernables pour les couches supérieures.

Cette perte est native au type cognitif d'Auxein : aucune covariance, orientation de second ordre ou identité de constituant n'est transmise.

En particulier, une relation parfaitement symétrique de centre nul, telle qu'un contexte constitué de `+a` et `-a` à masses égales, n'a aucun représentant vectoriel non nul compatible avec l'invariance orthogonale. Elle reste silencieuse.

### 5.5 Readout externe du NETWORK

Chaque instance reçoit une étiquette d'univers :

\[
\boxed{u_N\in\mathrm{String}^+.}
\]

`u_N` est une chaîne non vide, égale à `"auxein"` par défaut. Elle identifie le contexte sémantique extérieur de l'instance et n'intervient dans aucune décision cognitive interne.

Pour tout triplet `(k,s,i)` tel que la `CELL i` de `L_k` reçoit une responsabilité positive sur le noyau présenté :

\[
X_{ks}=(r_{ks},c_{ks},v_{ks}),
\]

produire la reconnaissance éphémère :

\[
\boxed{R_{ksi}=(u_N,c_{ks},C_{ki}^-).}
\]

Sa représentation externe canonique reste le triplet ordonné JSON-compatible :

```text
[universe, local_input, recognised]
```

La dispersion interne `v_{ks}` ne fait pas partie de l'identité externe d'une reconnaissance. Elle participe à l'admissibilité interne, pas à la valeur vectorielle reconnue.

Le `readout` du `NETWORK` est l'ensemble exact des reconnaissances produites sur toutes les `LAYER` effectivement parcourues :

\[
\boxed{
\operatorname{readout}_N
=
\{(u_N,c_{ks},C_{ki}^-):\rho_{kis}>0\}.
}
\]

Deux occurrences de triplets exactement identiques constituent la même reconnaissance et sont coalescées sans multiplicité.

Le `readout` ne contient ni indice de `LAYER`, ni identité de `CELL`, ni masse, ni responsabilité, ni provenance. Il est dérivé, éphémère, n'est jamais relu par Auxein et n'appartient pas à l'état persistant.

### 5.6 Récursion du NETWORK

Le `NETWORK` est une suite ordonnée :

```text
L0 → L1 → L2 → ...
```

`L0` reçoit la présentation extérieure uniformisée du §1.1.

Pour chaque `LAYER` suivante qui existait déjà au début du pas, elle reçoit l'unique noyau de contexte émissible produit par la couche précédente. Si aucun contexte émissible n'est produit, aucune couche supérieure n'est parcourue pour cette branche causale ; il n'existe qu'une branche.

Une `LAYER` sans `CELL` ne produit aucun contexte vertical. Elle apprend uniquement les noyaux reçus dans `Σ_L`.

### 5.7 Croissance verticale

Si une `LAYER` terminale produit un contexte émissible et qu'aucune `LAYER` suivante n'existe, la géométrie demande la création d'une nouvelle `LAYER` vide, seulement si `β>0`.

Cette création appartient à la transaction globale du §7.3. Si elle est refusée, l'état cognitif existant reste inchangé. Le contexte courant n'est rejoué ni mémorisé hors de toute `LAYER`.

Une `LAYER` créée pendant le pas ne lit pas le contexte qui a provoqué sa création. Une nouvelle profondeur exige donc au moins une nouvelle occurrence future du contexte.

---

## 6. Causalité d'une présentation

À toute `LAYER` effectivement parcourue sont associés trois états conceptuels :

\[
\boxed{
L^-\xrightarrow{\text{perception unique}}L^*
\xrightarrow{\text{normalisation}}L^+.
}
\]

`L^-` est le snapshot persistant au moment où la `LAYER` reçoit sa présentation.

Tous les `CONCERN`, `ALLOCATE`, reconnaissances, ensembles `R_s`, noyau de contexte, cibles EMA et décisions privées de `Σ_L` du pas sont calculés exclusivement depuis `L^-` et la présentation courante.

`L^*` contient les noyaux préexistants après leur unique mise à jour. `L^+` est la forme canonique du §4.4.

Aucun objet absent de `L^-` ne peut lire, concerner, apprendre, être reconnu ou participer au contexte vertical pendant cette présentation. Aucune transformation de `L^*` ou `L^+` ne provoque de replay.

Pour chaque présentation extérieure :

1. restaurer d'abord la solvabilité matérielle si nécessaire (§7.4) ;
2. figer la suite des `LAYER` existantes pour ce pas et initialiser le `readout` éphémère ;
3. construire la présentation uniforme du §1.1 et la remettre à `L0` ;
4. pour chaque `LAYER` existante recevant une présentation non vide, dans l'ordre du réseau :
   1. coalescer les atomes de géométrie exactement identique `(c,v)` ;
   2. figer `L^-` ;
   3. appliquer `CONCERN/ALLOCATE` aux `CELL` de `L^-` ;
   4. produire les reconnaissances du `readout` et le noyau de contexte `H_L^{\uparrow}` depuis ces mêmes `CELL` ;
   5. si le contexte est émissible et que la `LAYER` suivante existait au début du pas, lui transmettre immédiatement la présentation singleton `{H_L^{\uparrow}}` ;
   6. mettre à jour exactement une fois les `CELL` préexistantes ;
   7. appliquer `DETECT` aux seuls atomes inconnus depuis le `Σ_L` de `L^-`, puis mettre à jour exactement une fois ses composantes préexistantes ;
   8. normaliser la `LAYER` selon le §4.4 ;
   9. si la `LAYER` est terminale, que son contexte était émissible, qu'aucun successeur n'existait et que `β>0`, former une demande de nouvelle `LAYER` ;
5. réunir tous les seeds survivants et l'éventuelle demande de `LAYER` en une transaction globale de croissance (§7.3) ;
6. exécuter cette transaction entière si elle est payable, sinon ne rien créer ;
7. terminer la matérialisation éventuelle du `readout`, puis retourner le `readout` et l'état post-pas.

Une reconnaissance peut être livrée dès sa production si le triplet exact n'a pas déjà été livré pendant le pas ; cet ordre de livraison n'a aucune autorité.

---

## 7. Économie matérielle

### 7.1 Principe

Le budget est un plafond entier exact :

\[
\boxed{B_{units}\in\mathbb N.}
\]

L'empreinte persistante de l'état est :

\[
\boxed{M_{units}(\mathcal A)\in\mathbb N.}
\]

Un état est solvable si et seulement si :

\[
\boxed{M_{units}(\mathcal A)\le B_{units}.}
\]

Le budget n'est pas une monnaie accumulée. Une destruction cesse d'occuper de la capacité ; elle ne crée aucun crédit.

L'économie ne modifie aucune loi géométrique.

### 7.2 Packing canonique

Soit :

- `p=4` pour un réel persistant `f32` ;
- `p=8` pour un réel persistant `f64` ;
- un `u64` coûte `8` unités ;
- un tag discret coûte `1` unité.

Le noyau `(W,C,V)` occupe :

\[
\boxed{U_H=(D+2)p.}
\]

Une composante de `Σ_L` et une `CELL` possèdent exactement ce payload :

\[
\boxed{U_C=U_H.}
\]

Le header logique du `NETWORK` contient :

- `format_version=2`, `dimension`, `steps_seen`, `layer_count` sur `u64` ;
- un tag `scalar` ;
- `memory`, `eta` dans le format persistant.

Donc :

\[
\boxed{U_N=33+2p.}
\]

Chaque `LAYER` possède deux compteurs `u64`. Aucun slot cognitif inutilisé n'est réservé.

Pour `N_C(L)` `CELL` dans `L` :

\[
\boxed{
M_{units}(\mathcal A)
=
U_N+
\sum_L
\left[16+(|\Sigma_L|+N_C(L))U_H\right].
}
\]

Le contexte vertical est éphémère et ne possède aucun coût persistant propre.

La promotion `Σ_L→CELL` conserve exactement le même payload et remplace un slot privé par un slot de connaissance. Son coût marginal est donc :

\[
\boxed{c_{promote}=0.}
\]

Chaque nouveau noyau effectivement ajouté à `Σ_L` coûte :

\[
\boxed{c_{seed}=U_H.}
\]

Une nouvelle `LAYER` vide coûte seulement son header :

\[
\boxed{c_{layer}=16.}
\]

L'état minimal exécutable est `NETWORK + L0` vide :

\[
\boxed{M_{min}=U_N+16.}
\]

Si `B_{units}<M_{min}`, l'environnement est inexécutable.

Une interface peut exprimer ergonomiquement le budget en unités `U_H`, mais toute décision interne utilise exclusivement `B_{units}`.

### 7.3 Croissance

Les promotions du §4.4 sont géométriques et matériellement neutres ; elles sont appliquées avant toute création matérielle.

Après lecture de toutes les `LAYER`, réunir :

- toutes les demandes de nouveaux noyaux `Σ_L` encore admissibles du §4.4 ;
- l'éventuelle nouvelle `LAYER` de frontière requise par le §5.7.

Cet ensemble forme l'unique transaction de croissance matérielle `G_t`.

À `β=0`, `G_t` est vide.

`\mathcal A\oplus G_t` désigne l'état obtenu en appliquant simultanément toutes les créations de ce lot, avec coalescence exacte des noyaux privés identiques.

La transaction est exécutée si et seulement si :

\[
\boxed{M_{units}(\mathcal A\oplus G_t)\le B_{units}.}
\]

Sinon aucune création de `G_t` n'a lieu.

L'économie ne sélectionne jamais un sous-ensemble de demandes géométriquement simultanées.

### 7.4 Solvabilité forcée

Une baisse de budget peut rendre l'état courant insolvable. La contraction a lieu avant toute nouvelle perception.

Si :

\[
M_{units}(\mathcal A)>B_{units},
\]

alors :

1. vider simultanément toutes les mémoires `Σ_L` ;
2. supprimer toute `LAYER` terminale sans `CELL`, sans jamais supprimer `L0` ;
3. si l'état reste insolvable, considérer les valeurs distinctes `K_i` des `CELL` restantes et, pour chaque valeur `k`, l'état `\mathcal A_{>k}` obtenu en conservant exactement les `CELL` telles que `K_i>k`, puis en supprimant les `LAYER` terminales devenues vides ;
4. s'il existe un `k` tel que `M_{units}(\mathcal A_{>k})\le B_{units}`, choisir le plus petit et remplacer l'état par `\mathcal A_{>k}` ;
5. sinon, si l'état minimal `NETWORK + L0` vide est solvable, supprimer toutes les `CELL` et ramener le réseau à cet état minimal ; sinon l'environnement est inexécutable.

Cette forme détruit exactement des classes entières de même valeur :

\[
\boxed{K_i=K_j\Longrightarrow(i\text{ survit}\iff j\text{ survit}).}
\]

`A_i`, l'âge et l'absence du flux ne participent jamais à la décision. Une destruction ne réinjecte aucun passé dans `Σ_L`.

### 7.5 Absence de remplacement volontaire

Un état solvable ne détruit aucune `CELL` afin d'en financer une autre. Si la transaction de croissance n'est pas payable, elle attend une frontière future.

`K_i` n'est consulté que lorsqu'une perte de connaissance est déjà matériellement obligatoire.

### 7.6 Mutation des paramètres

Le budget peut changer entre deux présentations. Une hausse ne crée rien immédiatement. Une baisse est résolue par le §7.4 à la frontière suivante.

Une modification de `eta` est atomique et redéfinit seulement `β` et `λ` à la frontière suivante. Elle ne crée, ne fusionne ni ne détruit aucun noyau au moment de la mutation.

### 7.7 Invariant et terminaison

À toute frontière solvable :

\[
\boxed{M_{units}(\mathcal A)\le B_{units}.}
\]

La contraction forcée termine car elle opère sur des populations finies, puis choisit au plus un cutoff dans l'ensemble fini des valeurs `K_i`.

Après perception, les EMA, promotions, suppressions de travail couvert et coalescences n'augmentent pas l'empreinte ; la seule croissance persistante est `G_t`, soumise à un unique test de payabilité.

Toute transition finie termine donc sur un état solvable ou sur le verdict « environnement inexécutable ».

---

## 8. Invariances, dégénérescences et exigences numériques

Toute réalisation conforme préserve :

1. permutation des atomes d'une présentation ;
2. coalescence ou découpage d'atomes de géométrie exactement identique ;
3. rotation orthogonale de l'espace vectoriel ;
4. changement d'échelle uniforme avec `C→aC` et `V→a²V` ;
5. zero-padding exact ;
6. renommage bijectif des éventuelles poignées administratives ;
7. conservation de la masse des responsabilités ;
8. conservation de la masse contextuelle `W_L^{\uparrow}=Σ_{s:n_s>0}r_s` ;
9. indépendance cognitive des `LAYER` hors présentation `(r,c,v)` ;
10. absence de replay ;
11. absence de subdivision causale d'une présentation par une optimisation d'exécution ;
12. absence d'autorité des supports `A_i` et des responsabilités `ρ_i` dans la géométrie du contexte vertical ;
13. unicité du noyau contextuel émis par couche et par présentation ;
14. silence vertical d'une reconnaissance réduite à une seule valeur distincte ;
15. silence vertical d'un contexte exactement centré en zéro.

L'origine `0` est sémantique ; une translation uniforme n'est donc pas une invariance exigée.

Une égalité géométrique exacte ne peut être résolue par un ID, une adresse mémoire, un ordre de conteneur ou un axe arbitraire.

Une masse nulle n'apprend rien. Une `CELL` avec `C_i=0` ne concerne aucun atome. Un contexte avec `C^{\uparrow}=0` n'est pas émis.

Deux contextes distincts ayant exactement le même quotient `(W,C,V)` sont cognitivement indistinguables pour la couche suivante. Auxein n'invente aucune structure supplémentaire pour les séparer.

### 8.1 Calcul numérique

Les valeurs persistantes utilisent `f32` ou `f64`. Les calculs intermédiaires doivent être réalisés au moins en `binary64` avant projection atomique dans le format persistant.

Les réductions dont l'ordre n'a aucune autorité doivent être reproductibles et indépendantes de l'ordre d'itération.

Les variances et fusions utilisent les formes positives du §2 ; aucune soustraction de grands moments presque égaux n'est nécessaire à la loi canonique.

Aucune valeur seulement petite ne peut être remplacée par zéro au moyen d'un epsilon comportemental. Les zéros structurellement démontrés peuvent être construits exactement.

Le test `V_L^{\uparrow}=0` signifie que toutes les valeurs reconnues distinctes fusionnées ont exactement la même position après quotient ; aucune tolérance ne crée ni ne détruit une relation verticale.

Une implémentation doit empêcher qu'un support positif soit interprété comme une destruction cognitive uniquement à cause d'un sous-flux numérique lors de l'oubli. Une renormalisation commune ou toute représentation mathématiquement équivalente est admissible si elle conserve exactement les décisions canoniques.

### 8.2 Frontière d'implémentation

Caches, index, décroissances différées, queues, parallélisme, chunking, mémoïsation et structures de travail sont autorisés s'ils sont entièrement reconstructibles depuis l'état canonique et n'altèrent aucune décision.

Un index géométrique peut réduire les candidats aux concernements publics et privés, mais seuls les prédicats du §2.4 possèdent l'autorité.

La construction du contexte peut être incrémentale, mais elle doit être exactement équivalente à la fusion commutative des contributions du §5.1.

Une présentation reste un événement causal unique quelle que soit sa réalisation physique.

---

## 9. État persistant canonique

L'état persistant est minimal.

### 9.1 NETWORK

- ordre des `LAYER` ;
- `format_version=2` administratif ;
- `dimension` ;
- `scalar∈{f32,f64}` ;
- `memory` ;
- `eta` ;
- compteur de présentations achevées.

Le budget appartient à l'environnement matériel et n'est pas une connaissance apprise.

L'étiquette d'univers du `readout` appartient à l'interface et n'est pas une mémoire cognitive.

### 9.2 LAYER

- `Σ_L`, population finie de noyaux `(W,C,V)` ;
- population de `CELL`.

À toute frontière causale, cet état est sous la forme normalisée du §4.4 : aucun centre nul, aucun clone exact dans une même population et aucun noyau privé déjà couvert par une `CELL`.

Index, horloges d'exécution paresseuses et tables de travail sont dérivés et ne possèdent aucune autorité cognitive.

### 9.3 CELL

- `H_i=(A_i,C_i,V_i)`.

Aucune autre mémoire cognitive n'est requise.

### 9.4 Éléments non persistants

Ne sont notamment pas persistés :

- présentations courantes ;
- responsabilités ;
- ensembles `R_s` ;
- noyaux de contexte `H_L^{\uparrow}` ;
- `readout` ;
- demandes de croissance non encore commises ;
- caches et index d'exécution.

---

## 10. Fermeture

Pour un état canonique fini `\mathcal A_t`, une présentation extérieure finie `\mathcal X_t`, une configuration causale valide et un environnement matériel exécutable, les sections précédentes définissent une transition finie et un contexte reconnu éphémère :

\[
\boxed{
(\mathcal A_t,\mathcal X_t;u_N)
\longmapsto
(\mathcal A_{t+1},\operatorname{readout}_{N,t}).
}
\]

Le noyau cognitif d'une `LAYER` est :

```text
présentation de noyaux
→ CELL concernées
→ reconnaissances
→ contexte reconnu unique
→ LAYER suivante
```

Ce qu'aucune `CELL` ne reconnaît suit :

```text
inconnu
→ Σ_L
→ récurrence
→ CELL locale
```

La croissance horizontale et la croissance verticale sont donc strictement distinctes :

\[
\boxed{
\text{inconnu récurrent}
\longrightarrow
\text{nouvelle connaissance dans la même LAYER},
}
\]

\[
\boxed{
\text{connaissances distinctes reconnues dans une même présentation}
\longrightarrow
\text{contexte de la LAYER suivante}.
}
\]

Les quatre primitives cognitives nommées sont :

```text
CONCERN
ALLOCATE
DETECT
CONTEXT
```

`CONCERN` et `ALLOCATE` utilisent l'unique primitive de population du §2.4. `DETECT` applique cette même géométrie privément à `Σ_L`. `CONTEXT` fusionne les valeurs reconnues, sans responsabilité d'apprentissage ni identité, par la même loi de noyau du §2.1.

Une abstraction supérieure n'est ni un constituant oublié ni une branche issue d'une interprétation particulière. Elle est une régularité récurrente du **contexte compact des connaissances déjà reconnues** par l'étage précédent.

La géométrie détermine les connaissances présentes, les concernements et les créations admissibles. L'économie maintient un état fini sans sélectionner arbitrairement entre créations simultanées. Une connaissance acquise persiste indépendamment de son actualité et ne peut être perdue que lorsqu'une contraction matérielle est obligatoire, selon sa valeur géométrique intrinsèque.
