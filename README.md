# Outil de couverture d'un portefeuille d'options by Pierre Casagranda
*Ce projet a été réalisé avec l'aide de Claude Opus 5.5

**Version en ligne : [outil-couverture-options-pc.streamlit.app](https://outil-couverture-options-pc.streamlit.app/)**, utilisable depuis un navigateur, sans rien installer.

Pour un portefeuille d'options sur un même sous-jacent, l'outil indique immédiatement combien d'actions détenir pour le couvrir en delta, comment cette couverture évolue quand le marché bouge, où elle est la plus instable, et ce que coûte une couverture réajustée seulement de temps en temps. Projet de Master 2 MAS IREF, Université de Bordeaux, écrit en Python avec Streamlit et Plotly.

<!-- Pour ajouter une capture : déposer l'image dans le dépôt sous le nom apercu.png, puis retirer les marques de commentaire de la ligne suivante. -->
<!-- ![Aperçu de l'outil](apercu.png) -->

## Ce que fait l'outil

- **Couverture en delta** du portefeuille, avec l'ordre écrit en toutes lettres : « vendre 57 actions ».
- **Scénarios de marché** : trois curseurs font varier le cours, la volatilité et le temps écoulé ; chiffres et graphiques se mettent à jour immédiatement. Un bandeau rappelle en permanence si l'on regarde la situation saisie ou un scénario modifié, et un bouton ramène les curseurs au repos.
- **Ajustement** à faire pour rester couvert, affiché en texte et par une flèche sur le graphique de couverture.
- **Zones d'instabilité** : le graphique de gamma colore la plage de cours où la couverture se dégrade le plus vite.
- **Grecques complètes** : delta, gamma, vega et theta, par option et pour le portefeuille.
- **Dividendes** : un rendement de dividende continu (modèle de Black-Scholes-Merton), nul par défaut pour retrouver exactement Black-Scholes.
- **P&L du scénario**, avec et sans couverture, qui montre qu'une position couverte ne réagit plus qu'au gamma et au temps.
- **Couverture delta-gamma** : avec une option de couverture au choix, l'outil calcule combien en traiter pour annuler le gamma, puis combien d'actions pour annuler le delta restant, et compare le P&L des deux stratégies.
- **Simulation Monte Carlo** de la couverture discrète : 2 000 trajectoires, trois fréquences de réajustement comparées, volatilité réalisée et coûts de transaction réglables, avec la fréquence la plus efficace désignée.
- **Portefeuilles d'exemple** chargés en un clic : straddle, strangle, call spread, vente de puts, position neutralisée.

## Lancer l'outil sur son ordinateur

```
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS et Linux
pip install -r requirements.txt
python -m streamlit run app.py
```

L'application s'ouvre dans le navigateur à l'adresse `http://localhost:8501`. Sous Windows, un double-clic sur `lancer_app.bat` fait la même chose une fois l'installation faite.

On utilise `python -m streamlit` plutôt que `streamlit` : certains Windows bloquent le petit lanceur `streamlit.exe` créé par pip, alors que `python.exe` est toujours autorisé. Python 3.14 requis.

## Lancer les tests

```
python -m pytest -v
```

136 tests couvrent le pricing et les quatre grecques, avec et sans dividende (contre la parité call-put et des dérivées numériques), les signes des quatre positions, l'agrégation du portefeuille, les scénarios, le P&L, la couverture delta-gamma, la simulation Monte Carlo et ses coûts de transaction, la validation des saisies, les courbes, les zones d'instabilité et la cohérence des arrondis. Ils sont relancés automatiquement par GitHub à chaque modification du dépôt.

## Structure du projet

| Fichier | Rôle |
|---|---|
| `options_risk.py` | Toute la logique financière, sans aucune dépendance à Streamlit |
| `test_options_risk.py` | Suite de tests pytest, organisée dans le même ordre que le module |
| `app.py` | Interface Streamlit, une fonction par zone de l'écran, aucun calcul financier |
| `requirements.txt` | Versions des bibliothèques |
| `lancer_app.bat` | Lanceur Windows à double-cliquer |
| `.github/workflows/tests.yml` | Tests automatiques sur GitHub |

`options_risk.py` est découpé en dix sections : pricing, grecques, position, portefeuille, scénario, validation, courbes, couverture delta-gamma, simulation, analyse de scénario. Chaque onglet de l'interface obtient ses chiffres par un seul appel : `analyser_scenario`, `analyser_couverture_delta_gamma` et `simuler_couverture`.

## Le modèle

Le prix suit le modèle de Black-Scholes-Merton, avec un rendement de dividende q versé en continu :

- d₁ = [ln(S/K) + (r − q + σ²/2) T] / (σ√T) et d₂ = d₁ − σ√T
- call = S e^(−qT) N(d₁) − K e^(−rT) N(d₂), put = K e^(−rT) N(−d₂) − S e^(−qT) N(−d₁)
- delta du call = e^(−qT) N(d₁), delta du put = e^(−qT) [N(d₁) − 1], gamma = e^(−qT) N′(d₁) / (S σ √T)

Avec q = 0, valeur par défaut, on retrouve exactement Black-Scholes : les fonctions `bs_price` et `bs_greeks` s'appellent alors comme en séance, sans le dernier paramètre. Le dividende versé fait baisser le cours de l'action, ce qui abaisse les calls et relève les puts. Dans la simulation, la tendance des trajectoires devient r − q, et les actions de couverture touchent le dividende si elles sont détenues ou le paient au prêteur si elles sont vendues à découvert.

## Conventions

**1 option porte sur 1 action du sous-jacent.** Une position de 100 options expose donc à 100 actions.

**La quantité saisie est toujours positive.** Le sens est porté par le type de position. Chaque grandeur d'une ligne vaut signe × quantité × la même grandeur pour une option, avec un signe de +1 à l'achat et −1 à la vente.

| Grandeur | Formule |
|---|---|
| Valeur, delta, gamma, vega, theta d'une ligne | signe × quantité × grandeur unitaire |
| Grandeurs du portefeuille | somme sur les lignes |
| Couverture | − delta du portefeuille |
| Actions à détenir | couverture arrondie à l'action entière |
| Ajustement | actions à détenir dans le scénario − actions détenues |
| P&L avec couverture | P&L des options + actions détenues × variation du cours |

| Position | Delta | Couverture | Gamma | Vega | Theta |
|---|---|---|---|---|---|
| Long call | positif | vendre des actions | positif | positif | négatif |
| Short call | négatif | acheter des actions | négatif | négatif | positif |
| Long put | négatif | acheter des actions | positif | positif | en général négatif |
| Short put | positif | vendre des actions | négatif | négatif | en général positif |

**Unités.** Le gamma s'exprime en actions par euro, le vega en euros par point de volatilité, le theta en euros par jour calendaire (365 jours par an).

**Valeur d'une position vendeuse.** Elle est négative : c'est le montant à payer aujourd'hui pour déboucler la position.

**Arrondi des ordres.** On ne négocie que des actions entières. Chaque couverture est arrondie à l'entier le plus proche, les demis s'éloignant de zéro (0,5 donne 1, alors que la fonction `round` de Python donnerait 0). L'ajustement étant la différence de deux nombres déjà arrondis, les chiffres affichés s'additionnent toujours exactement.

**Temps écoulé.** Le curseur retire le même nombre de jours à toutes les options, en laissant au moins un jour de vie à la plus courte.

**Zone instable.** Plage de cours où la valeur absolue du gamma dépasse 50 % de son maximum sur la grille (constante `SEUIL_INSTABILITE`).

## La couverture delta-gamma

Une action a un delta de 1 et un gamma nul : elle neutralise le delta, jamais le gamma. Pour couvrir aussi le gamma, il faut une option, choisie dans le deuxième onglet (type, strike, maturité). Le calcul se fait en deux étapes :

1. Nombre d'options de couverture = − gamma du portefeuille / gamma d'une option de couverture, arrondi à l'entier.
2. Nombre d'actions = − (delta du portefeuille + nombre d'options × delta d'une option de couverture), arrondi à l'entier.

Pour les 100 calls de l'énoncé, couverts avec un call de strike 105 à six mois, on vend 100 calls puis 14 actions. Un mouvement du cours à 112 € ne donne plus qu'un P&L de −20 € au lieu de +175 € avec le delta seul.

L'option de couverture apporte son propre vega et son propre theta. À maturité égale, gamma et vega étant proportionnels, neutraliser l'un neutralise presque l'autre ; avec une maturité différente, un risque de volatilité subsiste. Une option sans gamma (très loin de la monnaie) est refusée, car il en faudrait un nombre absurde.

## La simulation de couverture discrète

Black-Scholes suppose une couverture réajustée en continu. En pratique, on la réajuste à intervalles réguliers, et l'écart entre deux réajustements crée un P&L aléatoire. La simulation le mesure :

1. 2 000 trajectoires de cours sont tirées selon un mouvement brownien géométrique, avec la volatilité réalisée choisie et une tendance égale au taux sans risque.
2. Sur chaque trajectoire, le portefeuille est couvert en delta exact, puis réajusté tous les jours, toutes les semaines ou tous les mois, le cash étant capitalisé au taux sans risque.
3. Chaque ordre sur actions peut payer un coût de transaction, exprimé en points de base du montant échangé, à la mise en place et à chaque réajustement.
4. Les trois fréquences sont comparées sur les mêmes trajectoires, et la graine aléatoire est fixée pour que les résultats soient reproductibles.
5. Un nuage de points relie le P&L de chaque trajectoire à son cours final, pour voir d'où vient la forme de la distribution.

Deux résultats en ressortent. Quand la volatilité réalisée égale la volatilité saisie, le P&L moyen est quasi nul et sa dispersion grandit avec l'intervalle de réajustement, à peu près comme sa racine carrée. Quand la volatilité réalisée dépasse la volatilité saisie, l'acheteur d'options couvert gagne en moyenne et le vendeur perd : c'est le pari de tout trader qui achète ou vend du gamma.

Les coûts de transaction font apparaître l'arbitrage central de la couverture. Réajuster souvent réduit la dispersion, mais multiplie les frais. L'outil retient la fréquence dont l'erreur quadratique (racine de la moyenne des P&L au carré, qui combine le coût moyen et la dispersion) est la plus faible. Sans frais, le réajustement quotidien l'emporte ; à 50 points de base, il coûte 243 € en moyenne et le réajustement hebdomadaire devient le meilleur choix.

Le nuage de points dessine un V centré sur le strike pour un acheteur : les trajectoires qui finissent loin du strike ont fait un grand mouvement, que le gamma transforme en gain, tandis que celles qui finissent près du strike ont payé le theta sans contrepartie. Avec peu de réajustements, la distribution garde ainsi une traîne vers la droite : la perte d'une période est bornée par le theta, le gain ne l'est pas.

## Choix de conception

- **Streamlit** : une interface web en Python pur, qui se recalcule à chaque mouvement de curseur sans gestion d'événements à écrire.
- **Plotly** : la lecture de la valeur exacte au survol est l'usage naturel d'un trader, et bandes, lignes et flèches s'ajoutent en une instruction.
- **Logique séparée de l'interface** : `options_risk.py` n'importe pas Streamlit, il se teste donc entièrement avec pytest.
- **Validation par liste de messages** : toutes les erreurs sont affichées d'un coup, avec le numéro de ligne concerné.
- **Volatilité commune** : Black-Scholes suppose une seule volatilité par sous-jacent.
- **Temps écoulé plutôt que maturité** : avec plusieurs maturités, c'est la seule déformation cohérente pour tout le portefeuille.
- **Simulation mise en cache** : Streamlit réexécute la page à chaque clic ; la simulation ne se relance que si ses paramètres changent.

## Hypothèses et limites

- Modèle de Black-Scholes-Merton, options européennes. Le dividende est un rendement continu : c'est une très bonne approximation pour un indice, plus grossière pour une action seule qui verse un ou deux dividendes à dates fixes. Le modèle ignore aussi l'exercice anticipé d'une option américaine juste avant un détachement.
- Une seule volatilité, constante : le smile de volatilité est ignoré.
- Le P&L du scénario néglige le portage de la couverture (intérêts du cash, dividendes des actions).
- La simulation couvre avec le delta exact ; les coûts de transaction, réglables, ne portent que sur les actions et pas sur le débouclage final.
- La couverture delta-gamma est posée dans la situation initiale puis conservée sans réajustement.
- La grille de cours s'étend de −40 % à +40 % autour du cours saisi (constante `AMPLITUDE_GRILLE`).
- Quand l'échéance approche, le gamma à la monnaie diverge : c'est le comportement du modèle.

## Repères numériques

Avec S = 100 €, K = 100 €, T = 0,5 an, r = 3 %, σ = 20 % et sans dividende, pour 100 calls achetés :

| Grandeur | Valeur |
|---|---|
| Prix unitaire du call | 6,37 € |
| Delta du portefeuille | 57,02 actions |
| Couverture | vendre 57 actions |
| Gamma | 2,777 actions par € |
| Vega | 27,77 € par point de volatilité |
| Theta | −1,94 € par jour |
| Ajustement si le cours passe à 112 € | vendre 27 actions de plus |
| Couverture delta-gamma avec un call 105 à six mois | vendre 100 calls, puis vendre 14 actions |

Avec un dividende de 3 %, le call ne vaut plus que 5,55 €, son delta tombe à 0,520 et la couverture passe à 52 actions vendues. Le dividende compense alors exactement le taux : call et put à la monnaie valent le même prix.
