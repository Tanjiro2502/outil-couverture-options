# Outil de gestion du risque d'un portefeuille d'options

Projet de M2 MAS IREF. L'outil prend en entrée un portefeuille d'options sur un
même sous-jacent (achats ou ventes de calls et de puts) et affiche
immédiatement le nombre d'actions à détenir pour le couvrir en delta, la
façon dont cette couverture évolue avec le marché, et les zones de cours où
elle est la plus instable.

## Lancer l'outil

```
streamlit run app.py
```

L'application s'ouvre dans le navigateur à l'adresse `http://localhost:8501`.
Pour l'arrêter, `Ctrl + C` dans le terminal.

## Installation

```
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS et Linux
pip install -r requirements.txt
```

Python 3.14 requis.

## Lancer les tests

```
pytest -v
```

79 tests couvrent le pricing, les grecques, les signes des quatre positions,
l'agrégation du portefeuille, les scénarios, la validation des saisies, les
courbes et la détection des zones d'instabilité.

## Utilisation

1. **Barre latérale** : paramètres communs du marché, à savoir le cours S₀,
   le taux sans risque r et la volatilité σ.
2. **Tableau du portefeuille** : une ligne par option (type de position,
   quantité, strike K, maturité T). Au chargement, il contient la position
   d'exemple de l'énoncé : un achat de 100 calls de strike 100 € à 6 mois.
   Une ligne seule reproduit exactement l'outil mono-position demandé par le
   sujet.
3. **Curseurs du scénario** : cours simulé S₁ (±40 % autour de S₀), volatilité
   simulée, et temps écoulé en jours. Tous les chiffres et graphiques se
   mettent à jour immédiatement.
4. **Résultats** : valeur, delta, gamma et couverture du portefeuille dans le
   scénario, avec sous chaque chiffre son écart par rapport à la situation
   initiale, puis le détail par option (dont le prix unitaire).
5. **Graphique de couverture** : la courbe bleue est la couverture à détenir,
   la ligne grise la couverture en place, et la flèche rouge l'ajustement à
   faire. Si la volatilité ou le temps ont changé, la courbe initiale est
   tracée en pointillés.
6. **Graphique de gamma** : la bande orange marque la plage de cours où la
   couverture est la plus instable, les lignes pointillées verticales les
   strikes.
7. **Ajustement** : phrase qui résume le passage de la situation initiale au
   scénario et l'ordre à passer.

## Structure du projet

| Fichier | Rôle |
|---|---|
| `options_risk.py` | Toute la logique financière, sans aucune dépendance à Streamlit |
| `test_options_risk.py` | Suite de tests pytest, organisée dans le même ordre que le module |
| `app.py` | Interface Streamlit, une fonction par zone de l'écran, aucun calcul financier |
| `requirements.txt` | Versions des bibliothèques |
| `README.md` | Ce fichier |

`options_risk.py` est découpé en huit sections : pricing, grecques, position,
portefeuille, scénario, validation, courbes, analyse de scénario. La dernière
contient `analyser_scenario`, l'unique point d'entrée appelé par l'interface :
elle renvoie en un seul dictionnaire tout ce que l'écran affiche.

## Conventions

**1 option porte sur 1 action du sous-jacent.** Une position de 100 options
expose donc à 100 actions.

**La quantité saisie est toujours positive.** Le sens de la position est porté
par le type de position, pas par le signe de la quantité. On pose
`signe = +1` pour un achat et `signe = −1` pour une vente.

| Grandeur | Formule |
|---|---|
| Valeur d'une ligne | signe × quantité × prix unitaire |
| Delta d'une ligne | signe × quantité × delta unitaire |
| Gamma d'une ligne | signe × quantité × gamma unitaire |
| Delta, gamma, valeur du portefeuille | somme sur les lignes |
| Couverture | − delta du portefeuille |
| Ajustement | couverture du scénario − couverture en place |

| Position | Delta | Couverture | Gamma |
|---|---|---|---|
| Long call | positif | vendre des actions | positif |
| Short call | négatif | acheter des actions | négatif |
| Long put | négatif | acheter des actions | positif |
| Short put | positif | vendre des actions | négatif |

Le signe du gamma suit le sens de la position, pas la nature call ou put.

**Valeur d'une position vendeuse.** Elle est affichée négative : c'est le
montant qu'il faudrait payer aujourd'hui pour déboucler la position.

**Couverture en place.** C'est la couverture calculée dans la situation
initiale (cours S₀, volatilité saisie, aujourd'hui). L'ajustement mesure
l'écart entre ce qu'on détient et ce qu'il faudrait détenir dans le scénario.

**Temps écoulé.** Le curseur de maturité retire le même nombre de jours à
toutes les options, comme le ferait le passage du temps. Il est borné pour
laisser au moins un jour de vie à l'option la plus courte. Convention : 365
jours par an.

**Zone instable.** Plage de cours où la valeur absolue du gamma dépasse 50 %
de son maximum sur la grille (constante `SEUIL_INSTABILITE`). Un portefeuille
à plusieurs strikes peut en avoir plusieurs.

## Hypothèses du modèle

- Modèle de Black-Scholes, options européennes, sans dividende.
- Une seule volatilité pour le sous-jacent, commune à toutes les options :
  le modèle ignore le smile de volatilité.
- Le taux sans risque peut être négatif. La volatilité, les maturités, les
  quantités, le cours et les strikes doivent être strictement positifs, sinon
  la saisie est refusée avec un message qui désigne le champ et la ligne.

## Saisie invalide

Les champs de la barre latérale reprennent leur dernière valeur si on les
vide : Streamlit empêche lui-même la saisie vide. Les cases du tableau, elles,
peuvent être vidées, et l'outil affiche alors par exemple
« Ligne 2 : valeur manquante pour le strike K. » sans planter.

## Repères numériques

Avec S = 100 €, K = 100 €, T = 0,5 an, r = 3 % et σ = 20 % :

- prix du call : 6,37 €
- delta unitaire du call : 0,5702
- gamma unitaire : 0,0278

Un achat de 100 de ces calls se couvre en vendant 57 actions. Si le cours
monte à 112 €, il faut vendre 27 actions supplémentaires.

## Limites connues

- Le modèle ignore les dividendes et le coût de financement de la couverture.
- La grille de cours s'étend de −40 % à +40 % autour du cours saisi. Au-delà,
  il faut modifier la constante `AMPLITUDE_GRILLE` dans `options_risk.py`.
- Quand l'échéance approche, le gamma à la monnaie diverge : c'est le
  comportement attendu du modèle, pas un défaut de l'outil.