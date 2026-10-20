"""Logique financière de l'outil de gestion du risque d'un portefeuille d'options.

Ce module ne contient aucun appel à Streamlit : il peut être importé, testé et
utilisé en dehors de l'interface graphique. Conventions retenues partout :
1 option porte sur 1 action du sous-jacent, et toutes les options d'un même
portefeuille partagent le même sous-jacent (même cours, même taux, même
volatilité).
"""

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Demi-largeur de la plage de cours explorée par les graphiques (le sujet impose au moins ±40 %).
AMPLITUDE_GRILLE = 0.40

# Nombre de points de la grille de cours : impair pour que le spot saisi tombe exactement au centre.
NB_POINTS_GRILLE = 201

# Convention de taille de contrat : 1 option = 1 action sous-jacente.
ACTIONS_PAR_OPTION = 1

# Les quatre positions de base proposées à l'utilisateur, dans l'ordre d'affichage.
TYPES_POSITION = ("long call", "short call", "long put", "short put")

# Nombre de décimales retenu pour les ordres sur actions : on n'achète pas une demi-action.
DECIMALES_ACTIONS = 0

# Seuil en dessous duquel un ordre est considéré comme nul, pour éviter d'afficher « acheter 0 actions ».
SEUIL_ORDRE_NUL = 0.5

# Part du gamma maximal au-delà de laquelle la couverture est jugée instable : 50 % du pic.
SEUIL_INSTABILITE = 0.5

# Convention de calendrier pour passer des jours aux années.
JOURS_PAR_AN = 365

# Maturité minimale laissée à l'option la plus courte quand on simule le passage du temps.
JOURS_RESIDUELS_MIN = 1


# ---------------------------------------------------------------------------
# Section pricing
# ---------------------------------------------------------------------------

def _en_float_si_scalaire(valeur):
    """Convertit un résultat numpy scalaire en float, laisse les tableaux inchangés."""
    # On veut des float purs pour l'affichage et des tableaux pour les courbes.
    tableau = np.asarray(valeur)
    return float(tableau) if tableau.ndim == 0 else tableau


def verifier_type_option(type_option):
    """Vérifie que le type d'option vaut « call » ou « put », sinon lève une erreur."""
    # Un type mal orthographié donnerait silencieusement un mauvais prix : on bloque tout de suite.
    if type_option not in ("call", "put"):
        raise ValueError(f"Type d'option inconnu : {type_option!r} (attendu « call » ou « put »).")


def _d1_d2(S, K, T, r, sigma):
    """Calcule les termes d1 et d2 de la formule de Black-Scholes."""
    # d1 et d2 servent au prix, au delta et au gamma : on les calcule une seule fois.
    racine_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * racine_T)
    d2 = d1 - sigma * racine_T
    return d1, d2


def bs_price(S, K, T, r, sigma, type_option):
    """Prix Black-Scholes d'une option européenne, sans dividende.

    S accepte un nombre ou un tableau numpy, ce qui permet de tracer une courbe
    entière en un seul appel.
    """
    verifier_type_option(type_option)
    d1, d2 = _d1_d2(S, K, T, r, sigma)

    # Valeur actualisée du strike, commune aux deux formules.
    strike_actualise = K * np.exp(-r * T)

    # Formule du call, puis celle du put obtenue en remplaçant N(d) par N(-d).
    if type_option == "call":
        prix = S * norm.cdf(d1) - strike_actualise * norm.cdf(d2)
    else:
        prix = strike_actualise * norm.cdf(-d2) - S * norm.cdf(-d1)

    return _en_float_si_scalaire(prix)


# ---------------------------------------------------------------------------
# Section grecques
# ---------------------------------------------------------------------------

def bs_delta(S, K, T, r, sigma, type_option):
    """Delta d'une option unitaire : entre 0 et 1 pour un call, entre -1 et 0 pour un put."""
    verifier_type_option(type_option)
    d1, _ = _d1_d2(S, K, T, r, sigma)

    # Le delta du put se déduit de celui du call par la parité : delta_put = delta_call - 1.
    delta = norm.cdf(d1) if type_option == "call" else norm.cdf(d1) - 1.0

    return _en_float_si_scalaire(delta)


def bs_gamma(S, K, T, r, sigma):
    """Gamma d'une option unitaire : identique pour un call et un put, toujours positif."""
    # Pas d'argument type_option : la dérivée seconde du prix par rapport à S est la même des deux côtés.
    d1, _ = _d1_d2(S, K, T, r, sigma)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    return _en_float_si_scalaire(gamma)


def bs_greeks(S, K, T, r, sigma, type_option):
    """Regroupe le delta et le gamma unitaires dans un dictionnaire."""
    # Fonction d'assemblage uniquement : tout le calcul est délégué à bs_delta et bs_gamma.
    return {
        "delta": bs_delta(S, K, T, r, sigma, type_option),
        "gamma": bs_gamma(S, K, T, r, sigma),
    }


# ---------------------------------------------------------------------------
# Section position (une seule ligne d'options)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Position:
    """Ligne d'options : un type de position, une quantité et les paramètres de marché.

    La quantité est toujours positive : le sens achat ou vente est porté par
    type_position, qui vaut par exemple « long call » ou « short put ».
    """

    type_position: str
    quantite: float
    S: float
    K: float
    T: float
    r: float
    sigma: float

    def __post_init__(self):
        """Refuse à la construction un type de position hors des quatre cas prévus."""
        # La dataclass est figée : une seule vérification à la création suffit.
        if self.type_position not in TYPES_POSITION:
            raise ValueError(
                f"Type de position inconnu : {self.type_position!r} "
                f"(attendu l'un de {TYPES_POSITION})."
            )

    @property
    def sens(self):
        """Renvoie « long » ou « short », premier mot du type de position."""
        return self.type_position.split()[0]

    @property
    def type_option(self):
        """Renvoie « call » ou « put », second mot du type de position."""
        return self.type_position.split()[1]


def formater_nombre(valeur, decimales=2):
    """Met un nombre au format français : espace pour les milliers, virgule pour les décimales."""
    # Python formate à l'anglaise par défaut, on permute les séparateurs après coup.
    texte = f"{valeur:,.{decimales}f}"
    return texte.replace(",", " ").replace(".", ",")


def signe_position(sens):
    """Renvoie +1 pour un achat (long) et -1 pour une vente (short)."""
    # Ce signe unique propage le sens de la position au prix, au delta et au gamma.
    if sens not in ("long", "short"):
        raise ValueError(f"Sens inconnu : {sens!r} (attendu « long » ou « short »).")
    return 1 if sens == "long" else -1


def prix_unitaire(position):
    """Prix Black-Scholes d'une seule option de la position."""
    # Le prix unitaire est toujours positif : il ne dépend pas du sens de la position.
    return bs_price(
        position.S, position.K, position.T, position.r, position.sigma, position.type_option
    )


def valeur_position(position):
    """Valeur de marché de la position : négative pour une vente, positive pour un achat."""
    # Pour une vente, cette valeur est ce qu'il faudrait payer aujourd'hui pour déboucler.
    return signe_position(position.sens) * position.quantite * prix_unitaire(position)


def delta_position(position):
    """Delta total de la position, exprimé en nombre d'actions équivalentes."""
    # Signe × quantité × delta unitaire : les quatre combinaisons de signes en découlent.
    delta_unitaire = bs_delta(
        position.S, position.K, position.T, position.r, position.sigma, position.type_option
    )
    return signe_position(position.sens) * position.quantite * ACTIONS_PAR_OPTION * delta_unitaire


def gamma_position(position):
    """Gamma total de la position : positif pour un achat, négatif pour une vente."""
    # Le gamma unitaire est toujours positif, c'est le signe de la position qui tranche.
    gamma_unitaire = bs_gamma(position.S, position.K, position.T, position.r, position.sigma)
    return signe_position(position.sens) * position.quantite * ACTIONS_PAR_OPTION * gamma_unitaire


def couverture_actions(delta_de_la_position):
    """Nombre d'actions à détenir pour neutraliser le delta : l'opposé du delta."""
    # Détenir -delta actions annule la sensibilité au premier ordre de l'ensemble.
    return -delta_de_la_position


def est_ordre_nul(nombre_actions):
    """Indique si un nombre d'actions est trop petit pour donner lieu à un ordre."""
    # Source unique du seuil, utilisée par le texte des ordres et par l'interface.
    return abs(nombre_actions) < SEUIL_ORDRE_NUL


def texte_ordre_actions(nombre_actions):
    """Traduit un nombre d'actions signé en ordre lisible : « vendre 57 actions »."""
    # Le sujet demande explicitement le sens écrit en clair, pas seulement un nombre négatif.
    if est_ordre_nul(nombre_actions):
        return "aucune action à échanger"

    verbe = "acheter" if nombre_actions > 0 else "vendre"
    quantite_texte = formater_nombre(abs(nombre_actions), DECIMALES_ACTIONS)
    accord = "action" if abs(round(nombre_actions)) == 1 else "actions"
    return f"{verbe} {quantite_texte} {accord}"


def position_au_spot(position, nouveau_S):
    """Copie la position en remplaçant uniquement le cours du sous-jacent."""
    # Sert à évaluer la même position à un autre niveau de marché sans modifier l'originale.
    return replace(position, S=nouveau_S)


def ajustement_actions(position, nouveau_S):
    """Nombre d'actions à échanger pour rester couvert quand le cours passe de S0 à S1."""
    # Différence entre la couverture cible au nouveau cours et la couverture déjà en place.
    couverture_initiale = couverture_actions(delta_position(position))
    couverture_cible = couverture_actions(delta_position(position_au_spot(position, nouveau_S)))
    return couverture_cible - couverture_initiale


def resume_position(position):
    """Rassemble les chiffres d'une position seule."""
    # Fonction d'assemblage : elle n'ajoute aucun calcul, elle appelle les fonctions ci-dessus.
    delta = delta_position(position)
    couverture = couverture_actions(delta)
    return {
        "prix_unitaire": prix_unitaire(position),
        "valeur_position": valeur_position(position),
        "delta_position": delta,
        "gamma_position": gamma_position(position),
        "couverture_actions": couverture,
        "texte_couverture": texte_ordre_actions(couverture),
    }


# ---------------------------------------------------------------------------
# Section portefeuille (plusieurs lignes sur le même sous-jacent)
# ---------------------------------------------------------------------------

def construire_portefeuille(lignes, S, r, sigma):
    """Transforme les lignes saisies en tuple de Position partageant S, r et sigma.

    Chaque ligne est un dictionnaire avec les clés type_position, quantite, K et T.
    Les lignes doivent avoir été validées au préalable.
    """
    # Même sous-jacent pour toutes les lignes : cours, taux et volatilité sont communs.
    return tuple(
        Position(
            type_position=ligne["type_position"],
            quantite=float(ligne["quantite"]),
            S=float(S),
            K=float(ligne["K"]),
            T=float(ligne["T"]),
            r=float(r),
            sigma=float(sigma),
        )
        for ligne in lignes
    )


def valeur_portefeuille(portefeuille):
    """Valeur de marché du portefeuille : somme des valeurs de chaque ligne."""
    # Les valeurs sont additives : c'est la linéarité du prix par rapport aux quantités.
    return sum(valeur_position(position) for position in portefeuille)


def delta_portefeuille(portefeuille):
    """Delta agrégé du portefeuille : somme des deltas de chaque ligne."""
    # Le delta est une dérivée, et la dérivée d'une somme est la somme des dérivées.
    return sum(delta_position(position) for position in portefeuille)


def gamma_portefeuille(portefeuille):
    """Gamma agrégé du portefeuille : somme des gammas de chaque ligne."""
    # Même raisonnement que pour le delta : les grecques s'additionnent sur un même sous-jacent.
    return sum(gamma_position(position) for position in portefeuille)


def resume_portefeuille(portefeuille):
    """Rassemble les chiffres agrégés affichés dans l'encart de résultats."""
    # Fonction d'assemblage : la couverture se déduit du delta agrégé.
    delta = delta_portefeuille(portefeuille)
    couverture = couverture_actions(delta)
    return {
        "valeur": valeur_portefeuille(portefeuille),
        "delta": delta,
        "gamma": gamma_portefeuille(portefeuille),
        "couverture": couverture,
        "texte_couverture": texte_ordre_actions(couverture),
    }


def detail_lignes(portefeuille):
    """Renvoie, pour chaque ligne, ses paramètres et ses chiffres propres."""
    # Alimente le tableau de détail : on y retrouve notamment le prix unitaire demandé par le sujet.
    return [
        {
            "type_position": position.type_position,
            "quantite": position.quantite,
            "K": position.K,
            "T": position.T,
            "prix_unitaire": prix_unitaire(position),
            "valeur": valeur_position(position),
            "delta": delta_position(position),
            "gamma": gamma_position(position),
        }
        for position in portefeuille
    ]


def variation_resume(resume_initial, resume_scenario):
    """Écart entre deux résumés de portefeuille, indicateur par indicateur."""
    # Sert à afficher sous chaque chiffre de combien il a bougé depuis la situation initiale.
    return {
        cle: resume_scenario[cle] - resume_initial[cle]
        for cle in ("valeur", "delta", "gamma", "couverture")
    }


def ajustement_portefeuille(portefeuille_initial, portefeuille_scenario):
    """Nombre d'actions à échanger pour passer de la couverture en place à la couverture cible."""
    # Couverture cible dans le scénario moins couverture déjà détenue dans la situation initiale.
    couverture_en_place = couverture_actions(delta_portefeuille(portefeuille_initial))
    couverture_cible = couverture_actions(delta_portefeuille(portefeuille_scenario))
    return couverture_cible - couverture_en_place


# ---------------------------------------------------------------------------
# Section scénario (cours, volatilité et temps simulés)
# ---------------------------------------------------------------------------

def jours_vers_annees(nombre_jours):
    """Convertit un nombre de jours en fraction d'année."""
    # Le curseur parle en jours, les formules en années.
    return nombre_jours / JOURS_PAR_AN


def nombre_jours_simulables(portefeuille):
    """Nombre maximal de jours que l'on peut faire passer sans faire expirer aucune option."""
    # L'option la plus courte fixe la limite, en lui laissant au moins un jour de vie.
    maturite_minimale_en_jours = min(position.T for position in portefeuille) * JOURS_PAR_AN
    return max(0, math.floor(maturite_minimale_en_jours - JOURS_RESIDUELS_MIN))


def appliquer_scenario(position, nouveau_S, nouvelle_sigma, temps_ecoule):
    """Copie une ligne avec un nouveau cours, une nouvelle volatilité et une maturité réduite."""
    # Le temps qui passe raccourcit la maturité restante : elle doit rester strictement positive.
    maturite_restante = position.T - temps_ecoule
    if maturite_restante <= 0:
        raise ValueError("Le temps écoulé dépasse la maturité d'une option du portefeuille.")
    return replace(position, S=nouveau_S, sigma=nouvelle_sigma, T=maturite_restante)


def portefeuille_au_scenario(portefeuille, nouveau_S, nouvelle_sigma, temps_ecoule):
    """Applique le même scénario de marché à toutes les lignes du portefeuille."""
    # Même sous-jacent : le cours, la volatilité et le temps bougent ensemble pour toutes les lignes.
    return tuple(
        appliquer_scenario(position, nouveau_S, nouvelle_sigma, temps_ecoule)
        for position in portefeuille
    )


# ---------------------------------------------------------------------------
# Section validation
# ---------------------------------------------------------------------------

def _est_nombre(valeur):
    """Indique si une valeur saisie est un nombre exploitable (ni vide, ni texte, ni NaN)."""
    # Une case vide arrive sous forme de None ou de NaN depuis l'interface : elle ne doit pas planter l'outil.
    if valeur is None or isinstance(valeur, bool):
        return False
    try:
        return not np.isnan(float(valeur))
    except (TypeError, ValueError):
        return False


def pourcent_vers_decimal(valeur_pourcent):
    """Convertit un pourcentage saisi dans l'interface en taux décimal : 3 devient 0,03."""
    # L'utilisateur saisit 3 % et 20 %, les formules attendent 0,03 et 0,20.
    return float(valeur_pourcent) / 100.0


def decimal_vers_pourcent(valeur_decimale):
    """Convertit un taux décimal en pourcentage pour l'affichage : 0,2 devient 20."""
    # Opération inverse de pourcent_vers_decimal, utilisée par les textes de l'interface.
    return float(valeur_decimale) * 100.0


def _verifier_strictement_positifs(champs):
    """Renvoie un message pour chaque champ vide ou non strictement positif."""
    # Logique commune au marché et aux lignes ; la tournure « pour la/le ... » évite les accords.
    erreurs = []
    for libelle, valeur in champs.items():
        if not _est_nombre(valeur):
            erreurs.append(f"Valeur manquante pour {libelle}.")
        elif float(valeur) <= 0:
            erreurs.append(f"Valeur strictement positive attendue pour {libelle}.")
    return erreurs


def valider_marche(S, r, sigma):
    """Messages d'erreur sur les paramètres communs : cours, taux et volatilité."""
    erreurs = _verifier_strictement_positifs(
        {"le cours du sous-jacent S": S, "la volatilité σ": sigma}
    )

    # Le taux sans risque peut être négatif : on vérifie seulement que c'est un nombre.
    if not _est_nombre(r):
        erreurs.append("Valeur manquante pour le taux sans risque r.")
    return erreurs


def valider_ligne(type_position, quantite, K, T):
    """Messages d'erreur sur une ligne d'options : type, quantité, strike et maturité."""
    erreurs = []

    # Le type de position conditionne tous les signes : on le contrôle en premier.
    if type_position not in TYPES_POSITION:
        erreurs.append(
            f"Le type de position doit être choisi parmi : {', '.join(TYPES_POSITION)}."
        )

    erreurs.extend(
        _verifier_strictement_positifs(
            {"la quantité d'options": quantite, "le strike K": K, "la maturité T": T}
        )
    )
    return erreurs


def valider_saisie(type_position, quantite, S, K, T, r, sigma):
    """Renvoie la liste des messages d'erreur d'une position seule, vide si tout est correct.

    On renvoie une liste plutôt que de lever une exception : l'interface affiche
    toutes les erreurs ensemble et s'arrête proprement.
    """
    # Une position seule, c'est une ligne plus les paramètres de marché.
    return valider_ligne(type_position, quantite, K, T) + valider_marche(S, r, sigma)


def _prefixer_message(numero_ligne, message):
    """Préfixe un message d'erreur par le numéro de la ligne concernée."""
    # « Ligne 2 : valeur manquante pour le strike K. » désigne tout de suite la case fautive.
    return f"Ligne {numero_ligne} : {message[0].lower()}{message[1:]}"


def valider_portefeuille(lignes, S, r, sigma):
    """Renvoie tous les messages d'erreur du portefeuille, vide si tout est correct."""
    erreurs = valider_marche(S, r, sigma)

    # Un portefeuille vide n'a pas de couverture à calculer.
    if len(lignes) == 0:
        erreurs.append("Le portefeuille doit contenir au moins une option.")

    # Chaque ligne est contrôlée séparément, et ses messages portent son numéro.
    for numero, ligne in enumerate(lignes, start=1):
        messages = valider_ligne(
            ligne.get("type_position"), ligne.get("quantite"), ligne.get("K"), ligne.get("T")
        )
        erreurs.extend(_prefixer_message(numero, message) for message in messages)
    return erreurs


# ---------------------------------------------------------------------------
# Section courbes
# ---------------------------------------------------------------------------

def grille_spots(S0, amplitude=AMPLITUDE_GRILLE, nb_points=NB_POINTS_GRILLE):
    """Grille régulière de cours allant de S0×(1-amplitude) à S0×(1+amplitude)."""
    # Grille commune aux deux graphiques, pour que les points courants soient alignés.
    return np.linspace(S0 * (1 - amplitude), S0 * (1 + amplitude), nb_points)


def courbe_delta(position, spots):
    """Delta d'une ligne pour chaque cours de la grille."""
    # Appel vectorisé : bs_delta accepte un tableau, aucune boucle Python n'est nécessaire.
    deltas_unitaires = bs_delta(
        spots, position.K, position.T, position.r, position.sigma, position.type_option
    )
    return signe_position(position.sens) * position.quantite * ACTIONS_PAR_OPTION * deltas_unitaires


def courbe_couverture(position, spots):
    """Nombre d'actions de couverture d'une ligne pour chaque cours de la grille."""
    # La couverture est l'opposé du delta : cette courbe est le miroir de celle du delta.
    return couverture_actions(courbe_delta(position, spots))


def courbe_gamma(position, spots):
    """Gamma d'une ligne pour chaque cours de la grille."""
    # Le pic de cette courbe signale la zone où la couverture doit être réajustée le plus souvent.
    gammas_unitaires = bs_gamma(spots, position.K, position.T, position.r, position.sigma)
    return signe_position(position.sens) * position.quantite * ACTIONS_PAR_OPTION * gammas_unitaires


def courbe_delta_portefeuille(portefeuille, spots):
    """Delta agrégé du portefeuille pour chaque cours de la grille."""
    # Somme point par point des courbes de chaque ligne, en partant d'un tableau de zéros.
    return sum((courbe_delta(position, spots) for position in portefeuille), np.zeros_like(spots))


def courbe_couverture_portefeuille(portefeuille, spots):
    """Couverture agrégée du portefeuille pour chaque cours de la grille."""
    # Toujours l'opposé du delta, cette fois au niveau du portefeuille entier.
    return couverture_actions(courbe_delta_portefeuille(portefeuille, spots))


def courbe_gamma_portefeuille(portefeuille, spots):
    """Gamma agrégé du portefeuille pour chaque cours de la grille."""
    # Un long call et un short call identiques s'annulent ici exactement : gamma nul partout.
    return sum((courbe_gamma(position, spots) for position in portefeuille), np.zeros_like(spots))


def zones_instabilite(spots, gammas, seuil_relatif=SEUIL_INSTABILITE):
    """Intervalles de cours où |gamma| dépasse seuil_relatif fois son maximum sur la grille.

    Renvoie une liste de couples (début, fin). Un portefeuille à plusieurs strikes
    peut en avoir plusieurs, un portefeuille de gamma nul n'en a aucune.
    """
    amplitude = np.abs(gammas)
    maximum = amplitude.max()

    # Gamma nul partout : la couverture ne bouge jamais, il n'y a aucune zone à signaler.
    if maximum <= 0:
        return []

    # Masque booléen des points instables, puis repérage de ses fronts montants et descendants.
    dans_zone = amplitude >= seuil_relatif * maximum
    fronts = np.diff(dans_zone.astype(int))
    debuts = list(np.flatnonzero(fronts == 1) + 1)
    fins = list(np.flatnonzero(fronts == -1))

    # Une zone collée au bord de la grille n'a pas de front : on l'ouvre ou on la ferme à la main.
    if dans_zone[0]:
        debuts.insert(0, 0)
    if dans_zone[-1]:
        fins.append(len(spots) - 1)

    return [(float(spots[debut]), float(spots[fin])) for debut, fin in zip(debuts, fins)]


def texte_zones_instabilite(zones):
    """Phrase qui décrit au trader où la couverture est la plus instable."""
    # Cas particulier d'un gamma nul, par exemple un achat et une vente identiques.
    if not zones:
        return "Gamma nul sur toute la plage : la couverture ne bouge pas avec le cours."

    intervalles = " et entre ".join(
        f"{formater_nombre(debut)} € et {formater_nombre(fin)} €" for debut, fin in zones
    )
    return f"Couverture la plus instable entre {intervalles}."


def strikes_visibles(portefeuille, spots):
    """Strikes distincts du portefeuille qui tombent dans la plage du graphique."""
    # Un strike hors plage élargirait l'axe des abscisses : on ne le trace pas.
    return sorted(
        {position.K for position in portefeuille if spots[0] <= position.K <= spots[-1]}
    )


# ---------------------------------------------------------------------------
# Section analyse de scénario (tout ce que l'écran affiche)
# ---------------------------------------------------------------------------

def analyser_scenario(portefeuille_initial, nouveau_S, nouvelle_sigma, temps_ecoule):
    """Calcule en une fois les chiffres, courbes et zones affichés pour un scénario.

    Le portefeuille initial représente la situation où la couverture a été mise
    en place. Le scénario applique un nouveau cours, une nouvelle volatilité et
    un temps écoulé.
    """
    # Toutes les lignes partagent le même sous-jacent : on lit le cours et la volatilité initiaux sur la première.
    S0 = portefeuille_initial[0].S
    sigma0 = portefeuille_initial[0].sigma
    portefeuille_scenario = portefeuille_au_scenario(
        portefeuille_initial, nouveau_S, nouvelle_sigma, temps_ecoule
    )

    # Grille centrée sur le cours saisi : elle reste fixe quand le curseur se déplace.
    spots = grille_spots(S0)

    # Chiffres agrégés dans la situation initiale et dans le scénario.
    resume_initial = resume_portefeuille(portefeuille_initial)
    resume_scenario = resume_portefeuille(portefeuille_scenario)
    ajustement = ajustement_portefeuille(portefeuille_initial, portefeuille_scenario)

    # Courbes du scénario et courbes initiales, ces dernières servant de référence en pointillés.
    gamma_scenario = courbe_gamma_portefeuille(portefeuille_scenario, spots)
    zones = zones_instabilite(spots, gamma_scenario)

    return {
        "S0": S0,
        "S1": nouveau_S,
        "sigma0": sigma0,
        "sigma1": nouvelle_sigma,
        "scenario_modifie": (
            nouveau_S != S0 or nouvelle_sigma != sigma0 or temps_ecoule != 0
        ),
        "spots": spots,
        "resume_initial": resume_initial,
        "resume_scenario": resume_scenario,
        "variation": variation_resume(resume_initial, resume_scenario),
        "detail_scenario": detail_lignes(portefeuille_scenario),
        "ajustement": ajustement,
        "texte_ajustement": texte_ordre_actions(ajustement),
        "couverture_initiale_courbe": courbe_couverture_portefeuille(portefeuille_initial, spots),
        "couverture_scenario_courbe": courbe_couverture_portefeuille(portefeuille_scenario, spots),
        "gamma_initial_courbe": courbe_gamma_portefeuille(portefeuille_initial, spots),
        "gamma_scenario_courbe": gamma_scenario,
        "zones": zones,
        "texte_zones": texte_zones_instabilite(zones),
        "strikes": strikes_visibles(portefeuille_initial, spots),
    }