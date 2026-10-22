"""Logique financière de l'outil de gestion du risque d'un portefeuille d'options.

Ce module ne contient aucun appel à Streamlit : il peut être importé, testé et
utilisé en dehors de l'interface graphique. Conventions retenues partout :
1 option porte sur 1 action du sous-jacent, et toutes les options d'un même
portefeuille partagent le même sous-jacent (même cours, même taux, même
volatilité, même rendement du dividende). Le modèle est celui de
Black-Scholes-Merton ; avec un dividende nul, valeur par défaut, on retrouve
exactement Black-Scholes.
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

# Part du gamma maximal au-delà de laquelle la couverture est jugée instable : 50 % du pic.
SEUIL_INSTABILITE = 0.5

# Convention de calendrier pour passer des jours aux années.
JOURS_PAR_AN = 365

# Maturité minimale laissée à l'option la plus courte quand on simule le passage du temps.
JOURS_RESIDUELS_MIN = 1

# Nombre de trajectoires de la simulation de Monte Carlo : statistiques stables, calcul rapide.
NB_TRAJECTOIRES = 2000

# Graine du générateur aléatoire : la même simulation redonne toujours les mêmes chiffres.
GRAINE_ALEATOIRE = 42

# Intervalles de réajustement de la couverture comparés dans la simulation, en jours.
INTERVALLES_REAJUSTEMENT = (1, 7, 30)

# Gamma minimal d'une option de couverture : en dessous, il en faudrait un nombre absurde.
GAMMA_INSTRUMENT_MIN = 1e-4


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


def _d1_d2(S, K, T, r, sigma, q=0.0):
    """Calcule les termes d1 et d2 de la formule de Black-Scholes-Merton.

    q est le rendement du dividende, versé en continu ; avec q = 0, on retrouve
    exactement Black-Scholes.
    """
    # d1 et d2 servent au prix et à toutes les grecques : on les calcule une seule fois.
    # Le dividende réduit la tendance du cours : r devient r - q.
    racine_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * racine_T)
    d2 = d1 - sigma * racine_T
    return d1, d2


def bs_price(S, K, T, r, sigma, type_option, q=0.0):
    """Prix d'une option européenne (Black-Scholes-Merton, dividende continu q, nul par défaut).

    S accepte un nombre ou un tableau numpy, ce qui permet de tracer une courbe
    entière en un seul appel.
    """
    verifier_type_option(type_option)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)

    # Valeur actualisée du strike, et cours diminué des dividendes versés d'ici l'échéance.
    strike_actualise = K * np.exp(-r * T)
    cours_hors_dividendes = S * np.exp(-q * T)

    # Formule du call, puis celle du put obtenue en remplaçant N(d) par N(-d).
    if type_option == "call":
        prix = cours_hors_dividendes * norm.cdf(d1) - strike_actualise * norm.cdf(d2)
    else:
        prix = strike_actualise * norm.cdf(-d2) - cours_hors_dividendes * norm.cdf(-d1)

    return _en_float_si_scalaire(prix)


# ---------------------------------------------------------------------------
# Section grecques
# ---------------------------------------------------------------------------

def bs_delta(S, K, T, r, sigma, type_option, q=0.0):
    """Delta d'une option unitaire : entre 0 et 1 pour un call, entre -1 et 0 pour un put."""
    verifier_type_option(type_option)
    d1, _ = _d1_d2(S, K, T, r, sigma, q)

    # La parité donne delta_call - delta_put = exp(-qT), soit 1 sans dividende.
    actualisation_dividende = np.exp(-q * T)
    if type_option == "call":
        delta = actualisation_dividende * norm.cdf(d1)
    else:
        delta = actualisation_dividende * (norm.cdf(d1) - 1.0)

    return _en_float_si_scalaire(delta)


def bs_gamma(S, K, T, r, sigma, q=0.0):
    """Gamma d'une option unitaire : identique pour un call et un put, toujours positif."""
    # Pas d'argument type_option : la dérivée seconde du prix par rapport à S est la même des deux côtés.
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

    return _en_float_si_scalaire(gamma)


def bs_vega(S, K, T, r, sigma, q=0.0):
    """Vega d'une option unitaire, en euros pour une hausse de 1 point de volatilité."""
    # Identique pour un call et un put : la parité C - P = S exp(-qT) - K exp(-rT) ne dépend pas de sigma.
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100
    return _en_float_si_scalaire(vega)


def bs_theta(S, K, T, r, sigma, type_option, q=0.0):
    """Theta d'une option unitaire, en euros par jour calendaire écoulé."""
    verifier_type_option(type_option)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    strike_actualise = K * np.exp(-r * T)
    cours_hors_dividendes = S * np.exp(-q * T)

    # Terme lié à la volatilité, commun au call et au put : c'est la perte de valeur temps.
    terme_volatilite = -cours_hors_dividendes * norm.pdf(d1) * sigma / (2 * np.sqrt(T))

    # Termes de taux et de dividende, qui diffèrent entre call et put comme dans la formule du prix.
    if type_option == "call":
        theta_annuel = (
            terme_volatilite
            - r * strike_actualise * norm.cdf(d2)
            + q * cours_hors_dividendes * norm.cdf(d1)
        )
    else:
        theta_annuel = (
            terme_volatilite
            + r * strike_actualise * norm.cdf(-d2)
            - q * cours_hors_dividendes * norm.cdf(-d1)
        )

    # Convention de marché : le theta s'exprime par jour, pas par an.
    return _en_float_si_scalaire(theta_annuel / JOURS_PAR_AN)


def bs_greeks(S, K, T, r, sigma, type_option, q=0.0):
    """Regroupe le delta, le gamma, le vega et le theta unitaires dans un dictionnaire."""
    # Fonction d'assemblage uniquement : chaque grecque a sa propre fonction.
    return {
        "delta": bs_delta(S, K, T, r, sigma, type_option, q),
        "gamma": bs_gamma(S, K, T, r, sigma, q),
        "vega": bs_vega(S, K, T, r, sigma, q),
        "theta": bs_theta(S, K, T, r, sigma, type_option, q),
    }


# ---------------------------------------------------------------------------
# Section position (une seule ligne d'options)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Position:
    """Ligne d'options : un type de position, une quantité et les paramètres de marché.

    La quantité est toujours positive : le sens achat ou vente est porté par
    type_position, qui vaut par exemple « long call » ou « short put ».
    Le rendement du dividende vaut 0 par défaut, ce qui redonne Black-Scholes.
    """

    type_position: str
    quantite: float
    S: float
    K: float
    T: float
    r: float
    sigma: float
    rendement_dividende: float = 0.0

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


def facteur_position(position):
    """Multiplicateur d'une ligne : signe × quantité × nombre d'actions par option."""
    # Toutes les grandeurs d'une ligne valent ce facteur fois la grandeur d'une option unitaire.
    return signe_position(position.sens) * position.quantite * ACTIONS_PAR_OPTION


def prix_unitaire(position):
    """Prix Black-Scholes d'une seule option de la position."""
    # Le prix unitaire est toujours positif : il ne dépend pas du sens de la position.
    return bs_price(
        position.S, position.K, position.T, position.r, position.sigma, position.type_option,
        position.rendement_dividende,
    )


def valeur_position(position):
    """Valeur de marché de la position : négative pour une vente, positive pour un achat."""
    # Pour une vente, cette valeur est ce qu'il faudrait payer aujourd'hui pour déboucler.
    return facteur_position(position) * prix_unitaire(position)


def delta_position(position):
    """Delta total de la position, exprimé en nombre d'actions équivalentes."""
    # Signe × quantité × delta unitaire : les quatre combinaisons de signes en découlent.
    delta_unitaire = bs_delta(
        position.S, position.K, position.T, position.r, position.sigma, position.type_option,
        position.rendement_dividende,
    )
    return facteur_position(position) * delta_unitaire


def gamma_position(position):
    """Gamma total de la position : positif pour un achat, négatif pour une vente."""
    # Le gamma unitaire est toujours positif, c'est le signe de la position qui tranche.
    gamma_unitaire = bs_gamma(
        position.S, position.K, position.T, position.r, position.sigma, position.rendement_dividende
    )
    return facteur_position(position) * gamma_unitaire


def vega_position(position):
    """Vega total de la position : positif pour un achat, négatif pour une vente."""
    # Un acheteur d'options gagne quand la volatilité monte, un vendeur perd.
    vega_unitaire = bs_vega(
        position.S, position.K, position.T, position.r, position.sigma, position.rendement_dividende
    )
    return facteur_position(position) * vega_unitaire


def theta_position(position):
    """Theta total de la position, en euros par jour : en général négatif à l'achat."""
    # Contrepartie du gamma : l'acheteur paie chaque jour la valeur temps que le vendeur encaisse.
    theta_unitaire = bs_theta(
        position.S, position.K, position.T, position.r, position.sigma, position.type_option,
        position.rendement_dividende,
    )
    return facteur_position(position) * theta_unitaire


def couverture_actions(delta_de_la_position):
    """Nombre d'actions à détenir pour neutraliser le delta : l'opposé du delta."""
    # Détenir -delta actions annule la sensibilité au premier ordre de l'ensemble.
    return -delta_de_la_position


def arrondir_actions(nombre_actions):
    """Arrondit un nombre d'actions à l'entier le plus proche, les demis s'éloignant de zéro."""
    # round() de Python arrondit les demis vers le pair (2,5 donne 2) : on impose l'arrondi usuel (2,5 donne 3).
    return int(math.copysign(math.floor(abs(nombre_actions) + 0.5), nombre_actions))


def est_ordre_nul(nombre_actions):
    """Indique si un nombre d'actions s'arrondit à zéro et ne donne donc lieu à aucun ordre."""
    # Même règle d'arrondi que pour les ordres affichés : impossible d'afficher « acheter 0 actions ».
    return arrondir_actions(nombre_actions) == 0


def texte_ordre_actions(nombre_actions):
    """Traduit un nombre d'actions signé en ordre lisible : « vendre 57 actions »."""
    # Le sujet demande le sens écrit en clair ; on raisonne en actions entières, seules négociables.
    actions = arrondir_actions(nombre_actions)
    if actions == 0:
        return "aucune action à échanger"

    verbe = "acheter" if actions > 0 else "vendre"
    accord = "action" if abs(actions) == 1 else "actions"
    return f"{verbe} {formater_nombre(abs(actions), DECIMALES_ACTIONS)} {accord}"


# ---------------------------------------------------------------------------
# Section portefeuille (plusieurs lignes sur le même sous-jacent)
# ---------------------------------------------------------------------------

def construire_portefeuille(lignes, S, r, sigma, rendement_dividende=0.0):
    """Transforme les lignes saisies en tuple de Position partageant S, r et sigma.

    Chaque ligne est un dictionnaire avec les clés type_position, quantite, K et T.
    Les lignes doivent avoir été validées au préalable.
    """
    # Même sous-jacent pour toutes les lignes : cours, taux, volatilité et dividende sont communs.
    return tuple(
        Position(
            type_position=ligne["type_position"],
            quantite=float(ligne["quantite"]),
            S=float(S),
            K=float(ligne["K"]),
            T=float(ligne["T"]),
            r=float(r),
            sigma=float(sigma),
            rendement_dividende=float(rendement_dividende),
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


def vega_portefeuille(portefeuille):
    """Vega agrégé du portefeuille : somme des vegas de chaque ligne."""
    # Additif comme les autres grecques, puisque toutes les lignes partagent la même volatilité.
    return sum(vega_position(position) for position in portefeuille)


def theta_portefeuille(portefeuille):
    """Theta agrégé du portefeuille : somme des thetas de chaque ligne, en euros par jour."""
    return sum(theta_position(position) for position in portefeuille)


def resume_portefeuille(portefeuille):
    """Rassemble les chiffres agrégés affichés dans l'encart de résultats."""
    # La couverture exacte place le point sur la courbe ; le nombre entier d'actions sert aux ordres.
    delta = delta_portefeuille(portefeuille)
    couverture = couverture_actions(delta)
    actions_a_detenir = arrondir_actions(couverture)
    return {
        "valeur": valeur_portefeuille(portefeuille),
        "delta": delta,
        "gamma": gamma_portefeuille(portefeuille),
        "vega": vega_portefeuille(portefeuille),
        "theta": theta_portefeuille(portefeuille),
        "couverture": couverture,
        "actions_a_detenir": actions_a_detenir,
        "texte_couverture": texte_ordre_actions(actions_a_detenir),
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
            "vega": vega_position(position),
            "theta": theta_position(position),
        }
        for position in portefeuille
    ]


def variation_resume(resume_initial, resume_scenario):
    """Écart entre deux résumés de portefeuille, indicateur par indicateur."""
    # Sert à afficher sous chaque chiffre de combien il a bougé depuis la situation initiale.
    return {
        cle: resume_scenario[cle] - resume_initial[cle]
        for cle in ("valeur", "delta", "gamma", "vega", "theta", "actions_a_detenir")
    }


def ajustement_portefeuille(portefeuille_initial, portefeuille_scenario):
    """Nombre d'actions à échanger pour passer de la couverture en place à la couverture cible."""
    # Différence de deux nombres entiers d'actions : l'ordre affiché égale toujours l'écart des chiffres affichés.
    en_place = arrondir_actions(couverture_actions(delta_portefeuille(portefeuille_initial)))
    cible = arrondir_actions(couverture_actions(delta_portefeuille(portefeuille_scenario)))
    return cible - en_place


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


def portefeuille_a_date(portefeuille, temps_ecoule):
    """Le même portefeuille plus tard : maturités réduites, cours et volatilité inchangés."""
    # Sert à la simulation, où le cours vient des trajectoires et non de la position.
    return portefeuille_au_scenario(
        portefeuille, portefeuille[0].S, portefeuille[0].sigma, temps_ecoule
    )


def pnl_non_couvert(valeur_initiale, valeur_finale):
    """Gain ou perte des options seules entre deux dates de valorisation."""
    # Accepte des tableaux : on obtient d'un coup le P&L sur toute une grille de cours.
    return valeur_finale - valeur_initiale


def pnl_couvert(valeur_initiale, valeur_finale, actions_detenues, S_initial, S_final):
    """Gain ou perte des options plus des actions de couverture détenues sans réajustement.

    Le portage de la couverture est ignoré (intérêts du cash, dividendes des
    actions) : sur quelques jours, il est petit devant le gamma et le theta.
    """
    # Le gain des actions compense le delta : il ne reste que la courbure (gamma) et le temps (theta).
    return pnl_non_couvert(valeur_initiale, valeur_finale) + actions_detenues * (S_final - S_initial)


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


def points_de_base_vers_decimal(valeur_points_de_base):
    """Convertit des points de base en taux décimal : 10 points de base deviennent 0,001."""
    # Un point de base vaut 0,01 % : c'est l'unité habituelle des coûts de transaction.
    return float(valeur_points_de_base) / 10_000.0


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


def valider_marche(S, r, sigma, rendement_dividende=0.0):
    """Messages d'erreur sur les paramètres communs : cours, taux, volatilité et dividende."""
    erreurs = _verifier_strictement_positifs(
        {"le cours du sous-jacent S": S, "la volatilité σ": sigma}
    )

    # Le taux sans risque peut être négatif : on vérifie seulement que c'est un nombre.
    if not _est_nombre(r):
        erreurs.append("Valeur manquante pour le taux sans risque r.")

    # Le rendement du dividende peut être nul (cas par défaut), jamais négatif.
    if not _est_nombre(rendement_dividende):
        erreurs.append("Valeur manquante pour le rendement du dividende q.")
    elif float(rendement_dividende) < 0:
        erreurs.append("Le rendement du dividende q ne peut pas être négatif.")
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


def valider_instrument(type_option, K, T, S, r, sigma, temps_ecoule, rendement_dividende=0.0):
    """Messages d'erreur sur l'option choisie pour couvrir le gamma, vide si elle convient.

    Les paramètres de marché S, r et sigma sont supposés déjà validés.
    """
    if type_option not in ("call", "put"):
        return ["L'instrument de couverture doit être un call ou un put."]
    erreurs = _verifier_strictement_positifs(
        {"le strike de l'instrument": K, "la maturité de l'instrument": T}
    )
    if erreurs:
        return erreurs

    # L'instrument doit encore exister à la fin du scénario, avec au moins un jour de vie.
    if float(T) - temps_ecoule < JOURS_RESIDUELS_MIN / JOURS_PAR_AN:
        return ["La maturité de l'instrument doit dépasser d'au moins un jour le temps écoulé du scénario."]

    # Une option sans gamma ne peut pas en compenser : il en faudrait des quantités absurdes.
    if bs_gamma(S, float(K), float(T), r, sigma, rendement_dividende) < GAMMA_INSTRUMENT_MIN:
        return [
            "Cet instrument n'a presque pas de gamma : choisissez un strike plus proche "
            "du cours ou une maturité plus longue."
        ]
    return []


def _prefixer_message(numero_ligne, message):
    """Préfixe un message d'erreur par le numéro de la ligne concernée."""
    # « Ligne 2 : valeur manquante pour le strike K. » désigne tout de suite la case fautive.
    return f"Ligne {numero_ligne} : {message[0].lower()}{message[1:]}"


def valider_portefeuille(lignes, S, r, sigma, rendement_dividende=0.0):
    """Renvoie tous les messages d'erreur du portefeuille, vide si tout est correct.

    On renvoie une liste plutôt que de lever une exception : l'interface affiche
    toutes les erreurs ensemble et s'arrête proprement.
    """
    erreurs = valider_marche(S, r, sigma, rendement_dividende)

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
        spots, position.K, position.T, position.r, position.sigma, position.type_option,
        position.rendement_dividende,
    )
    return facteur_position(position) * deltas_unitaires


def courbe_gamma(position, spots):
    """Gamma d'une ligne pour chaque cours de la grille."""
    # Le pic de cette courbe signale la zone où la couverture doit être réajustée le plus souvent.
    gammas_unitaires = bs_gamma(
        spots, position.K, position.T, position.r, position.sigma, position.rendement_dividende
    )
    return facteur_position(position) * gammas_unitaires


def courbe_valeur(position, spots):
    """Valeur d'une ligne pour chaque cours de la grille."""
    # Même principe vectorisé que pour les grecques, appliqué au prix.
    prix = bs_price(
        spots, position.K, position.T, position.r, position.sigma, position.type_option,
        position.rendement_dividende,
    )
    return facteur_position(position) * prix


def courbe_valeur_portefeuille(portefeuille, spots):
    """Valeur du portefeuille pour chaque cours de la grille."""
    # Somme point par point des valeurs de chaque ligne.
    return sum((courbe_valeur(position, spots) for position in portefeuille), np.zeros_like(spots))


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
# Section couverture delta-gamma
# ---------------------------------------------------------------------------

def construire_instrument(type_option, K, T, S, r, sigma, rendement_dividende=0.0):
    """Option unitaire achetée, sur le même sous-jacent, servant à couvrir le gamma."""
    # Une quantité de 1 permet de lire directement son delta et son gamma unitaires.
    return Position(
        f"long {type_option}", 1.0, float(S), float(K), float(T), float(r), float(sigma),
        float(rendement_dividende),
    )


def couverture_delta_gamma(portefeuille, instrument):
    """Nombre d'options de couverture puis d'actions qui annulent le gamma, puis le delta.

    Les deux nombres sont entiers et signés : positif pour un achat, négatif pour une vente.
    """
    # Étape 1 : une action a un gamma nul, seule une option peut compenser celui du portefeuille.
    gamma_instrument = gamma_position(instrument)
    if gamma_instrument < GAMMA_INSTRUMENT_MIN:
        raise ValueError("L'instrument de couverture n'a presque pas de gamma.")
    options = arrondir_actions(-gamma_portefeuille(portefeuille) / gamma_instrument)

    # Étape 2 : ces options ajoutent leur propre delta, que l'on neutralise ensuite avec des actions.
    delta_restant = delta_portefeuille(portefeuille) + options * delta_position(instrument)
    return {"options": options, "actions": arrondir_actions(-delta_restant)}


def ligne_de_couverture(instrument, nombre_options):
    """Transforme un nombre signé d'options de couverture en ligne de portefeuille."""
    # Le signe devient le sens de la position, la quantité reste positive comme partout ailleurs.
    sens = "long" if nombre_options >= 0 else "short"
    return replace(
        instrument,
        type_position=f"{sens} {instrument.type_option}",
        quantite=float(abs(nombre_options)),
    )


def texte_ordre_options(nombre_options, instrument):
    """Traduit un nombre signé d'options en ordre lisible : « vendre 98 calls de strike 105,00 € »."""
    if nombre_options == 0:
        return "aucune option de couverture à traiter"
    verbe = "acheter" if nombre_options > 0 else "vendre"
    nom = instrument.type_option + ("s" if abs(nombre_options) > 1 else "")
    return (
        f"{verbe} {formater_nombre(abs(nombre_options), 0)} {nom} de strike "
        f"{formater_nombre(instrument.K)} € et de maturité {formater_nombre(instrument.T)} an"
    )


def analyser_couverture_delta_gamma(portefeuille_initial, instrument, nouveau_S, nouvelle_sigma, temps_ecoule):
    """Calcule la couverture delta-gamma et la compare à la couverture en delta seul dans un scénario.

    Les deux couvertures sont mises en place dans la situation initiale puis
    conservées sans réajustement, comme dans le premier onglet.
    """
    S0 = portefeuille_initial[0].S
    couverture = couverture_delta_gamma(portefeuille_initial, instrument)

    # Portefeuille complété par les options de couverture, dans la situation initiale puis dans le scénario.
    portefeuille_complete = portefeuille_initial + (ligne_de_couverture(instrument, couverture["options"]),)
    scenario_seul = portefeuille_au_scenario(portefeuille_initial, nouveau_S, nouvelle_sigma, temps_ecoule)
    scenario_complete = portefeuille_au_scenario(portefeuille_complete, nouveau_S, nouvelle_sigma, temps_ecoule)

    # Couverture en delta seul, pour comparaison : les mêmes actions entières que dans le premier onglet.
    actions_delta = arrondir_actions(couverture_actions(delta_portefeuille(portefeuille_initial)))
    actions_delta_gamma = couverture["actions"]

    # P&L des deux stratégies sur toute la grille de cours, puis au cours du scénario.
    spots = grille_spots(S0)
    valeur_seul = valeur_portefeuille(portefeuille_initial)
    valeur_complete = valeur_portefeuille(portefeuille_complete)
    courbe_seul = courbe_valeur_portefeuille(scenario_seul, spots)
    courbe_complete = courbe_valeur_portefeuille(scenario_complete, spots)

    return {
        "S0": S0,
        "S1": nouveau_S,
        "spots": spots,
        "options": couverture["options"],
        "actions": actions_delta_gamma,
        "texte_options": texte_ordre_options(couverture["options"], instrument),
        "texte_actions": texte_ordre_actions(actions_delta_gamma),
        # Grecques restantes une fois les deux couvertures en place ; une action a un delta de 1.
        "delta_residuel": delta_portefeuille(portefeuille_complete) + actions_delta_gamma,
        "gamma_residuel": gamma_portefeuille(portefeuille_complete),
        "vega_residuel": vega_portefeuille(portefeuille_complete),
        "theta_residuel": theta_portefeuille(portefeuille_complete),
        "pnl_delta": pnl_couvert(
            valeur_seul, valeur_portefeuille(scenario_seul), actions_delta, S0, nouveau_S
        ),
        "pnl_delta_gamma": pnl_couvert(
            valeur_complete, valeur_portefeuille(scenario_complete), actions_delta_gamma, S0, nouveau_S
        ),
        "pnl_delta_courbe": pnl_couvert(valeur_seul, courbe_seul, actions_delta, S0, spots),
        "pnl_delta_gamma_courbe": pnl_couvert(
            valeur_complete, courbe_complete, actions_delta_gamma, S0, spots
        ),
    }


# ---------------------------------------------------------------------------
# Section simulation (couverture discrète par Monte Carlo)
# ---------------------------------------------------------------------------

def simuler_trajectoires(S0, r, sigma, horizon, nb_pas, nb_trajectoires=NB_TRAJECTOIRES, graine=GRAINE_ALEATOIRE, q=0.0):
    """Trajectoires de cours en mouvement brownien géométrique : une ligne par trajectoire.

    La première colonne vaut S0, la dernière le cours à l'horizon (en années).
    """
    # Générateur avec graine : deux appels identiques donnent exactement les mêmes trajectoires.
    generateur = np.random.default_rng(graine)
    pas_de_temps = horizon / nb_pas

    # Accroissements du logarithme du cours sous la probabilité risque-neutre : tendance r - q,
    # car le dividende versé fait baisser le cours d'autant.
    chocs = generateur.standard_normal((nb_trajectoires, nb_pas))
    increments = (r - q - 0.5 * sigma ** 2) * pas_de_temps + sigma * np.sqrt(pas_de_temps) * chocs

    # Somme cumulée des accroissements, en partant de zéro pour que la première colonne vaille S0.
    log_rendements = np.concatenate(
        [np.zeros((nb_trajectoires, 1)), np.cumsum(increments, axis=1)], axis=1
    )
    return S0 * np.exp(log_rendements)


def pnl_couverture_discrete(portefeuille, trajectoires, horizon, intervalle_pas, cout_relatif=0.0):
    """P&L à l'horizon du portefeuille couvert en delta, réajusté tous les intervalle_pas pas.

    cout_relatif est le coût de chaque ordre sur actions, en fraction du montant
    échangé : 0,001 pour 10 points de base. Il s'applique à la mise en place
    de la couverture et à chaque réajustement, pas au débouclage final.

    Le portefeuille est valorisé et couvert avec sa propre volatilité, dite
    implicite ; les trajectoires peuvent avoir été simulées avec une autre,
    dite réalisée. Avec une couverture continue et des volatilités égales, ce
    P&L serait nul : c'est tout le principe de Black-Scholes.
    """
    nb_trajectoires, nb_points = trajectoires.shape
    nb_pas = nb_points - 1
    pas_de_temps = horizon / nb_pas
    taux = portefeuille[0].r
    dividende = portefeuille[0].rendement_dividende

    # Mise en place : les options, leur couverture en delta exact, et le cash qui rend la valeur initiale nulle.
    actions = np.full(nb_trajectoires, couverture_actions(delta_portefeuille(portefeuille)))
    cash = -(valeur_portefeuille(portefeuille) + actions * trajectoires[:, 0])

    # L'achat ou la vente des premières actions de couverture paie déjà le coût de transaction.
    cash = cash - cout_relatif * np.abs(actions) * trajectoires[:, 0]

    for pas in range(1, nb_pas + 1):
        # Le cash, positif ou négatif, se capitalise au taux sans risque entre deux dates.
        cash = cash * np.exp(taux * pas_de_temps)

        # Les actions détenues touchent le dividende ; vendues à découvert, elles le paient au prêteur.
        cash = cash + actions * trajectoires[:, pas - 1] * dividende * pas_de_temps

        # Réajustement aux dates prévues : nouvelle couverture avec la maturité restante, financée par le cash.
        if pas < nb_pas and pas % intervalle_pas == 0:
            cours = trajectoires[:, pas]
            portefeuille_du_jour = portefeuille_a_date(portefeuille, pas * pas_de_temps)
            nouvelles_actions = courbe_couverture_portefeuille(portefeuille_du_jour, cours)
            actions_echangees = nouvelles_actions - actions
            cash = cash - actions_echangees * cours - cout_relatif * np.abs(actions_echangees) * cours
            actions = nouvelles_actions

    # À l'horizon : valeur des options, des actions détenues et du cash accumulé.
    cours_final = trajectoires[:, -1]
    valeur_options = courbe_valeur_portefeuille(portefeuille_a_date(portefeuille, horizon), cours_final)
    return valeur_options + actions * cours_final + cash


def statistiques_pnl(pnl):
    """Moyenne, écart-type, quantile à 5 % et erreur quadratique d'une série de P&L simulés."""
    # Le quantile à 5 % se lit comme la perte dépassée une fois sur vingt.
    # L'erreur quadratique combine le biais (la moyenne, que les coûts creusent) et la dispersion.
    return {
        "moyenne": float(np.mean(pnl)),
        "ecart_type": float(np.std(pnl)),
        "quantile_5": float(np.quantile(pnl, 0.05)),
        "erreur_quadratique": float(np.sqrt(np.mean(pnl ** 2))),
    }


def simuler_couverture(portefeuille, sigma_realisee, horizon_jours, intervalles=INTERVALLES_REAJUSTEMENT, nb_trajectoires=NB_TRAJECTOIRES, graine=GRAINE_ALEATOIRE, cout_relatif=0.0):
    """Compare plusieurs fréquences de réajustement sur les mêmes trajectoires de cours.

    Renvoie un dictionnaire indexé par l'intervalle de réajustement en jours,
    avec pour chacun les P&L simulés nets des coûts, le cours final de chaque
    trajectoire, les statistiques et le coût moyen de la couverture.
    """
    if horizon_jours < 1:
        raise ValueError("L'horizon de simulation doit être d'au moins un jour.")

    # Un pas par jour ; les mêmes trajectoires servent à toutes les fréquences, pour une comparaison équitable.
    horizon = jours_vers_annees(horizon_jours)
    trajectoires = simuler_trajectoires(
        portefeuille[0].S, portefeuille[0].r, sigma_realisee, horizon, horizon_jours,
        nb_trajectoires, graine, portefeuille[0].rendement_dividende,
    )

    # Le cours final accompagne chaque P&L : il explique pourquoi une trajectoire a gagné ou perdu.
    cours_final = trajectoires[:, -1]
    resultats = {}
    for intervalle in intervalles:
        # Deux calculs sur les mêmes trajectoires : leur écart isole exactement ce que coûtent les ordres.
        pnl_brut = pnl_couverture_discrete(portefeuille, trajectoires, horizon, intervalle)
        pnl_net = pnl_couverture_discrete(portefeuille, trajectoires, horizon, intervalle, cout_relatif)
        resultats[intervalle] = {
            "pnl": pnl_net,
            "cours_final": cours_final,
            "cout_moyen": float(np.mean(pnl_brut - pnl_net)),
            **statistiques_pnl(pnl_net),
        }
    return resultats


def frequence_optimale(resultats):
    """Intervalle de réajustement dont l'erreur quadratique est la plus faible."""
    # Réajuster souvent réduit la dispersion mais multiplie les coûts : ce critère arbitre entre les deux.
    return min(resultats, key=lambda intervalle: resultats[intervalle]["erreur_quadratique"])


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

    # P&L par rapport à la situation initiale, la couverture en place n'étant pas réajustée.
    valeur_initiale = resume_initial["valeur"]
    actions_en_place = resume_initial["actions_a_detenir"]
    valeurs_scenario = courbe_valeur_portefeuille(portefeuille_scenario, spots)

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
        "pnl_non_couvert": pnl_non_couvert(valeur_initiale, resume_scenario["valeur"]),
        "pnl_couvert": pnl_couvert(
            valeur_initiale, resume_scenario["valeur"], actions_en_place, S0, nouveau_S
        ),
        "pnl_non_couvert_courbe": pnl_non_couvert(valeur_initiale, valeurs_scenario),
        "pnl_couvert_courbe": pnl_couvert(
            valeur_initiale, valeurs_scenario, actions_en_place, S0, spots
        ),
    }
