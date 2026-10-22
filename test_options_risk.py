"""Tests du module options_risk, organisés dans le même ordre que le module testé.

Lancement : pytest -v
"""

import numpy as np
import pytest

from options_risk import (
    AMPLITUDE_GRILLE,
    NB_POINTS_GRILLE,
    Position,
    ajustement_portefeuille,
    analyser_couverture_delta_gamma,
    analyser_scenario,
    INTERVALLES_REAJUSTEMENT,
    appliquer_scenario,
    arrondir_actions,
    bs_delta,
    bs_gamma,
    bs_greeks,
    bs_price,
    bs_theta,
    bs_vega,
    construire_instrument,
    construire_portefeuille,
    couverture_delta_gamma,
    courbe_couverture_portefeuille,
    courbe_delta,
    courbe_delta_portefeuille,
    courbe_gamma,
    courbe_gamma_portefeuille,
    courbe_valeur,
    courbe_valeur_portefeuille,
    couverture_actions,
    decimal_vers_pourcent,
    delta_portefeuille,
    delta_position,
    detail_lignes,
    est_ordre_nul,
    facteur_position,
    frequence_optimale,
    gamma_portefeuille,
    gamma_position,
    grille_spots,
    jours_vers_annees,
    ligne_de_couverture,
    nombre_jours_simulables,
    pnl_couvert,
    points_de_base_vers_decimal,
    pnl_couverture_discrete,
    pnl_non_couvert,
    portefeuille_a_date,
    portefeuille_au_scenario,
    pourcent_vers_decimal,
    prix_unitaire,
    resume_portefeuille,
    signe_position,
    simuler_couverture,
    simuler_trajectoires,
    statistiques_pnl,
    strikes_visibles,
    texte_ordre_actions,
    texte_ordre_options,
    texte_zones_instabilite,
    theta_portefeuille,
    theta_position,
    valeur_portefeuille,
    valeur_position,
    valider_instrument,
    valider_marche,
    valider_portefeuille,
    variation_resume,
    vega_portefeuille,
    vega_position,
    zones_instabilite,
)

# Jeu de paramètrès de référence, repris de l'énoncé du projet.
S_REF, K_REF, T_REF, R_REF, SIGMA_REF = 100.0, 100.0, 0.5, 0.03, 0.20

# Rendement du dividende utilisé pour tester le modèle de Merton : 3 %, courant pour une grande valeur.
Q_REF = 0.03


# ---------------------------------------------------------------------------
# Section pricing
# ---------------------------------------------------------------------------

def test_prix_call_de_reference():
    """Le call de l'énoncé doit valoir environ 6,4 euros."""
    # Valeur cible donnée dans le sujet : c'est le premier garde-fou du module.
    prix = bs_price(S_REF, K_REF, T_REF, R_REF, SIGMA_REF, "call")
    assert prix == pytest.approx(6.37, abs=0.05)


def test_parite_call_put():
    """Call moins put doit égaler S moins le strike actualisé."""
    # Relation d'arbitrage indépendante du modèle : si elle casse, une des deux formules est fausse.
    call = bs_price(S_REF, K_REF, T_REF, R_REF, SIGMA_REF, "call")
    put = bs_price(S_REF, K_REF, T_REF, R_REF, SIGMA_REF, "put")
    attendu = S_REF - K_REF * np.exp(-R_REF * T_REF)
    assert call - put == pytest.approx(attendu, abs=1e-10)


def test_prix_toujours_positifs():
    """Un prix d'option ne peut pas être négatif, même très en dehors de la monnaie."""
    # Cas limites : sous-jacent très bas puis très haut.
    for spot in (10.0, 1000.0):
        assert bs_price(spot, K_REF, T_REF, R_REF, SIGMA_REF, "call") >= 0
        assert bs_price(spot, K_REF, T_REF, R_REF, SIGMA_REF, "put") >= 0


def test_prix_superieur_a_la_valeur_intrinseque():
    """Le prix d'un call doit dépasser sa valeur intrinsèque actualisée."""
    # Borne inferieure théorique : max(S - K*exp(-rT), 0).
    spot = 130.0
    borne = max(spot - K_REF * np.exp(-R_REF * T_REF), 0.0)
    assert bs_price(spot, K_REF, T_REF, R_REF, SIGMA_REF, "call") >= borne


def test_maturite_tres_courte_converge_vers_la_valeur_intrinseque():
    """À maturité quasi nulle, le call vaut sa valeur intrinsèque."""
    # Cas limite demandé dans la recette : T très petit ne doit pas faire diverger le prix.
    prix = bs_price(110.0, K_REF, 1e-6, R_REF, SIGMA_REF, "call")
    assert prix == pytest.approx(10.0, abs=1e-3)


def test_type_option_invalide():
    """Un type d'option mal orthographié leve une erreur explicite."""
    # On veut une erreur immédiate plutôt qu'un prix silencieusement faux.
    with pytest.raises(ValueError):
        bs_price(S_REF, K_REF, T_REF, R_REF, SIGMA_REF, "CALL")


# ---------------------------------------------------------------------------
# Section grecques
# ---------------------------------------------------------------------------

def test_delta_call_de_reference():
    """Le delta du call de l'énoncé doit valoir environ 0,57."""
    # Seconde valeur cible fournie par le sujet.
    assert bs_delta(S_REF, K_REF, T_REF, R_REF, SIGMA_REF, "call") == pytest.approx(0.57, abs=0.01)


def test_bornes_des_deltas():
    """Le delta d'un call reste dans (0 ; 1), celui d'un put dans (-1 ; 0)."""
    # Vérification du domaine sur toute la plage de cours utilisée par les graphiques.
    spots = grille_spots(S_REF)
    deltas_call = bs_delta(spots, K_REF, T_REF, R_REF, SIGMA_REF, "call")
    deltas_put = bs_delta(spots, K_REF, T_REF, R_REF, SIGMA_REF, "put")
    assert np.all((deltas_call > 0) & (deltas_call < 1))
    assert np.all((deltas_put > -1) & (deltas_put < 0))


def test_delta_par_differences_finies():
    """Le delta analytique doit coïncider avec la dérivée numérique du prix."""
    # Difference centree : l'erreur est en h², donc négligeable devant la tolérance.
    pas = 1e-4 * S_REF
    for type_option in ("call", "put"):
        prix_haut = bs_price(S_REF + pas, K_REF, T_REF, R_REF, SIGMA_REF, type_option)
        prix_bas = bs_price(S_REF - pas, K_REF, T_REF, R_REF, SIGMA_REF, type_option)
        delta_numerique = (prix_haut - prix_bas) / (2 * pas)
        delta_analytique = bs_delta(S_REF, K_REF, T_REF, R_REF, SIGMA_REF, type_option)
        assert delta_analytique == pytest.approx(delta_numerique, rel=1e-5)


def test_gamma_par_differences_finies():
    """Le gamma analytique doit coïncider avec la dérivée seconde numérique du prix."""
    # Pas plus large que pour le delta : une dérivée seconde amplifie les erreurs d'arrondi.
    pas = 0.5
    prix_haut = bs_price(S_REF + pas, K_REF, T_REF, R_REF, SIGMA_REF, "call")
    prix_centre = bs_price(S_REF, K_REF, T_REF, R_REF, SIGMA_REF, "call")
    prix_bas = bs_price(S_REF - pas, K_REF, T_REF, R_REF, SIGMA_REF, "call")
    gamma_numerique = (prix_haut - 2 * prix_centre + prix_bas) / pas ** 2
    assert bs_gamma(S_REF, K_REF, T_REF, R_REF, SIGMA_REF) == pytest.approx(
        gamma_numerique, rel=1e-4
    )


def test_gamma_identique_call_et_put():
    """Le gamma ne dépend pas du type d'option."""
    # Conséquence directe de la parité call-put : la différence des prix est linéaire en S.
    greeks_call = bs_greeks(S_REF, K_REF, T_REF, R_REF, SIGMA_REF, "call")
    greeks_put = bs_greeks(S_REF, K_REF, T_REF, R_REF, SIGMA_REF, "put")
    assert greeks_call["gamma"] == pytest.approx(greeks_put["gamma"], abs=1e-12)


def test_gamma_maximal_a_la_monnaie():
    """Le gamma d'une option est maximal autour du strike."""
    # C'est l'information que le second graphique doit faire ressortir visuellement.
    gamma_monnaie = bs_gamma(100.0, K_REF, T_REF, R_REF, SIGMA_REF)
    assert gamma_monnaie > bs_gamma(60.0, K_REF, T_REF, R_REF, SIGMA_REF)
    assert gamma_monnaie > bs_gamma(140.0, K_REF, T_REF, R_REF, SIGMA_REF)


def test_vega_par_differences_finies():
    """Le vega analytique coïncide avec la dérivée numérique du prix par rapport à sigma."""
    # Division par 100 : le vega est exprimé pour un point de volatilité.
    pas = 1e-4
    for type_option in ("call", "put"):
        haut = bs_price(S_REF, K_REF, T_REF, R_REF, SIGMA_REF + pas, type_option)
        bas = bs_price(S_REF, K_REF, T_REF, R_REF, SIGMA_REF - pas, type_option)
        attendu = (haut - bas) / (2 * pas) / 100
        assert bs_vega(S_REF, K_REF, T_REF, R_REF, SIGMA_REF) == pytest.approx(attendu, rel=1e-5)


def test_theta_par_differences_finies():
    """Le theta analytique coïncide avec la variation numérique du prix quand la maturité diminue."""
    # Theta = -dV/dT, ramené à un jour : c'est la perte de valeur d'une journée qui passe.
    pas = 1e-5
    for type_option in ("call", "put"):
        plus_long = bs_price(S_REF, K_REF, T_REF + pas, R_REF, SIGMA_REF, type_option)
        plus_court = bs_price(S_REF, K_REF, T_REF - pas, R_REF, SIGMA_REF, type_option)
        attendu = -(plus_long - plus_court) / (2 * pas) / 365
        assert bs_theta(S_REF, K_REF, T_REF, R_REF, SIGMA_REF, type_option) == pytest.approx(
            attendu, rel=1e-4
        )


def test_theta_du_call_de_reference_negatif():
    """Un call à la monnaie perd de la valeur chaque jour."""
    # Ordre de grandeur : environ 1,9 centime par jour pour le call de l'énoncé.
    theta = bs_theta(S_REF, K_REF, T_REF, R_REF, SIGMA_REF, "call")
    assert theta == pytest.approx(-0.0194, abs=0.001)


def test_bs_greeks_contient_les_quatre_grecques():
    """Le dictionnaire des grecques contient delta, gamma, vega et theta."""
    # Fonction d'assemblage : on vérifie seulement ce qu'elle rassemble.
    assert set(bs_greeks(S_REF, K_REF, T_REF, R_REF, SIGMA_REF, "put")) == {
        "delta", "gamma", "vega", "theta"
    }


def test_dividende_nul_par_defaut():
    """Sans dividende précisé, les fonctions donnent exactement Black-Scholes, comme vu en séance."""
    # Garantit que l'ajout du paramètre q ne change aucun appel existant.
    for type_option in ("call", "put"):
        assert bs_price(S_REF, K_REF, T_REF, R_REF, SIGMA_REF, type_option) == bs_price(
            S_REF, K_REF, T_REF, R_REF, SIGMA_REF, type_option, 0.0
        )


def test_parite_call_put_avec_dividende():
    """Avec dividende, call moins put égale le cours hors dividendes moins le strike actualisé."""
    # C - P = S exp(-qT) - K exp(-rT) : le dividende retire au porteur de l'action ce qu'il verse.
    call = bs_price(S_REF, K_REF, T_REF, R_REF, SIGMA_REF, "call", Q_REF)
    put = bs_price(S_REF, K_REF, T_REF, R_REF, SIGMA_REF, "put", Q_REF)
    attendu = S_REF * np.exp(-Q_REF * T_REF) - K_REF * np.exp(-R_REF * T_REF)
    assert call - put == pytest.approx(attendu, abs=1e-10)


def test_valeurs_de_reference_avec_dividende():
    """Avec un dividende de 3 %, le call de l'énoncé vaut 5,55 € et son delta 0,520."""
    # Valeurs recalculées à la main : le dividende abaisse le call de 6,37 € à 5,55 €.
    assert bs_price(S_REF, K_REF, T_REF, R_REF, SIGMA_REF, "call", Q_REF) == pytest.approx(5.5533, abs=1e-4)
    assert bs_delta(S_REF, K_REF, T_REF, R_REF, SIGMA_REF, "call", Q_REF) == pytest.approx(0.5203, abs=1e-4)


def test_dividende_egal_au_taux_call_egal_put():
    """À la monnaie, quand le dividende égale le taux, le call et le put ont le même prix."""
    # Le cours à terme vaut alors exactement le cours actuel, d'où la symétrie.
    call = bs_price(S_REF, K_REF, T_REF, R_REF, SIGMA_REF, "call", R_REF)
    put = bs_price(S_REF, K_REF, T_REF, R_REF, SIGMA_REF, "put", R_REF)
    assert call == pytest.approx(put, abs=1e-10)


def test_dividende_fait_baisser_le_call_et_monter_le_put():
    """Un dividende plus élevé abaisse le call et relève le put."""
    # Le cours baisse du montant versé : mauvaise nouvelle pour l'acheteur de call, bonne pour celui de put.
    sans = {t: bs_price(S_REF, K_REF, T_REF, R_REF, SIGMA_REF, t) for t in ("call", "put")}
    avec = {t: bs_price(S_REF, K_REF, T_REF, R_REF, SIGMA_REF, t, Q_REF) for t in ("call", "put")}
    assert avec["call"] < sans["call"]
    assert avec["put"] > sans["put"]


def test_grecques_avec_dividende_par_differences_finies():
    """Avec dividende, delta, gamma, vega et theta coïncident avec les dérivées numériques du prix."""
    # Même méthode que sans dividende, appliquée cette fois au modèle de Merton.
    def prix(type_option, S=S_REF, T=T_REF, sigma=SIGMA_REF):
        return bs_price(S, K_REF, T, R_REF, sigma, type_option, Q_REF)

    pas_S, pas_T, pas_sigma = 1e-3 * S_REF, 1e-5, 1e-4
    for type_option in ("call", "put"):
        delta = (prix(type_option, S=S_REF + pas_S) - prix(type_option, S=S_REF - pas_S)) / (2 * pas_S)
        gamma = (prix(type_option, S=S_REF + pas_S) - 2 * prix(type_option) + prix(type_option, S=S_REF - pas_S)) / pas_S ** 2
        vega = (prix(type_option, sigma=SIGMA_REF + pas_sigma) - prix(type_option, sigma=SIGMA_REF - pas_sigma)) / (2 * pas_sigma) / 100
        theta = -(prix(type_option, T=T_REF + pas_T) - prix(type_option, T=T_REF - pas_T)) / (2 * pas_T) / 365
        assert bs_delta(S_REF, K_REF, T_REF, R_REF, SIGMA_REF, type_option, Q_REF) == pytest.approx(delta, rel=1e-5)
        assert bs_gamma(S_REF, K_REF, T_REF, R_REF, SIGMA_REF, Q_REF) == pytest.approx(gamma, rel=1e-4)
        assert bs_vega(S_REF, K_REF, T_REF, R_REF, SIGMA_REF, Q_REF) == pytest.approx(vega, rel=1e-5)
        assert bs_theta(S_REF, K_REF, T_REF, R_REF, SIGMA_REF, type_option, Q_REF) == pytest.approx(theta, rel=1e-4)


# ---------------------------------------------------------------------------
# Section position
# ---------------------------------------------------------------------------

def construire_position(type_position, quantite=100.0, S=S_REF):
    """Fabrique une position de test avec les paramètrès de référence."""
    # Évite de répéter les sept arguments dans chaque test.
    return Position(type_position, quantite, S, K_REF, T_REF, R_REF, SIGMA_REF)


def test_signes_du_delta_sur_les_quatre_positions():
    """Long call et short put ont un delta positif, short call et long put un delta négatif."""
    # Cœur de la convention de signes : une erreur ici inverserait tous les ordres de couverture.
    assert delta_position(construire_position("long call")) > 0
    assert delta_position(construire_position("short call")) < 0
    assert delta_position(construire_position("long put")) < 0
    assert delta_position(construire_position("short put")) > 0


def test_signes_du_gamma_sur_les_quatre_positions():
    """Le gamma est positif à l'achat et négatif à la vente, call comme put."""
    # Point explicitement vérifié par le sujet.
    assert gamma_position(construire_position("long call")) > 0
    assert gamma_position(construire_position("long put")) > 0
    assert gamma_position(construire_position("short call")) < 0
    assert gamma_position(construire_position("short put")) < 0


def test_valeur_position_negative_pour_une_vente():
    """La valeur de marche d'une position vendeuse est négative."""
    # Convention retenue : c'est le montant à payer pour déboucler la position.
    assert valeur_position(construire_position("long call")) > 0
    assert valeur_position(construire_position("short call")) < 0


def test_valeur_position_proportionnelle_a_la_quantite():
    """Doubler la quantité double la valeur de la position."""
    # Vérifie que la quantité est appliquée une seule fois, et au bon endroit.
    valeur_cent = valeur_position(construire_position("long call", quantite=100.0))
    valeur_deux_cents = valeur_position(construire_position("long call", quantite=200.0))
    assert valeur_deux_cents == pytest.approx(2 * valeur_cent)


def test_couverture_annule_le_delta_global():
    """Delta de la position plus delta des actions de couverture doit donner zéro."""
    # Définition même de la couverture en delta : une action a un delta de 1.
    position = construire_position("long call")
    delta = delta_position(position)
    assert delta + couverture_actions(delta) == pytest.approx(0.0, abs=1e-12)


def test_couverture_du_long_call_est_une_vente():
    """Un achat de 100 calls se couvre en vendant environ 57 actions."""
    # Exemple de référence du sujet, avec le sens écrit en clair.
    position = construire_position("long call")
    couverture = couverture_actions(delta_position(position))
    assert couverture == pytest.approx(-57.0, abs=1.0)
    assert texte_ordre_actions(couverture).startswith("vendre")


def test_texte_ordre_actions():
    """Le sens de l'ordre et le pluriel sont corrects, et un ordre négligeable est signalé."""
    # Message lu par le trader : il doit être sans ambiguïté.
    assert texte_ordre_actions(57.4) == "acheter 57 actions"
    assert texte_ordre_actions(-57.4) == "vendre 57 actions"
    assert texte_ordre_actions(1.0) == "acheter 1 action"
    assert texte_ordre_actions(0.0) == "aucune action à échanger"
    # Une demi-action s'arrondit à une action, jamais à « acheter 0 actions ».
    assert texte_ordre_actions(0.5) == "acheter 1 action"
    assert texte_ordre_actions(-0.5) == "vendre 1 action"
    assert texte_ordre_actions(2.5) == "acheter 3 actions"


def test_type_position_invalide():
    """Une position mal nommee est refusée dès la construction."""
    # Protège les propriétés sens et type_option, qui découpent la chaîne de caracteres.
    with pytest.raises(ValueError):
        Position("long future", 100.0, S_REF, K_REF, T_REF, R_REF, SIGMA_REF)


def test_signe_position():
    """Le signe vaut +1 pour un achat et -1 pour une vente."""
    # Brique elementaire reutilisée par le prix, le delta et le gamma.
    assert signe_position("long") == 1
    assert signe_position("short") == -1


def test_prix_unitaire_independant_du_sens():
    """Le prix d'une option est le même à l'achat et à la vente."""
    # Seul le signe de la position change, pas la valeur de l'instrument.
    assert prix_unitaire(construire_position("long call")) == pytest.approx(
        prix_unitaire(construire_position("short call"))
    )


def test_arrondir_actions():
    """Les demis s'arrondissent en s'éloignant de zéro, pas vers le nombre pair."""
    # round() de Python donnerait 2 pour 2,5 et 0 pour 0,5 : c'est le piège évité ici.
    assert arrondir_actions(0.5) == 1
    assert arrondir_actions(-0.5) == -1
    assert arrondir_actions(2.5) == 3
    assert arrondir_actions(2.4) == 2
    assert arrondir_actions(-2.6) == -3
    assert isinstance(arrondir_actions(57.02), int)


def test_est_ordre_nul():
    """Un ordre est nul exactement quand il s'arrondit à zéro action."""
    # Même règle que le texte des ordres : les deux ne peuvent pas se contredire.
    assert est_ordre_nul(0.4)
    assert est_ordre_nul(-0.4)
    assert not est_ordre_nul(0.5)
    assert not est_ordre_nul(-0.6)


def test_facteur_position():
    """Le facteur vaut plus la quantité à l'achat et moins la quantité à la vente."""
    # Brique commune à toutes les grandeurs d'une ligne.
    assert facteur_position(construire_position("long put", quantite=40.0)) == 40.0
    assert facteur_position(construire_position("short call", quantite=40.0)) == -40.0


def test_signes_du_vega_et_du_theta():
    """L'acheteur est long vega et paie le theta, le vendeur l'inverse."""
    # Pour un call à la monnaie, le theta de l'achat est négatif.
    assert vega_position(construire_position("long put")) > 0
    assert vega_position(construire_position("short call")) < 0
    assert theta_position(construire_position("long call")) < 0
    assert theta_position(construire_position("short call")) > 0


# ---------------------------------------------------------------------------
# Section portefeuille
# ---------------------------------------------------------------------------

def construire_lignes(*types_positions, K=K_REF, T=T_REF, quantite=100):
    """Fabrique des lignes saisies, une par type de position demandé."""
    # Même format que celui renvoyé par le tableau de l'interface.
    return [
        {"type_position": type_position, "quantite": quantite, "K": K, "T": T}
        for type_position in types_positions
    ]


def portefeuille_de_reference(*types_positions):
    """Construit un portefeuille de test avec les paramètres de marché de référence."""
    # Évite de répéter S, r et sigma dans chaque test.
    return construire_portefeuille(construire_lignes(*types_positions), S_REF, R_REF, SIGMA_REF)


def test_portefeuille_d_une_ligne_egal_position_seule():
    """Un portefeuille d'une seule ligne donne exactement les chiffres de la position seule."""
    # Garantit que le passage au portefeuille ne change rien au cas de base du sujet.
    portefeuille = portefeuille_de_reference("long call")
    position = construire_position("long call")
    assert valeur_portefeuille(portefeuille) == pytest.approx(valeur_position(position))
    assert delta_portefeuille(portefeuille) == pytest.approx(delta_position(position))
    assert gamma_portefeuille(portefeuille) == pytest.approx(gamma_position(position))


def test_grecques_agregees_egales_a_la_somme_des_lignes():
    """Le delta et le gamma du portefeuille sont la somme de ceux de chaque ligne prise seule."""
    # Additivité des grecques sur un même sous-jacent, vérifiée contre des positions construites à part.
    portefeuille = portefeuille_de_reference("long call", "short put")
    call_seul = construire_position("long call")
    put_seul = construire_position("short put")
    assert delta_portefeuille(portefeuille) == pytest.approx(
        delta_position(call_seul) + delta_position(put_seul)
    )
    assert gamma_portefeuille(portefeuille) == pytest.approx(
        gamma_position(call_seul) + gamma_position(put_seul)
    )


def test_achat_et_vente_identiques_s_annulent():
    """Acheter et vendre la même option donne un delta et un gamma nuls."""
    # Cas d'école : aucune couverture n'est nécessaire et le texte doit le dire.
    resume = resume_portefeuille(portefeuille_de_reference("long call", "short call"))
    assert resume["delta"] == pytest.approx(0.0, abs=1e-12)
    assert resume["gamma"] == pytest.approx(0.0, abs=1e-12)
    assert resume["texte_couverture"] == "aucune action à échanger"


def test_straddle_a_un_gamma_double():
    """Un achat de call et de put de même strike a deux fois le gamma d'un call seul."""
    # Le call et le put ont le même gamma unitaire, et les deux sont achetés.
    straddle = portefeuille_de_reference("long call", "long put")
    assert gamma_portefeuille(straddle) == pytest.approx(
        2 * gamma_position(construire_position("long call"))
    )


def test_construire_portefeuille_partage_les_parametres_de_marche():
    """Toutes les lignes reçoivent le même cours, le même taux et la même volatilité."""
    # Hypothèse du modèle : un seul sous-jacent, donc une seule volatilité.
    portefeuille = portefeuille_de_reference("long call", "short put")
    assert {(p.S, p.r, p.sigma) for p in portefeuille} == {(S_REF, R_REF, SIGMA_REF)}


def test_detail_lignes():
    """Le détail contient une entrée par ligne, avec le bon prix unitaire."""
    # Le sujet demande le prix unitaire : il est désormais affiché ligne par ligne.
    portefeuille = portefeuille_de_reference("long call", "short put")
    detail = detail_lignes(portefeuille)
    assert len(detail) == 2
    assert detail[0]["prix_unitaire"] == pytest.approx(prix_unitaire(portefeuille[0]))
    assert detail[1]["valeur"] < 0


def test_variation_resume_nulle_entre_resumes_identiques():
    """L'écart entre un résumé et lui-même est nul sur tous les indicateurs."""
    # Garde-fou simple contre une inversion entre résumé initial et résumé du scénario.
    resume = resume_portefeuille(portefeuille_de_reference("long call"))
    assert all(ecart == 0 for ecart in variation_resume(resume, resume).values())


def test_ajustement_de_l_enonce_quand_le_cours_monte_a_112():
    """Si le cours passe de 100 à 112 €, les 100 calls imposent de vendre 27 actions de plus."""
    # Valeur recalculée à la main : 84 actions à détenir au lieu de 57 ; le gamma positif fait vendre en hausse.
    portefeuille = portefeuille_de_reference("long call")
    scenario = portefeuille_au_scenario(portefeuille, 112.0, SIGMA_REF, 0.0)
    assert ajustement_portefeuille(portefeuille, scenario) == -27


def test_ajustement_nul_si_le_cours_ne_bouge_pas():
    """Aucun ajustement n'est nécessaire si le marché n'a pas bougé."""
    # Garde-fou simple contre une inversion entre couverture en place et couverture cible.
    portefeuille = portefeuille_de_reference("long call")
    scenario = portefeuille_au_scenario(portefeuille, S_REF, SIGMA_REF, 0.0)
    assert ajustement_portefeuille(portefeuille, scenario) == 0


def test_resume_portefeuille_contient_toutes_les_cles():
    """Le résumé alimente l'encart de résultats avec toutes les informations attendues."""
    # Si une clé disparaissait, l'interface planterait à l'affichage.
    resume = resume_portefeuille(portefeuille_de_reference("long call"))
    assert set(resume) == {
        "valeur", "delta", "gamma", "vega", "theta",
        "couverture", "actions_a_detenir", "texte_couverture",
    }


def test_vega_et_theta_agreges():
    """Le vega et le theta du portefeuille sont la somme de ceux des lignes."""
    # Un achat et une vente identiques s'annulent aussi sur ces grecques.
    resume = resume_portefeuille(portefeuille_de_reference("long call", "short call"))
    assert resume["vega"] == pytest.approx(0.0, abs=1e-12)
    assert resume["theta"] == pytest.approx(0.0, abs=1e-12)
    straddle = portefeuille_de_reference("long call", "long put")
    assert vega_portefeuille(straddle) == pytest.approx(2 * vega_position(construire_position("long call")))
    assert theta_portefeuille(straddle) < 0


def test_construire_portefeuille_transmet_le_dividende():
    """Le dividende saisi est donné à chaque ligne, et la couverture en tient compte."""
    # Avec 3 % de dividende, les 100 calls de l'énoncé se couvrent en vendant 52 actions au lieu de 57.
    lignes = construire_lignes("long call", "short put")
    portefeuille = construire_portefeuille(lignes, S_REF, R_REF, SIGMA_REF, Q_REF)
    assert all(position.rendement_dividende == Q_REF for position in portefeuille)
    seul = construire_portefeuille(construire_lignes("long call"), S_REF, R_REF, SIGMA_REF, Q_REF)
    assert resume_portefeuille(seul)["texte_couverture"] == "vendre 52 actions"


# ---------------------------------------------------------------------------
# Section scénario
# ---------------------------------------------------------------------------

def test_jours_vers_annees():
    """365 jours font une année, 73 jours en font 0,2."""
    # Convention de calendrier fixée par la constante JOURS_PAR_AN.
    assert jours_vers_annees(365) == pytest.approx(1.0)
    assert jours_vers_annees(73) == pytest.approx(0.2)


def test_nombre_jours_simulables_limite_par_l_option_la_plus_courte():
    """L'option la plus courte fixe le nombre de jours que l'on peut faire passer."""
    # 0,1 an fait 36,5 jours : on en garde un, il reste 35 jours simulables.
    lignes = construire_lignes("long call", T=0.5) + construire_lignes("long put", T=0.1)
    portefeuille = construire_portefeuille(lignes, S_REF, R_REF, SIGMA_REF)
    assert nombre_jours_simulables(portefeuille) == 35


def test_nombre_jours_simulables_nul_pour_une_maturite_tres_courte():
    """Avec un seul jour de maturité, on ne peut faire passer aucun jour."""
    # Cas limite : le curseur de temps doit alors être masqué par l'interface.
    lignes = construire_lignes("long call", T=1 / 365)
    portefeuille = construire_portefeuille(lignes, S_REF, R_REF, SIGMA_REF)
    assert nombre_jours_simulables(portefeuille) == 0


def test_appliquer_scenario_modifie_cours_volatilite_et_maturite():
    """Le scénario change le cours, la volatilité et raccourcit la maturité."""
    # La position d'origine doit rester intacte : la dataclass est figée.
    position = construire_position("long call")
    simulee = appliquer_scenario(position, 110.0, 0.25, 0.1)
    assert (simulee.S, simulee.sigma) == (110.0, 0.25)
    assert simulee.T == pytest.approx(0.4)
    assert position.T == T_REF


def test_appliquer_scenario_refuse_une_option_expiree():
    """Un temps écoulé supérieur à la maturité lève une erreur explicite."""
    # Black-Scholes n'est pas défini pour une maturité nulle ou négative.
    with pytest.raises(ValueError):
        appliquer_scenario(construire_position("long call"), S_REF, SIGMA_REF, T_REF)


def test_gamma_se_concentre_quand_l_echeance_approche():
    """À la monnaie, le gamma augmente quand la maturité se raccourcit."""
    # C'est la déformation que le curseur de temps doit faire apparaître sur le graphique.
    position = construire_position("long call")
    proche_echeance = appliquer_scenario(position, S_REF, SIGMA_REF, 0.45)
    assert gamma_position(proche_echeance) > gamma_position(position)


def test_gamma_a_la_monnaie_baisse_quand_la_volatilite_monte():
    """À la monnaie, une volatilité plus forte aplatit le gamma."""
    # La courbe s'étale : la couverture devient moins instable mais sur une plage plus large.
    position = construire_position("long call")
    volatilite_haute = appliquer_scenario(position, S_REF, 0.40, 0.0)
    assert gamma_position(volatilite_haute) < gamma_position(position)


def test_portefeuille_a_date():
    """Le portefeuille à une date future garde son cours et sa volatilité, mais vieillit."""
    # Utilisé par la simulation à chaque date de réajustement.
    plus_tard = portefeuille_a_date(portefeuille_de_reference("long call"), 0.1)
    assert plus_tard[0].T == pytest.approx(T_REF - 0.1)
    assert (plus_tard[0].S, plus_tard[0].sigma) == (S_REF, SIGMA_REF)


def test_pnl_non_couvert_et_couvert():
    """Le P&L couvert ajoute au P&L des options le gain des actions détenues."""
    # Vendre 57 actions et voir le cours monter de 2 € coûte 114 €.
    assert pnl_non_couvert(100.0, 150.0) == pytest.approx(50.0)
    assert pnl_couvert(100.0, 150.0, -57, 100.0, 102.0) == pytest.approx(50.0 - 114.0)


def test_pnl_couvert_proche_du_demi_gamma():
    """Pour un petit mouvement, le P&L couvert vaut environ ½ × gamma × (variation du cours)²."""
    # Toute la sensibilité de premier ordre a été neutralisée : il ne reste que la courbure.
    analyse = analyser_scenario(portefeuille_de_reference("long call"), S_REF + 1.0, SIGMA_REF, 0.0)
    attendu = 0.5 * analyse["resume_initial"]["gamma"] * 1.0 ** 2
    assert analyse["pnl_couvert"] == pytest.approx(attendu, rel=0.05)


def test_pnl_couvert_proche_du_theta_si_le_cours_ne_bouge_pas():
    """Sans mouvement de cours, dix jours de couverture coûtent environ dix fois le theta."""
    # C'est le prix payé par l'acheteur de gamma pour profiter des mouvements.
    portefeuille = portefeuille_de_reference("long call")
    analyse = analyser_scenario(portefeuille, S_REF, SIGMA_REF, 10 / 365)
    assert analyse["pnl_couvert"] == pytest.approx(10 * theta_portefeuille(portefeuille), rel=0.05)


# ---------------------------------------------------------------------------
# Section validation
# ---------------------------------------------------------------------------

def valider_une_position(type_position="long call", quantite=100, S=S_REF, K=K_REF, T=T_REF, r=R_REF, sigma=SIGMA_REF):
    """Valide une position seule, c'est-à-dire un portefeuille d'une ligne."""
    # Les cas du sujet portent sur une seule position : on les teste sous cette forme.
    ligne = {"type_position": type_position, "quantite": quantite, "K": K, "T": T}
    return valider_portefeuille([ligne], S, r, sigma)


def test_saisie_valide_ne_produit_aucune_erreur():
    """Une saisie correcte renvoie une liste vide."""
    # Cas nominal : la validation ne doit pas bloquer un utilisateur légitime.
    assert valider_une_position() == []


@pytest.mark.parametrize(
    "champ, valeur",
    [
        ("quantite", 0),
        ("quantite", -10),
        ("T", 0),
        ("T", -0.5),
        ("sigma", 0),
        ("sigma", -0.2),
        ("S", 0),
        ("K", -100),
    ],
)
def test_saisies_hors_domaine_refusees(champ, valeur):
    """Les valeurs nulles ou négatives interdites par le sujet sont refusées."""
    # Chaque cas est testé séparément pour identifier immédiatement le champ fautif.
    assert valider_une_position(**{champ: valeur}) != []


def test_champ_vide_refuse_sans_planter():
    """Un champ laissé vide produit un message d'erreur, pas une exception."""
    # Exigence explicite du sujet : l'outil ne doit jamais planter sur une saisie incomplète.
    erreurs = valider_une_position(quantite=None)
    assert len(erreurs) == 1


def test_taux_negatif_accepte():
    """Un taux sans risque négatif est économiquement possible et reste accepté."""
    # Seuls sigma, T, la quantité, S et K sont contraints à être strictement positifs.
    assert valider_une_position(r=-0.005) == []


def test_erreurs_cumulees():
    """Plusieurs saisies fautives produisent plusieurs messages."""
    # L'interface affiche tout d'un coup plutôt que de corriger erreur après erreur.
    erreurs = valider_une_position(quantite=0, T=0, sigma=0)
    assert len(erreurs) == 3


def test_pourcent_vers_decimal():
    """Un pourcentage saisi dans l'interface devient un taux décimal."""
    # Conversion isolée dans une fonction pour qu'aucun calcul ne traîne dans app.py.
    assert pourcent_vers_decimal(3) == pytest.approx(0.03)
    assert pourcent_vers_decimal(20) == pytest.approx(0.20)


def test_portefeuille_valide_sans_erreur():
    """Un portefeuille correct renvoie une liste vide."""
    # Cas nominal du tableau de saisie.
    lignes = construire_lignes("long call", "short put")
    assert valider_portefeuille(lignes, S_REF, R_REF, SIGMA_REF) == []


def test_portefeuille_vide_refuse():
    """Un tableau sans aucune ligne produit un message explicite."""
    # Arrive si l'utilisateur supprime toutes les lignes du tableau.
    assert valider_portefeuille([], S_REF, R_REF, SIGMA_REF) == [
        "Le portefeuille doit contenir au moins une option."
    ]


def test_case_vide_signalee_avec_son_numero_de_ligne():
    """Une case vide du tableau produit un message qui indique la ligne concernée."""
    # Le tableau renvoie NaN pour une case numérique effacée : c'est le cas « champ vide » du sujet.
    lignes = construire_lignes("long call", "short put")
    lignes[1]["K"] = float("nan")
    assert valider_portefeuille(lignes, S_REF, R_REF, SIGMA_REF) == [
        "Ligne 2 : valeur manquante pour le strike K."
    ]


def test_type_vide_signale():
    """Une ligne sans type de position est refusée."""
    # Une nouvelle ligne dont on a effacé le menu déroulant renvoie None.
    lignes = construire_lignes("long call")
    lignes[0]["type_position"] = None
    erreurs = valider_portefeuille(lignes, S_REF, R_REF, SIGMA_REF)
    assert len(erreurs) == 1
    assert erreurs[0].startswith("Ligne 1 : le type de position")


def test_erreurs_de_marche_et_de_ligne_cumulees():
    """Les erreurs de marché et de lignes sont toutes renvoyées ensemble."""
    # Volatilité nulle dans la barre latérale et quantité nulle dans le tableau.
    lignes = construire_lignes("long call", quantite=0)
    assert len(valider_portefeuille(lignes, S_REF, R_REF, 0)) == 2


def test_decimal_vers_pourcent():
    """Un taux décimal devient un pourcentage pour l'affichage."""
    # Opération inverse de pourcent_vers_decimal.
    assert decimal_vers_pourcent(0.2) == pytest.approx(20.0)


def test_points_de_base_vers_decimal():
    """Dix points de base valent 0,1 %, soit 0,001."""
    # Unité habituelle des coûts de transaction.
    assert points_de_base_vers_decimal(10) == pytest.approx(0.001)


def test_instrument_valide():
    """Un call proche du cours, de maturité suffisante, est accepté."""
    # Cas nominal de l'onglet delta-gamma.
    assert valider_instrument("call", 105.0, 0.5, S_REF, R_REF, SIGMA_REF, 0.0) == []


def test_instrument_sans_gamma_refuse():
    """Une option très loin de la monnaie n'a pas assez de gamma pour servir de couverture."""
    # Un call de strike 200 sur un cours de 100 a un gamma quasi nul à six mois.
    erreurs = valider_instrument("call", 200.0, 0.5, S_REF, R_REF, SIGMA_REF, 0.0)
    assert len(erreurs) == 1 and "gamma" in erreurs[0]


def test_instrument_qui_expire_pendant_le_scenario_refuse():
    """Un instrument qui expire avant la fin du scénario est refusé."""
    # 60 jours écoulés pour une option de 0,1 an, soit 36 jours : elle n'existe plus.
    erreurs = valider_instrument("put", 100.0, 0.1, S_REF, R_REF, SIGMA_REF, 60 / 365)
    assert len(erreurs) == 1 and "maturité" in erreurs[0]


def test_instrument_saisie_incomplete():
    """Un strike vide ou une maturité négative produisent un message, pas une exception."""
    # Même logique de validation que pour les lignes du portefeuille.
    assert valider_instrument("call", None, 0.5, S_REF, R_REF, SIGMA_REF, 0.0) != []
    assert valider_instrument("call", 105.0, -1.0, S_REF, R_REF, SIGMA_REF, 0.0) != []
    assert valider_instrument("future", 105.0, 0.5, S_REF, R_REF, SIGMA_REF, 0.0) != []


def test_validation_du_dividende():
    """Un dividende nul est accepté ; un dividende négatif ou absent est refusé."""
    # Le dividende vaut 0 par défaut, mais l'utilisateur peut vider ou fausser le champ.
    assert valider_marche(S_REF, R_REF, SIGMA_REF, 0.0) == []
    assert valider_marche(S_REF, R_REF, SIGMA_REF, -0.01) == ["Le rendement du dividende q ne peut pas être négatif."]
    assert valider_marche(S_REF, R_REF, SIGMA_REF, None) == ["Valeur manquante pour le rendement du dividende q."]


# ---------------------------------------------------------------------------
# Section courbes
# ---------------------------------------------------------------------------

def test_grille_spots_centree_et_de_bonne_largeur():
    """La grille couvre au moins ±40 % et le spot saisi tombe en son centre."""
    # Le point central doit exister pour que le point courant initial soit pile sur la courbe.
    spots = grille_spots(S_REF)
    assert len(spots) == NB_POINTS_GRILLE
    assert spots[0] == pytest.approx(S_REF * (1 - AMPLITUDE_GRILLE))
    assert spots[-1] == pytest.approx(S_REF * (1 + AMPLITUDE_GRILLE))
    assert spots[NB_POINTS_GRILLE // 2] == pytest.approx(S_REF)


def test_courbes_coherentes_avec_les_valeurs_ponctuelles():
    """Au centre de la grille, les courbes retrouvent exactement les chiffres de l'encart."""
    # Garantit que graphiques et encart de résultats racontent la même histoire.
    position = construire_position("short put")
    spots = grille_spots(position.S)
    indice_central = NB_POINTS_GRILLE // 2
    assert courbe_delta(position, spots)[indice_central] == pytest.approx(delta_position(position))
    assert courbe_gamma(position, spots)[indice_central] == pytest.approx(gamma_position(position))


def test_courbe_couverture_est_l_oppose_de_la_courbe_delta():
    """La courbe de couverture est le miroir de celle du delta."""
    # Relation structurelle : si elle casse, un des deux graphiques est faux.
    position = construire_position("long call")
    spots = grille_spots(position.S)
    assert np.allclose(courbe_couverture_portefeuille((position,), spots), -courbe_delta(position, spots))


def test_courbe_delta_monotone_pour_un_long_call():
    """Le delta d'un achat de calls croît avec le cours du sous-jacent."""
    # Traduction graphique du gamma positif.
    position = construire_position("long call")
    deltas = courbe_delta(position, grille_spots(position.S))
    assert np.all(np.diff(deltas) > 0)


def test_signe_de_la_courbe_de_gamma():
    """La courbe de gamma reste entièrement positive à l'achat et négative à la vente."""
    # Le sujet demande un signe juste pour les quatre positions, sur toute la plage.
    spots = grille_spots(S_REF)
    assert np.all(courbe_gamma(construire_position("long put"), spots) > 0)
    assert np.all(courbe_gamma(construire_position("short call"), spots) < 0)


def test_courbes_sans_valeur_aberrante():
    """Aucune courbe ne contient de NaN ni d'infini, même aux extrémités de la grille."""
    # Cas limites de la recette : cours très bas ou très haut ne doivent pas casser l'affichage.
    portefeuille = (construire_position("long call"),)
    spots = grille_spots(S_REF)
    for courbe in (courbe_couverture_portefeuille, courbe_gamma_portefeuille, courbe_valeur_portefeuille):
        assert np.all(np.isfinite(courbe(portefeuille, spots)))


def test_courbe_portefeuille_somme_des_courbes_des_lignes():
    """La courbe de delta du portefeuille est la somme point par point des courbes des lignes."""
    # Même additivité que pour les chiffres ponctuels, mais sur toute la grille.
    portefeuille = portefeuille_de_reference("long call", "short put")
    spots = grille_spots(S_REF)
    attendu = courbe_delta(portefeuille[0], spots) + courbe_delta(portefeuille[1], spots)
    assert np.allclose(courbe_delta_portefeuille(portefeuille, spots), attendu)


def test_zone_instable_unique_autour_du_strike():
    """Pour une option seule, il y a une zone instable et elle contient le strike."""
    # Le gamma d'une option culmine près du strike : c'est là qu'il faut réajuster le plus souvent.
    position = construire_position("long call")
    spots = grille_spots(S_REF)
    zones = zones_instabilite(spots, courbe_gamma(position, spots))
    assert len(zones) == 1
    debut, fin = zones[0]
    assert debut < K_REF < fin


def test_aucune_zone_si_gamma_nul():
    """Un gamma nul partout ne produit aucune zone."""
    # Évite une division par zéro implicite dans le calcul du seuil.
    spots = grille_spots(S_REF)
    assert zones_instabilite(spots, np.zeros_like(spots)) == []


def test_deux_zones_pour_deux_strikes_eloignes():
    """Deux options de strikes éloignés et de maturité courte donnent deux zones séparées."""
    # Avec une échéance proche, chaque gamma forme un pic étroit autour de son strike.
    lignes = construire_lignes("long put", K=80.0, T=0.05) + construire_lignes(
        "long call", K=120.0, T=0.05
    )
    portefeuille = construire_portefeuille(lignes, S_REF, R_REF, SIGMA_REF)
    spots = grille_spots(S_REF)
    zones = zones_instabilite(spots, courbe_gamma_portefeuille(portefeuille, spots))
    assert len(zones) == 2
    assert zones[0][0] < 80.0 < zones[0][1]
    assert zones[1][0] < 120.0 < zones[1][1]


def test_zone_collee_au_bord_de_la_grille():
    """Une zone qui commence au premier point de la grille est correctement ouverte."""
    # Pas de front montant dans ce cas : la fonction doit fermer la zone à la main.
    spots = grille_spots(S_REF)
    gammas = np.linspace(1.0, 0.0, len(spots))
    zones = zones_instabilite(spots, gammas)
    assert len(zones) == 1
    assert zones[0][0] == pytest.approx(spots[0])


def test_texte_zones_instabilite():
    """Le texte décrit les zones en euros, ou signale un gamma nul."""
    # Phrase lue par le trader sous le graphique de gamma.
    assert texte_zones_instabilite([]).startswith("Gamma nul")
    assert texte_zones_instabilite([(92.0, 108.0)]) == (
        "Couverture la plus instable entre 92,00 € et 108,00 €."
    )


def test_strikes_visibles():
    """Seuls les strikes distincts compris dans la plage du graphique sont gardés."""
    # Un strike à 200 € sortirait d'une grille qui va de 60 € à 140 €.
    lignes = construire_lignes("long call", "short call") + construire_lignes("long put", K=200.0)
    portefeuille = construire_portefeuille(lignes, S_REF, R_REF, SIGMA_REF)
    assert strikes_visibles(portefeuille, grille_spots(S_REF)) == [K_REF]


def test_courbe_valeur_portefeuille():
    """La courbe de valeur retrouve, au centre de la grille, la valeur du portefeuille."""
    # Point de contrôle entre la valorisation ponctuelle et la valorisation vectorisée.
    portefeuille = portefeuille_de_reference("long call", "short put")
    spots = grille_spots(S_REF)
    courbe = courbe_valeur_portefeuille(portefeuille, spots)
    assert courbe[NB_POINTS_GRILLE // 2] == pytest.approx(valeur_portefeuille(portefeuille))
    assert np.allclose(courbe, courbe_valeur(portefeuille[0], spots) + courbe_valeur(portefeuille[1], spots))


# ---------------------------------------------------------------------------
# Section couverture delta-gamma
# ---------------------------------------------------------------------------

def instrument_de_reference(type_option="call", K=105.0, T=T_REF):
    """Option de couverture de test, sur le marché de référence."""
    # Par défaut un call de strike 105 et de même maturité que la position de l'énoncé.
    return construire_instrument(type_option, K, T, S_REF, R_REF, SIGMA_REF)


def test_couverture_delta_gamma_de_l_enonce():
    """Les 100 calls de l'énoncé se couvrent en vendant 100 calls 105 puis 14 actions."""
    # 2,777 / 0,02781 ≈ 99,9 options ; il reste 57,02 - 100 × 0,4332 ≈ 13,7 de delta.
    couverture = couverture_delta_gamma(portefeuille_de_reference("long call"), instrument_de_reference())
    assert couverture == {"options": -100, "actions": -14}


def test_couverture_delta_gamma_neutralise_gamma_et_delta():
    """Une fois les deux couvertures en place, il ne reste qu'un gamma et un delta d'arrondi."""
    # Le gamma résiduel ne dépasse pas une demi-option de couverture, le delta pas une demi-action plus ce reste.
    analyse = analyser_couverture_delta_gamma(
        portefeuille_de_reference("long call"), instrument_de_reference(), S_REF, SIGMA_REF, 0.0
    )
    assert abs(analyse["gamma_residuel"]) <= gamma_position(instrument_de_reference()) / 2
    assert abs(analyse["delta_residuel"]) <= 1.0


def test_vente_de_puts_couverte_par_un_achat_d_options():
    """Un portefeuille à gamma négatif se couvre en achetant des options."""
    # Le signe des options de couverture est l'opposé du signe du gamma à neutraliser.
    couverture = couverture_delta_gamma(portefeuille_de_reference("short put"), instrument_de_reference("put", 95.0))
    assert couverture["options"] > 0


def test_couverture_delta_gamma_plus_stable_que_delta_seul():
    """Après un mouvement de 12 €, la couverture delta-gamma bouge bien moins que la couverture delta."""
    # C'est tout l'intérêt de neutraliser aussi le gamma.
    portefeuille, instrument = portefeuille_de_reference("long call"), instrument_de_reference()
    for nouveau_S in (88.0, 112.0):
        analyse = analyser_couverture_delta_gamma(portefeuille, instrument, nouveau_S, SIGMA_REF, 0.0)
        assert abs(analyse["pnl_delta_gamma"]) < abs(analyse["pnl_delta"]) / 5


def test_meme_maturite_neutralise_aussi_le_vega():
    """Couvrir le gamma avec une option de même maturité neutralise presque aussi le vega."""
    # Gamma et vega sont proportionnels à maturité égale : vega = gamma × S² × sigma × T.
    analyse = analyser_couverture_delta_gamma(
        portefeuille_de_reference("long call"), instrument_de_reference(), S_REF, 0.25, 0.0
    )
    assert abs(analyse["pnl_delta_gamma"]) < 5.0
    assert analyse["pnl_delta"] > 100.0


def test_pnl_delta_gamma_nul_sans_changement():
    """Sans changement de marché, les deux couvertures affichent un P&L nul au cours initial."""
    # Point de départ des deux courbes du graphique.
    analyse = analyser_couverture_delta_gamma(
        portefeuille_de_reference("long call"), instrument_de_reference(), S_REF, SIGMA_REF, 0.0
    )
    assert analyse["pnl_delta"] == pytest.approx(0.0, abs=1e-9)
    assert analyse["pnl_delta_gamma"] == pytest.approx(0.0, abs=1e-9)


def test_ligne_de_couverture_et_texte():
    """Un nombre négatif d'options devient une vente, et l'ordre est écrit en clair."""
    # Même convention que pour les lignes saisies : quantité positive, sens dans le type.
    instrument = instrument_de_reference()
    ligne = ligne_de_couverture(instrument, -100)
    assert (ligne.type_position, ligne.quantite) == ("short call", 100.0)
    assert texte_ordre_options(-100, instrument) == (
        "vendre 100 calls de strike 105,00 € et de maturité 0,50 an"
    )
    assert texte_ordre_options(0, instrument) == "aucune option de couverture à traiter"


def test_instrument_sans_gamma_leve_une_erreur():
    """Le calcul refuse explicitement un instrument dont le gamma est quasi nul."""
    # Garde-fou : la validation l'empêche déjà en amont dans l'interface.
    with pytest.raises(ValueError):
        couverture_delta_gamma(portefeuille_de_reference("long call"), instrument_de_reference(K=300.0))


# ---------------------------------------------------------------------------
# Section simulation
# ---------------------------------------------------------------------------

def test_trajectoires_forme_et_point_de_depart():
    """Les trajectoires partent toutes de S0, restent positives et ont la bonne taille."""
    # Une ligne par trajectoire, une colonne par date, la première étant aujourd'hui.
    trajectoires = simuler_trajectoires(S_REF, R_REF, SIGMA_REF, 0.25, 90, nb_trajectoires=500)
    assert trajectoires.shape == (500, 91)
    assert np.all(trajectoires[:, 0] == S_REF)
    assert np.all(trajectoires > 0)


def test_trajectoires_reproductibles():
    """La même graine redonne exactement les mêmes trajectoires."""
    # Indispensable pour que l'écran affiche les mêmes chiffres à chaque visite.
    premier = simuler_trajectoires(S_REF, R_REF, SIGMA_REF, 0.25, 30, nb_trajectoires=100)
    second = simuler_trajectoires(S_REF, R_REF, SIGMA_REF, 0.25, 30, nb_trajectoires=100)
    assert np.array_equal(premier, second)


def test_trajectoires_tendance_risque_neutre():
    """En moyenne, le cours final vaut S0 capitalisé au taux sans risque."""
    # Tolérance d'environ trois écarts-types de la moyenne de 2 000 trajectoires.
    trajectoires = simuler_trajectoires(S_REF, R_REF, SIGMA_REF, 0.5, 100)
    assert trajectoires[:, -1].mean() == pytest.approx(S_REF * np.exp(R_REF * 0.5), abs=1.5)


def test_pnl_moyen_nul_si_volatilites_egales():
    """Réajustée chaque jour, avec la volatilité réalisée égale à l'implicite, la couverture ne gagne rien en moyenne."""
    # C'est la réplication de Black-Scholes : l'écart restant vient seulement des réajustements discrets.
    resultats = simuler_couverture(portefeuille_de_reference("long call"), SIGMA_REF, 90, intervalles=(1,))
    assert abs(resultats[1]["moyenne"]) < 5.0


def test_dispersion_croit_quand_on_reajuste_moins_souvent():
    """Plus les réajustements sont espacés, plus le P&L de couverture est dispersé."""
    # Les trois fréquences sont comparées sur les mêmes trajectoires.
    resultats = simuler_couverture(portefeuille_de_reference("long call"), SIGMA_REF, 90)
    ecarts = [resultats[intervalle]["ecart_type"] for intervalle in INTERVALLES_REAJUSTEMENT]
    assert ecarts == sorted(ecarts)
    assert ecarts[0] < ecarts[-1] / 2


def test_long_gamma_gagne_si_la_volatilite_realisee_depasse_l_implicite():
    """Un acheteur d'options couvert gagne si le marché bouge plus que prévu, un vendeur perd."""
    # Résultat central de la gestion d'options : on achète du gamma quand on pense la volatilité sous-évaluée.
    achat = simuler_couverture(portefeuille_de_reference("long call"), 0.30, 90, intervalles=(1,))
    vente = simuler_couverture(portefeuille_de_reference("short call"), 0.30, 90, intervalles=(1,))
    assert achat[1]["moyenne"] > 0
    assert vente[1]["moyenne"] < 0


def test_pnl_couverture_discrete_sans_reajustement_ni_mouvement():
    """Sur une trajectoire où le cours ne bouge pas, le P&L reflète seulement le temps et le taux."""
    # Cas déterministe construit à la main : aucun aléa, on peut vérifier le signe.
    portefeuille = portefeuille_de_reference("long call")
    trajectoire_plate = np.full((1, 31), S_REF)
    pnl = pnl_couverture_discrete(portefeuille, trajectoire_plate, 30 / 365, 1)
    assert pnl[0] < 0


def test_statistiques_pnl():
    """Les statistiques renvoient la moyenne, l'écart-type et le quantile à 5 %."""
    # Sur une série connue, les trois valeurs se vérifient à la main.
    stats = statistiques_pnl(np.arange(1.0, 101.0))
    assert stats["moyenne"] == pytest.approx(50.5)
    assert stats["quantile_5"] == pytest.approx(5.95)
    assert set(stats) == {"moyenne", "ecart_type", "quantile_5", "erreur_quadratique"}
    # Erreur quadratique : racine de la moyenne des carrés, ici √(3383,5).
    assert stats["erreur_quadratique"] == pytest.approx(np.sqrt(np.mean(np.arange(1.0, 101.0) ** 2)))


def test_cout_nul_ne_change_rien():
    """Avec un coût nul, le P&L est identique à celui de la version sans coût."""
    # Garantit que l'ajout des coûts n'a pas modifié le calcul de base.
    portefeuille = portefeuille_de_reference("long call")
    trajectoires = simuler_trajectoires(S_REF, R_REF, SIGMA_REF, 0.25, 90, nb_trajectoires=200)
    assert np.array_equal(
        pnl_couverture_discrete(portefeuille, trajectoires, 0.25, 7),
        pnl_couverture_discrete(portefeuille, trajectoires, 0.25, 7, 0.0),
    )


def test_couts_plus_eleves_quand_on_reajuste_souvent():
    """Réajuster chaque jour coûte plus cher que réajuster chaque mois."""
    # Plus d'ordres, donc plus de frais, même si chaque ordre est plus petit.
    resultats = simuler_couverture(
        portefeuille_de_reference("long call"), SIGMA_REF, 90, cout_relatif=points_de_base_vers_decimal(10)
    )
    assert resultats[1]["cout_moyen"] > resultats[7]["cout_moyen"] > resultats[30]["cout_moyen"] > 0


def test_couts_creusent_le_pnl_moyen():
    """Le P&L moyen net des coûts est inférieur au P&L sans coûts du montant du coût moyen."""
    # Les deux simulations utilisent les mêmes trajectoires : l'écart est exactement le coût.
    portefeuille = portefeuille_de_reference("long call")
    sans = simuler_couverture(portefeuille, SIGMA_REF, 60, intervalles=(1,))
    avec = simuler_couverture(portefeuille, SIGMA_REF, 60, intervalles=(1,), cout_relatif=0.001)
    assert avec[1]["moyenne"] == pytest.approx(sans[1]["moyenne"] - avec[1]["cout_moyen"])


def test_frequence_optimale_selon_les_couts():
    """Sans coût, réajuster chaque jour est optimal ; avec des coûts élevés, il faut réajuster moins souvent."""
    # C'est l'arbitrage entre risque et coût que la simulation met en évidence.
    portefeuille = portefeuille_de_reference("long call")
    assert frequence_optimale(simuler_couverture(portefeuille, SIGMA_REF, 181)) == 1
    couteux = simuler_couverture(portefeuille, SIGMA_REF, 181, cout_relatif=points_de_base_vers_decimal(50))
    assert frequence_optimale(couteux) != 1


def test_trajectoires_tendance_avec_dividende():
    """Avec dividende, le cours final moyen croît au taux sans risque diminué du dividende."""
    # Ici r = q : le cours reste en moyenne à son niveau de départ.
    trajectoires = simuler_trajectoires(S_REF, R_REF, SIGMA_REF, 0.5, 100, q=R_REF)
    assert trajectoires[:, -1].mean() == pytest.approx(S_REF, abs=1.5)


def test_couverture_equitable_avec_dividende():
    """Avec dividende, une couverture quotidienne reste équitable en moyenne si les volatilités sont égales."""
    # Ce test échouerait si les dividendes touchés ou payés sur les actions de couverture étaient oubliés.
    portefeuille = construire_portefeuille(construire_lignes("long call"), S_REF, R_REF, SIGMA_REF, Q_REF)
    resultats = simuler_couverture(portefeuille, SIGMA_REF, 90, intervalles=(1,))
    assert abs(resultats[1]["moyenne"]) < 5.0


def test_simuler_couverture_refuse_un_horizon_nul():
    """Un horizon de zéro jour est refusé avec une erreur explicite."""
    # Sans pas de temps, la simulation n'a pas de sens.
    with pytest.raises(ValueError):
        simuler_couverture(portefeuille_de_reference("long call"), SIGMA_REF, 0)


def test_simuler_couverture_une_entree_par_frequence():
    """Le résultat contient une entrée par intervalle de réajustement demandé."""
    # Format attendu par le tableau et l'histogramme de l'interface.
    resultats = simuler_couverture(portefeuille_de_reference("long put"), SIGMA_REF, 30)
    assert set(resultats) == set(INTERVALLES_REAJUSTEMENT)
    assert len(resultats[7]["pnl"]) == 2000


def test_simulation_renvoie_le_cours_final_commun():
    """Chaque fréquence reçoit le cours final des mêmes trajectoires, dans le même ordre que ses P&L."""
    # C'est ce qui permet de tracer le P&L de chaque trajectoire en fonction de son cours final.
    resultats = simuler_couverture(portefeuille_de_reference("long call"), SIGMA_REF, 30)
    assert len(resultats[30]["cours_final"]) == len(resultats[30]["pnl"])
    assert np.array_equal(resultats[1]["cours_final"], resultats[30]["cours_final"])
    assert np.all(resultats[1]["cours_final"] > 0)


# ---------------------------------------------------------------------------
# Section analyse de scénario
# ---------------------------------------------------------------------------

def analyse_de_reference(*types_positions, nouveau_S=S_REF, nouvelle_sigma=SIGMA_REF, temps=0.0):
    """Analyse un portefeuille de test dans un scénario donné."""
    # Par défaut, le scénario reproduit exactement la situation initiale.
    return analyser_scenario(
        portefeuille_de_reference(*types_positions), nouveau_S, nouvelle_sigma, temps
    )


def test_analyse_sans_changement():
    """Sans changement de marché, le scénario n'est pas modifié et aucun ajustement n'est dû."""
    # Situation au chargement de l'application.
    analyse = analyse_de_reference("long call")
    assert not analyse["scenario_modifie"]
    assert analyse["ajustement"] == pytest.approx(0.0, abs=1e-12)


def test_analyse_detecte_chaque_type_de_changement():
    """Un changement de cours, de volatilité ou de temps rend le scénario modifié."""
    # C'est ce drapeau qui déclenche l'affichage des courbes de référence en pointillés.
    assert analyse_de_reference("long call", nouveau_S=110.0)["scenario_modifie"]
    assert analyse_de_reference("long call", nouvelle_sigma=0.3)["scenario_modifie"]
    assert analyse_de_reference("long call", temps=0.1)["scenario_modifie"]


def test_point_courant_sur_la_courbe_au_centre_de_la_grille():
    """Au cours initial, les chiffres de l'encart retrouvent la valeur centrale des courbes."""
    # Garantit que le point rouge est bien posé sur la courbe.
    analyse = analyse_de_reference("short put")
    centre = NB_POINTS_GRILLE // 2
    assert analyse["couverture_scenario_courbe"][centre] == pytest.approx(
        analyse["resume_scenario"]["couverture"]
    )
    assert analyse["gamma_scenario_courbe"][centre] == pytest.approx(
        analyse["resume_scenario"]["gamma"]
    )


def test_analyse_ajustement_coherent_avec_les_resumes():
    """L'ajustement est l'écart entre les actions à détenir dans le scénario et celles déjà détenues."""
    # Pour un long call, une hausse du cours impose de vendre des actions supplémentaires.
    analyse = analyse_de_reference("long call", nouveau_S=112.0)
    attendu = (
        analyse["resume_scenario"]["actions_a_detenir"]
        - analyse["resume_initial"]["actions_a_detenir"]
    )
    assert analyse["ajustement"] == attendu
    assert analyse["variation"]["actions_a_detenir"] == attendu
    assert analyse["texte_ajustement"].startswith("vendre")


@pytest.mark.parametrize("nouveau_S", [95.0, 100.5, 101.0, 103.7, 108.2, 115.0])
def test_chiffres_affiches_toujours_coherents(nouveau_S):
    """Les actions affichées avant et après diffèrent exactement de l'ordre affiché."""
    # Garantit qu'on ne verra jamais « −58 » puis « −58 » avec « vendre 1 action » entre les deux.
    analyse = analyse_de_reference("long call", "short put", nouveau_S=nouveau_S)
    en_place = analyse["resume_initial"]["actions_a_detenir"]
    cible = analyse["resume_scenario"]["actions_a_detenir"]
    assert isinstance(analyse["ajustement"], int)
    assert en_place + analyse["ajustement"] == cible


def test_analyse_fournit_zones_et_strikes():
    """L'analyse renvoie au moins une zone instable et le strike à tracer."""
    # Ce sont les éléments ajoutés sur le graphique de gamma.
    analyse = analyse_de_reference("long call")
    assert len(analyse["zones"]) >= 1
    assert analyse["strikes"] == [K_REF]
    assert analyse["texte_zones"].startswith("Couverture la plus instable")


def test_analyse_pnl_nul_sans_changement():
    """Sans changement de marché, les deux P&L sont nuls au cours initial."""
    # Point de départ de la courbe de P&L.
    analyse = analyse_de_reference("long call")
    assert analyse["pnl_non_couvert"] == pytest.approx(0.0, abs=1e-9)
    assert analyse["pnl_couvert"] == pytest.approx(0.0, abs=1e-9)


def test_courbe_pnl_couvert_convexe_pour_un_achat():
    """Couvert, un achat d'options gagne quel que soit le sens du mouvement, une vente perd."""
    # Le delta est neutralisé : il ne reste que le gamma, positif à l'achat, négatif à la vente.
    achat = analyse_de_reference("long call")["pnl_couvert_courbe"]
    vente = analyse_de_reference("short call")["pnl_couvert_courbe"]
    assert np.all(achat >= -1e-9)
    assert np.all(vente <= 1e-9)
