"""Tests du module options_risk, organisés dans le même ordre que le module testé.

Lancement : pytest -v
"""

import numpy as np
import pytest

from options_risk import (
    AMPLITUDE_GRILLE,
    NB_POINTS_GRILLE,
    Position,
    ajustement_actions,
    ajustement_portefeuille,
    analyser_scenario,
    appliquer_scenario,
    bs_delta,
    bs_gamma,
    bs_greeks,
    bs_price,
    construire_portefeuille,
    courbe_couverture,
    courbe_delta,
    courbe_delta_portefeuille,
    courbe_gamma,
    courbe_gamma_portefeuille,
    couverture_actions,
    decimal_vers_pourcent,
    delta_portefeuille,
    delta_position,
    detail_lignes,
    est_ordre_nul,
    gamma_portefeuille,
    gamma_position,
    grille_spots,
    jours_vers_annees,
    nombre_jours_simulables,
    pourcent_vers_decimal,
    position_au_spot,
    prix_unitaire,
    resume_portefeuille,
    resume_position,
    signe_position,
    strikes_visibles,
    texte_ordre_actions,
    texte_zones_instabilite,
    valeur_portefeuille,
    valeur_position,
    valider_portefeuille,
    valider_saisie,
    variation_resume,
    zones_instabilite,
)

# Jeu de paramètrès de référence, repris de l'énoncé du projet.
S_REF, K_REF, T_REF, R_REF, SIGMA_REF = 100.0, 100.0, 0.5, 0.03, 0.20


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


def test_position_au_spot_ne_modifie_pas_l_originale():
    """Changer le spot crée une nouvelle position et laisse l'ancienne intacte."""
    # La dataclass est figée, ce test protège contre un futur passage à une classe mutable.
    position = construire_position("long call")
    nouvelle = position_au_spot(position, 120.0)
    assert nouvelle.S == 120.0
    assert position.S == S_REF


def test_ajustement_vers_le_haut_pour_un_long_call():
    """Si le cours monte, le long call doit vendre des actions supplémentaires."""
    # Conséquence du gamma positif : le delta augmente, donc la couverture devient plus négative.
    position = construire_position("long call")
    assert ajustement_actions(position, 112.0) < 0


def test_ajustement_nul_si_le_cours_ne_bouge_pas():
    """Aucun ajustement n'est nécessaire si S1 est égal à S0."""
    # Garde-fou simple contre une inversion entre couverture initiale et couverture cible.
    position = construire_position("long call")
    assert ajustement_actions(position, S_REF) == pytest.approx(0.0, abs=1e-12)


def test_resume_position_contient_toutes_les_cles():
    """Le resume alimente l'encart de résultats avec les six informations attendues."""
    # Si une cle disparait, l'interface planterait à l'affichage.
    resume = resume_position(construire_position("long call"))
    attendues = {
        "prix_unitaire",
        "valeur_position",
        "delta_position",
        "gamma_position",
        "couverture_actions",
        "texte_couverture",
    }
    assert attendues == set(resume)


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


def test_est_ordre_nul():
    """Un ordre de moins d'une demi-action est considéré comme nul."""
    # Le seuil est partagé par le texte des ordres et par la flèche du graphique.
    assert est_ordre_nul(0.4)
    assert est_ordre_nul(-0.4)
    assert not est_ordre_nul(0.6)


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


def test_ajustement_portefeuille_coherent_avec_la_position_seule():
    """Pour une seule ligne, l'ajustement du portefeuille égale celui de la position seule."""
    # Les deux chemins de calcul doivent donner le même nombre d'actions à échanger.
    portefeuille = portefeuille_de_reference("long call")
    scenario = tuple(position_au_spot(position, 112.0) for position in portefeuille)
    attendu = ajustement_actions(construire_position("long call"), 112.0)
    assert ajustement_portefeuille(portefeuille, scenario) == pytest.approx(attendu)


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


# ---------------------------------------------------------------------------
# Section validation
# ---------------------------------------------------------------------------

def test_saisie_valide_ne_produit_aucune_erreur():
    """Une saisie correcte renvoie une liste vide."""
    # Cas nominal : la validation ne doit pas bloquer un utilisateur legitime.
    assert valider_saisie("long call", 100, S_REF, K_REF, T_REF, R_REF, SIGMA_REF) == []


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
    # Chaque cas est teste séparément pour identifier immédiatement le champ fautif.
    saisie = {
        "type_position": "long call",
        "quantite": 100,
        "S": S_REF,
        "K": K_REF,
        "T": T_REF,
        "r": R_REF,
        "sigma": SIGMA_REF,
    }
    saisie[champ] = valeur
    assert valider_saisie(**saisie) != []


def test_champ_vide_refuse_sans_planter():
    """Un champ laissé vide produit un message d'erreur, pas une exception."""
    # Exigence explicite du sujet : l'outil ne doit jamais planter sur une saisie incomplète.
    erreurs = valider_saisie("long call", None, S_REF, K_REF, T_REF, R_REF, SIGMA_REF)
    assert len(erreurs) == 1


def test_taux_negatif_accepte():
    """Un taux sans risque négatif est économiquement possible et reste accepté."""
    # Seuls sigma, T, la quantité, S et K sont contraints à être strictement positifs.
    assert valider_saisie("long call", 100, S_REF, K_REF, T_REF, -0.005, SIGMA_REF) == []


def test_erreurs_cumulees():
    """Plusieurs saisies fautives produisent plusieurs messages."""
    # L'interface affiche tout d'un coup plutôt que de corriger erreur après erreur.
    erreurs = valider_saisie("long call", 0, S_REF, K_REF, 0, R_REF, 0)
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
    assert np.allclose(courbe_couverture(position, spots), -courbe_delta(position, spots))


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
    position = construire_position("long call")
    spots = grille_spots(position.S)
    for courbe in (courbe_couverture, courbe_gamma):
        valeurs = courbe(position, spots)
        assert np.all(np.isfinite(valeurs))


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
    """L'ajustement est l'écart entre la couverture du scénario et la couverture en place."""
    # Pour un long call, une hausse du cours impose de vendre des actions supplémentaires.
    analyse = analyse_de_reference("long call", nouveau_S=112.0)
    attendu = analyse["resume_scenario"]["couverture"] - analyse["resume_initial"]["couverture"]
    assert analyse["ajustement"] == pytest.approx(attendu)
    assert analyse["texte_ajustement"].startswith("vendre")


def test_analyse_fournit_zones_et_strikes():
    """L'analyse renvoie au moins une zone instable et le strike à tracer."""
    # Ce sont les éléments ajoutés sur le graphique de gamma.
    analyse = analyse_de_reference("long call")
    assert len(analyse["zones"]) >= 1
    assert analyse["strikes"] == [K_REF]
    assert analyse["texte_zones"].startswith("Couverture la plus instable")