"""Interface graphique de l'outil de gestion du risque d'un portefeuille d'options.

Lancement : streamlit run app.py

Ce fichier ne contient aucun calcul financier : il saisit, met en forme et
affiche. Tous les chiffres et toutes les courbes proviennent du module
options_risk : analyser_scenario pour le premier onglet,
analyser_couverture_delta_gamma pour le deuxième et simuler_couverture pour
le troisième.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from options_risk import (
    AMPLITUDE_GRILLE,
    DECIMALES_ACTIONS,
    INTERVALLES_REAJUSTEMENT,
    NB_TRAJECTOIRES,
    TYPES_POSITION,
    analyser_couverture_delta_gamma,
    analyser_scenario,
    construire_instrument,
    construire_portefeuille,
    decimal_vers_pourcent,
    est_ordre_nul,
    formater_nombre,
    frequence_optimale,
    jours_vers_annees,
    nombre_jours_simulables,
    points_de_base_vers_decimal,
    pourcent_vers_decimal,
    simuler_couverture,
    valider_instrument,
    valider_portefeuille,
)


# ---------------------------------------------------------------------------
# Constantes d'interface
# ---------------------------------------------------------------------------

# Titre affiché dans l'onglet du navigateur et en haut de la page.
TITRE_APP = "Couverture d'un portefeuille d'options"

# Conventions rappelées à l'écran, comme demandé dans le sujet.
TEXTE_CONVENTION = (
    "Convention : 1 option porte sur 1 action. Modèle de Black-Scholes-Merton, avec un "
    "dividende versé en continu (nul par défaut). Toutes les options portent sur le même sous-jacent."
)

# Portefeuilles d'exemple, pensés pour un cours de 100 € ; le premier est chargé à l'ouverture.
EXEMPLES = {
    "Position de l'énoncé : achat de 100 calls à la monnaie": [
        {"type_position": "long call", "quantite": 100, "K": 100.0, "T": 0.5},
    ],
    "Straddle : achat d'un call et d'un put au même strike": [
        {"type_position": "long call", "quantite": 100, "K": 100.0, "T": 0.5},
        {"type_position": "long put", "quantite": 100, "K": 100.0, "T": 0.5},
    ],
    "Strangle à échéance proche : put 80 et call 120": [
        {"type_position": "long put", "quantite": 100, "K": 80.0, "T": 0.05},
        {"type_position": "long call", "quantite": 100, "K": 120.0, "T": 0.05},
    ],
    "Call spread : achat du call 95, vente du call 105": [
        {"type_position": "long call", "quantite": 100, "K": 95.0, "T": 0.5},
        {"type_position": "short call", "quantite": 100, "K": 105.0, "T": 0.5},
    ],
    "Vente de puts : une position à gamma négatif": [
        {"type_position": "short put", "quantite": 100, "K": 100.0, "T": 0.5},
    ],
    "Position neutralisée : achat et vente du même call": [
        {"type_position": "long call", "quantite": 100, "K": 100.0, "T": 0.5},
        {"type_position": "short call", "quantite": 100, "K": 100.0, "T": 0.5},
    ],
}

# Valeurs proposées dans une nouvelle ligne du tableau ; le strike reste vide pour être choisi.
QUANTITE_PAR_DEFAUT = 100
MATURITE_PAR_DEFAUT = 0.5

# Pas du curseur de cours en fraction du cours saisi : 0,5 % donne 0,50 € pour un spot à 100 €.
PAS_CURSEUR_RELATIF = 0.005

# Bornes et pas du curseur de volatilité, en pourcentage.
VOLATILITE_MIN_CURSEUR = 1.0
VOLATILITE_MAX_CURSEUR = 100.0
PAS_CURSEUR_VOLATILITE = 0.5

# Clés de st.session_state, nommées une seule fois pour éviter les fautes de frappe.
CLE_EXEMPLE = "choix_exemple"
CLE_EDITEUR = "editeur_portefeuille"
CLE_CURSEUR_SPOT = "curseur_spot"
CLE_CURSEUR_VOLATILITE = "curseur_volatilite"
CLE_CURSEUR_JOURS = "curseur_jours"
CLE_VOLATILITE_REALISEE = "curseur_volatilite_realisee"
CLE_HORIZON = "curseur_horizon"
CLE_COUT = "curseur_cout"
CLE_TYPE_INSTRUMENT = "type_instrument"

# Curseurs du scénario remis à zéro par le bouton « Revenir à la situation initiale ».
CLES_CURSEURS_SCENARIO = (CLE_CURSEUR_SPOT, CLE_CURSEUR_VOLATILITE, CLE_CURSEUR_JOURS)

# Instrument de couverture proposé par défaut : un call 5 % au-dessus du cours, à six mois.
RATIO_STRIKE_INSTRUMENT = 1.05
MATURITE_INSTRUMENT_PAR_DEFAUT = 0.5

# Coût de transaction dans la simulation, en points de base du montant échangé.
COUT_MAX_POINTS_DE_BASE = 50

# Indicateurs de l'encart de résultats : libellé avec son unité, clé, décimales, suffixe.
INDICATEURS_COUVERTURE = (
    ("Valeur du portefeuille", "valeur", 2, " €"),
    ("Delta agrégé (actions équivalentes)", "delta", 2, ""),
    ("Gamma agrégé (actions par €)", "gamma", 3, ""),
    ("Actions à détenir", "actions_a_detenir", DECIMALES_ACTIONS, ""),
)

# Seconde ligne : les autres sensibilités, puis le P&L du scénario avec et sans couverture.
INDICATEURS_RISQUE = (
    ("Vega agrégé (par point de vol.)", "vega", 2, " €"),
    ("Theta agrégé (par jour)", "theta", 2, " €"),
    ("P&L sans couverture", "pnl_non_couvert", 2, " €"),
    ("P&L avec couverture", "pnl_couvert", 2, " €"),
)

# Indicateurs de l'onglet delta-gamma : ce qu'il faut traiter, puis ce qu'il reste comme risque.
INDICATEURS_DELTA_GAMMA = (
    ("Options de couverture", "options", 0, ""),
    ("Actions", "actions", 0, ""),
    ("Delta résiduel", "delta_residuel", 2, ""),
    ("Gamma résiduel (actions par €)", "gamma_residuel", 3, ""),
)
INDICATEURS_RESIDUELS = (
    ("Vega résiduel (par point de vol.)", "vega_residuel", 2, " €"),
    ("Theta résiduel (par jour)", "theta_residuel", 2, " €"),
    ("P&L, couverture delta seule", "pnl_delta", 2, " €"),
    ("P&L, couverture delta-gamma", "pnl_delta_gamma", 2, " €"),
)

# Mise en forme des graphiques.
HAUTEUR_GRAPHIQUE = 420
COULEUR_COURBE = "#1f77b4"
COULEUR_POINT = "#d62728"
COULEUR_REFERENCE = "#7f7f7f"
COULEUR_ZONE = "#ff7f0e"
OPACITE_ZONE = 0.18

# Histogrammes de la simulation : une couleur et une opacité par fréquence, la plus étroite au premier plan.
COULEURS_FREQUENCES = {1: "#1f77b4", 7: "#ff7f0e", 30: "#8c8c8c"}
OPACITES_FREQUENCES = {1: 0.8, 7: 0.65, 30: 0.5}
NB_CLASSES_HISTOGRAMME = 80

# Nuage de points de la simulation : taille et transparence des 2 000 points.
TAILLE_POINT_NUAGE = 4
OPACITE_NUAGE = 0.4
CLE_FREQUENCE_NUAGE = "choix_frequence_nuage"


# ---------------------------------------------------------------------------
# Zone : configuration de la page
# ---------------------------------------------------------------------------

def configurer_page():
    """Règle le titre, la largeur de page et affiche l'en-tête."""
    # set_page_config doit être la toute première commande Streamlit appelée.
    st.set_page_config(page_title=TITRE_APP, page_icon="📉", layout="wide")
    st.title(TITRE_APP)
    st.caption(TEXTE_CONVENTION)


# ---------------------------------------------------------------------------
# Zone : saisie (marché et portefeuille)
# ---------------------------------------------------------------------------

def zone_mode_emploi():
    """Affiche un mode d'emploi repliable pour un utilisateur qui découvre l'outil."""
    # Replié par défaut : il ne gêne pas l'utilisateur habitué, et reste visible sur téléphone.
    with st.expander("Mode d'emploi"):
        st.markdown(
            "1. **Marché** : renseignez le cours actuel, le taux, le rendement du dividende "
            "(0 si l'action n'en verse pas) et la volatilité du sous-jacent.\n"
            "2. **Portefeuille** : chargez un exemple ou saisissez vos options, une par ligne.\n"
            "3. **Scénario** : faites varier le cours, la volatilité et le temps avec les curseurs.\n"
            "4. **Lecture** : l'encart indique l'ordre à passer pour rester couvert, les graphiques "
            "montrent comment la couverture évolue et où elle est la plus instable.\n"
            "5. **Simulation** : le second onglet mesure ce que coûte une couverture réajustée "
            "seulement de temps en temps."
        )


def zone_marche():
    """Affiche les paramètres communs à toutes les options, en haut de la page."""
    st.subheader("Marché")
    colonne_cours, colonne_taux, colonne_dividende, colonne_volatilite = st.columns(4)

    # Dans la page plutôt que dans la barre latérale, qui se replie sur téléphone.
    # Aucune borne min_value : c'est la validation qui refuse et explique.
    cours = colonne_cours.number_input("S₀ : cours actuel du sous-jacent (€)", value=100.0, step=1.0)
    taux = colonne_taux.number_input("r : taux sans risque (%)", value=3.0, step=0.25)
    # Nul par défaut : sans dividende, l'outil redonne exactement les chiffres de l'énoncé.
    dividende = colonne_dividende.number_input(
        "q : rendement du dividende (%), 0 si aucun", value=0.0, step=0.5
    )
    volatilite = colonne_volatilite.number_input("σ : volatilité annuelle (%)", value=20.0, step=1.0)

    return {"S": cours, "r_pourcent": taux, "q_pourcent": dividende, "sigma_pourcent": volatilite}


def zone_exemple():
    """Affiche le menu des portefeuilles d'exemple et renvoie le nom de celui choisi."""
    st.subheader("Portefeuille d'options")
    return st.selectbox(
        "Charger un exemple (pensé pour un cours de 100 €)", tuple(EXEMPLES), key=CLE_EXEMPLE
    )


def zone_portefeuille(nom_exemple):
    """Affiche le tableau éditable des options et renvoie ses lignes sous forme de dictionnaires."""
    st.caption(
        "Une ligne par option. Cliquez sur une case pour la modifier, utilisez la dernière "
        "ligne du tableau pour en ajouter une, et sélectionnez une ligne puis la touche "
        "Suppr pour la retirer."
    )

    # La clé dépend de l'exemple : en changer crée un tableau neuf, rempli avec cet exemple.
    tableau = st.data_editor(
        pd.DataFrame(EXEMPLES[nom_exemple]),
        key=f"{CLE_EDITEUR}_{nom_exemple}",
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "type_position": st.column_config.SelectboxColumn(
                "Type de position", options=TYPES_POSITION, default=TYPES_POSITION[0]
            ),
            "quantite": st.column_config.NumberColumn(
                "Quantité (options)", default=QUANTITE_PAR_DEFAUT, step=1, format="%d"
            ),
            "K": st.column_config.NumberColumn("K : strike (€)", step=0.01, format="%.2f"),
            "T": st.column_config.NumberColumn(
                "T : maturité (années)", default=MATURITE_PAR_DEFAUT, step=0.0001, format="%.4f"
            ),
        },
    )

    # Le module de calcul attend une liste de dictionnaires, pas un DataFrame.
    return tableau.to_dict("records")


def zone_erreurs(erreurs):
    """Affiche les messages de saisie invalide dans la zone principale."""
    # Toutes les erreurs sont montrées ensemble plutôt qu'une par une.
    st.error("La saisie est incorrecte, l'outil ne peut pas calculer la couverture :")
    for message in erreurs:
        st.write(f"- {message}")


# ---------------------------------------------------------------------------
# Zone : curseurs du scénario
# ---------------------------------------------------------------------------

def curseur_recentre(conteneur, libelle, cle, valeur_reference, borne_basse, borne_haute, pas, format_affichage, contexte=None):
    """Affiche un curseur qui revient à sa valeur de référence dès que ses bornes ou son contexte changent."""
    cle_signature = f"{cle}_signature"
    signature = (valeur_reference, borne_basse, borne_haute, contexte)

    # Trois cas imposent de repositionner le curseur avant de l'afficher :
    # la saisie a changé (l'ancienne valeur peut sortir des nouvelles bornes),
    # le contexte a changé (un autre exemple a été chargé),
    # ou Streamlit a effacé son état parce qu'il n'était pas affiché au passage précédent.
    if st.session_state.get(cle_signature) != signature or cle not in st.session_state:
        st.session_state[cle_signature] = signature
        st.session_state[cle] = valeur_reference

    return conteneur.slider(
        libelle,
        min_value=borne_basse,
        max_value=borne_haute,
        step=pas,
        format=format_affichage,
        key=cle,
    )


def reinitialiser_curseurs():
    """Oublie la position des curseurs du scénario : ils reviennent à la situation saisie."""
    # Appelée par le bouton, avant la réexécution de la page : curseur_recentre les recentrera.
    for cle in CLES_CURSEURS_SCENARIO:
        st.session_state.pop(f"{cle}_signature", None)


def zone_curseurs(cours_saisi, volatilite_saisie_pourcent, jours_maximum, nom_exemple):
    """Affiche les trois curseurs du scénario et renvoie les valeurs choisies.

    Charger un autre exemple remet les trois curseurs à la situation initiale.
    """
    st.subheader("Scénario de marché")
    colonne_cours, colonne_volatilite, colonne_temps = st.columns(3)

    # Curseur de cours : ±40 % autour du cours saisi, comme la grille des graphiques.
    cours = curseur_recentre(
        colonne_cours,
        "Cours simulé S₁ (€)",
        CLE_CURSEUR_SPOT,
        float(cours_saisi),
        float(cours_saisi * (1 - AMPLITUDE_GRILLE)),
        float(cours_saisi * (1 + AMPLITUDE_GRILLE)),
        max(float(cours_saisi) * PAS_CURSEUR_RELATIF, 0.01),
        "%.2f",
        contexte=nom_exemple,
    )

    # Curseur de volatilité : la plage s'élargit si la volatilité saisie sort des bornes par défaut.
    volatilite = curseur_recentre(
        colonne_volatilite,
        "Volatilité simulée (%)",
        CLE_CURSEUR_VOLATILITE,
        float(volatilite_saisie_pourcent),
        min(VOLATILITE_MIN_CURSEUR, float(volatilite_saisie_pourcent)),
        max(VOLATILITE_MAX_CURSEUR, float(volatilite_saisie_pourcent)),
        PAS_CURSEUR_VOLATILITE,
        "%.1f",
        contexte=nom_exemple,
    )

    # Curseur de temps : masqué si l'option la plus courte expire dans moins de deux jours.
    if jours_maximum >= 1:
        jours = curseur_recentre(
            colonne_temps, "Temps écoulé (jours)", CLE_CURSEUR_JOURS, 0, 0, jours_maximum, 1, "%d",
            contexte=nom_exemple,
        )
    else:
        colonne_temps.caption("Maturité trop courte pour simuler le passage du temps.")
        jours = 0

    # Bouton actif seulement si un curseur a bougé ; le rappel s'exécute avant la réexécution de la page.
    scenario_au_repos = (
        cours == float(cours_saisi) and volatilite == float(volatilite_saisie_pourcent) and jours == 0
    )
    st.button(
        "Revenir à la situation initiale",
        on_click=reinitialiser_curseurs,
        disabled=scenario_au_repos,
    )

    return {"S": cours, "sigma_pourcent": volatilite, "jours": jours}


# ---------------------------------------------------------------------------
# Zone : encart de résultats
# ---------------------------------------------------------------------------

def tableau_detail(detail):
    """Met en forme le détail par ligne pour l'affichage : libellés français et nombres formatés."""
    # Simple mise en forme : tous les chiffres viennent déjà calculés du module options_risk.
    return [
        {
            "Position": ligne["type_position"],
            "Quantité": formater_nombre(ligne["quantite"], 0),
            "K (€)": formater_nombre(ligne["K"]),
            "T restante (années)": formater_nombre(ligne["T"], 4),
            "Prix unitaire (€)": formater_nombre(ligne["prix_unitaire"]),
            "Valeur (€)": formater_nombre(ligne["valeur"]),
            "Delta (actions)": formater_nombre(ligne["delta"]),
            "Gamma (actions par €)": formater_nombre(ligne["gamma"], 3),
            "Vega (€ par point)": formater_nombre(ligne["vega"]),
            "Theta (€ par jour)": formater_nombre(ligne["theta"]),
        }
        for ligne in detail
    ]


def afficher_indicateurs(indicateurs, valeurs, variation):
    """Affiche une ligne d'indicateurs, avec sous chacun son écart à la situation initiale s'il existe."""
    # Les P&L sont déjà des écarts : ils n'ont pas de variation propre à afficher.
    colonnes = st.columns(len(indicateurs))
    for colonne, (libelle, cle, decimales, unite) in zip(colonnes, indicateurs):
        ecart = None
        if variation is not None and cle in variation:
            ecart = formater_nombre(variation[cle], decimales) + unite
        colonne.metric(
            libelle,
            formater_nombre(valeurs[cle], decimales) + unite,
            delta=ecart,
            delta_color="off",
        )


def zone_bandeau_scenario(analyse, nombre_jours):
    """Rappelle au-dessus des résultats si l'on regarde la situation saisie ou un scénario modifié."""
    initial = (
        f"S₀ = {formater_nombre(analyse['S0'])} €, "
        f"σ = {formater_nombre(decimal_vers_pourcent(analyse['sigma0']), 1)} %, aujourd'hui"
    )

    # Sans bandeau, un utilisateur peut prendre les chiffres d'un scénario pour ceux de sa saisie.
    if analyse["scenario_modifie"]:
        scenario = (
            f"S₁ = {formater_nombre(analyse['S1'])} €, "
            f"σ = {formater_nombre(decimal_vers_pourcent(analyse['sigma1']), 1)} %, "
            f"{texte_horizon(nombre_jours)}"
        )
        st.warning(
            f"**Scénario affiché : {scenario}.** Les chiffres ci-dessous ne sont pas ceux de la "
            f"situation saisie ({initial}). Le bouton « Revenir à la situation initiale » ramène "
            "les curseurs au repos."
        )
    else:
        st.info(f"Situation saisie affichée : {initial}.")


def zone_resultats(analyse):
    """Affiche les chiffres agrégés du scénario, l'ordre de couverture et le détail par ligne."""
    st.subheader("Résultats du scénario")
    resume = analyse["resume_scenario"]
    variation = analyse["variation"] if analyse["scenario_modifie"] else None

    # Première ligne : la couverture ; seconde ligne : les autres risques et le P&L du scénario.
    afficher_indicateurs(INDICATEURS_COUVERTURE, resume, variation)
    valeurs_risque = {
        **resume,
        "pnl_non_couvert": analyse["pnl_non_couvert"],
        "pnl_couvert": analyse["pnl_couvert"],
    }
    afficher_indicateurs(INDICATEURS_RISQUE, valeurs_risque, variation)

    # L'ordre à passer est écrit en toutes lettres, comme l'exige le sujet.
    st.success(f"Pour être couvert dans ce scénario : {resume['texte_couverture']}.")
    if variation is not None:
        st.caption("Sous chaque chiffre : l'écart par rapport à la situation initiale.")

    # Détail ligne par ligne, avec le prix unitaire de chaque option.
    st.markdown("**Détail par option**")
    st.dataframe(tableau_detail(analyse["detail_scenario"]), hide_index=True)


# ---------------------------------------------------------------------------
# Zone : graphiques (fonctions de dessin réutilisables)
# ---------------------------------------------------------------------------

def figure_de_base(titre_axe_y, titre_axe_x="Cours du sous-jacent S (€)"):
    """Crée une figure vide avec les axes titrés, la légende et la ligne du zéro."""
    figure = go.Figure()

    # Ligne horizontale à zéro : elle montre d'un coup d'œil le changement de signe.
    figure.add_hline(y=0, line_width=1, line_dash="dot", line_color="grey")

    # Axes titrés avec leurs unités et légende horizontale au-dessus du graphique.
    figure.update_layout(
        xaxis_title=titre_axe_x,
        yaxis_title=titre_axe_y,
        height=HAUTEUR_GRAPHIQUE,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    return figure


def ajouter_courbe(figure, spots, valeurs, nom, couleur, pointillee=False):
    """Ajoute une courbe à la figure, en trait plein ou en pointillés."""
    # Les pointillés servent à la courbe de référence, le trait plein à la courbe du scénario.
    figure.add_trace(
        go.Scatter(
            x=spots,
            y=valeurs,
            mode="lines",
            name=nom,
            line={"color": couleur, "width": 2, "dash": "dash" if pointillee else "solid"},
            hovertemplate="S = %{x:.2f} €<br>" + nom + " : %{y:.4g}<extra></extra>",
        )
    )


def ajouter_point(figure, x, y, nom, couleur):
    """Ajoute un point marqué à la figure, avec son entrée dans la légende."""
    # Sert au point courant du scénario et au point de la couverture en place.
    figure.add_trace(
        go.Scatter(
            x=[x],
            y=[y],
            mode="markers",
            name=nom,
            marker={"color": couleur, "size": 12, "line": {"width": 1, "color": "white"}},
            hovertemplate=nom + "<br>S = %{x:.2f} €<br>%{y:.4g}<extra></extra>",
        )
    )


def ajouter_niveau_en_place(figure, niveau):
    """Trace la ligne horizontale de la couverture actuellement détenue."""
    # Tant qu'on ne réajuste pas, on détient ce nombre d'actions quel que soit le cours.
    figure.add_hline(
        y=niveau,
        line_width=1,
        line_dash="dash",
        line_color=COULEUR_REFERENCE,
        annotation_text="couverture en place",
        annotation_position="bottom left",
    )


def ajouter_fleche_ajustement(figure, x, niveau_depart, niveau_arrivee, texte):
    """Trace une flèche verticale de la couverture en place vers la couverture cible."""
    # La longueur de la flèche est exactement le nombre d'actions à échanger.
    figure.add_annotation(
        x=x,
        y=niveau_arrivee,
        ax=x,
        ay=niveau_depart,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        text=texte,
        showarrow=True,
        arrowhead=3,
        arrowsize=1.2,
        arrowwidth=2,
        arrowcolor=COULEUR_POINT,
        font={"color": COULEUR_POINT},
        bgcolor="rgba(255, 255, 255, 0.85)",
    )


def ajouter_lignes_strikes(figure, strikes):
    """Trace une ligne verticale à chaque strike du portefeuille."""
    # Le gamma culmine près des strikes : ces repères aident à lire le graphique.
    for strike in strikes:
        figure.add_vline(
            x=strike,
            line_width=1,
            line_dash="dot",
            line_color=COULEUR_REFERENCE,
            annotation_text=f"K = {formater_nombre(strike)} €",
            annotation_position="bottom right",
        )


def ajouter_zones_instabilite(figure, zones):
    """Colore en arrière-plan les plages de cours où la couverture est la plus instable."""
    # layer="below" place la bande derrière les courbes pour ne pas les masquer.
    for debut, fin in zones:
        figure.add_vrect(
            x0=debut,
            x1=fin,
            fillcolor=COULEUR_ZONE,
            opacity=OPACITE_ZONE,
            line_width=0,
            layer="below",
            annotation_text="zone instable",
            annotation_position="top left",
        )


# ---------------------------------------------------------------------------
# Zone : graphique de couverture
# ---------------------------------------------------------------------------

def zone_graphique_couverture(analyse):
    """Affiche la couverture en fonction du cours, avec la couverture en place et l'ajustement."""
    st.subheader("Couverture en fonction du cours")
    figure = figure_de_base("Actions à détenir (nombre d'actions)")
    spots = analyse["spots"]

    # Points posés sur la courbe (valeurs exactes) ; ligne et flèche en actions entières, seules négociables.
    point_initial = analyse["resume_initial"]["couverture"]
    point_scenario = analyse["resume_scenario"]["couverture"]
    actions_en_place = analyse["resume_initial"]["actions_a_detenir"]
    actions_cibles = analyse["resume_scenario"]["actions_a_detenir"]

    # Courbe de référence en pointillés, seulement si la volatilité ou le temps l'ont déformée.
    if analyse["scenario_modifie"]:
        ajouter_courbe(
            figure, spots, analyse["couverture_initiale_courbe"],
            "Couverture, situation initiale", COULEUR_REFERENCE, pointillee=True,
        )
    ajouter_courbe(
        figure, spots, analyse["couverture_scenario_courbe"], "Couverture, scénario", COULEUR_COURBE
    )

    # Couverture détenue : ligne horizontale et point gris au cours où elle a été mise en place.
    ajouter_niveau_en_place(figure, actions_en_place)
    ajouter_point(figure, analyse["S0"], point_initial, "Couverture en place", COULEUR_REFERENCE)
    ajouter_point(figure, analyse["S1"], point_scenario, "Scénario courant", COULEUR_POINT)

    # Flèche d'ajustement, sauf si l'écart s'arrondit à zéro action.
    if not est_ordre_nul(analyse["ajustement"]):
        ajouter_fleche_ajustement(
            figure, analyse["S1"], actions_en_place, actions_cibles, analyse["texte_ajustement"]
        )

    st.plotly_chart(figure)
    st.caption(
        "La ligne grise est la couverture détenue, la courbe bleue celle qu'il faudrait détenir. "
        "La flèche rouge mesure l'écart, arrondi à l'action entière : c'est l'ordre à passer."
    )


# ---------------------------------------------------------------------------
# Zone : graphique de gamma
# ---------------------------------------------------------------------------

def zone_graphique_gamma(analyse):
    """Affiche le gamma en fonction du cours, avec les strikes et les zones d'instabilité."""
    st.subheader("Gamma en fonction du cours")
    figure = figure_de_base("Gamma du portefeuille (actions par €)")
    spots = analyse["spots"]

    # Repères d'abord, pour qu'ils restent derrière les courbes.
    ajouter_zones_instabilite(figure, analyse["zones"])
    ajouter_lignes_strikes(figure, analyse["strikes"])

    # Courbe de référence en pointillés, puis courbe du scénario et point courant.
    if analyse["scenario_modifie"]:
        ajouter_courbe(
            figure, spots, analyse["gamma_initial_courbe"],
            "Gamma, situation initiale", COULEUR_REFERENCE, pointillee=True,
        )
    ajouter_courbe(figure, spots, analyse["gamma_scenario_courbe"], "Gamma, scénario", COULEUR_COURBE)
    ajouter_point(
        figure, analyse["S1"], analyse["resume_scenario"]["gamma"], "Scénario courant", COULEUR_POINT
    )

    st.plotly_chart(figure)
    st.warning(analyse["texte_zones"])
    st.caption(
        "Plus la valeur absolue du gamma est élevée, plus la couverture se dégrade vite "
        "quand le cours bouge. La bande orange marque la plage où il dépasse la moitié de son pic."
    )


# ---------------------------------------------------------------------------
# Zone : graphique de P&L
# ---------------------------------------------------------------------------

def zone_graphique_pnl(analyse):
    """Affiche le gain ou la perte du scénario en fonction du cours, avec et sans couverture."""
    st.subheader("P&L du scénario en fonction du cours")
    figure = figure_de_base("Gain ou perte depuis la situation initiale (€)")
    spots = analyse["spots"]

    # Deux courbes : les options seules, puis les options avec la couverture en place non réajustée.
    ajouter_courbe(figure, spots, analyse["pnl_non_couvert_courbe"], "Sans couverture", COULEUR_ZONE)
    ajouter_courbe(figure, spots, analyse["pnl_couvert_courbe"], "Avec couverture", COULEUR_COURBE)
    ajouter_point(figure, analyse["S1"], analyse["pnl_couvert"], "Scénario courant", COULEUR_POINT)

    st.plotly_chart(figure)
    st.caption(
        "Couvert, le portefeuille ne réagit plus à la direction du cours : il ne reste que la "
        "courbure due au gamma et l'érosion due au theta. Un acheteur d'options couvert gagne "
        "sur les grands mouvements et perd un peu chaque jour où le cours ne bouge pas. "
        "Le coût de financement de la couverture est négligé."
    )


# ---------------------------------------------------------------------------
# Zone : ajustement de la couverture
# ---------------------------------------------------------------------------

def texte_horizon(nombre_jours):
    """Décrit le temps écoulé en toutes lettres : aujourd'hui, dans 1 jour, dans 30 jours."""
    # Simple mise en forme, accord du pluriel compris.
    if nombre_jours == 0:
        return "aujourd'hui"
    return f"dans {nombre_jours} jour" + ("s" if nombre_jours > 1 else "")


def zone_ajustement(analyse, nombre_jours):
    """Affiche en toutes lettres le passage de la situation initiale au scénario et l'ordre à passer."""
    st.subheader("Ajustement de la couverture")

    # Volatilités affichées en pourcentage pour rester dans les unités de la saisie.
    volatilite_initiale = formater_nombre(decimal_vers_pourcent(analyse["sigma0"]), 1)
    volatilite_scenario = formater_nombre(decimal_vers_pourcent(analyse["sigma1"]), 1)

    st.info(
        f"Situation initiale : S = {formater_nombre(analyse['S0'])} €, "
        f"σ = {volatilite_initiale} %, aujourd'hui.  \n"
        f"Scénario : S = {formater_nombre(analyse['S1'])} €, "
        f"σ = {volatilite_scenario} %, {texte_horizon(nombre_jours)}.  \n"
        f"Pour rester couvert : **{analyse['texte_ajustement']}**."
    )


# ---------------------------------------------------------------------------
# Zone : couverture delta-gamma
# ---------------------------------------------------------------------------

def zone_instrument(cours_saisi):
    """Affiche le choix de l'option de couverture et renvoie ses caractéristiques."""
    st.subheader("Couverture delta-gamma")
    st.caption(
        "Une action a un gamma nul : elle ne peut couvrir que le delta. Pour neutraliser aussi le "
        "gamma, il faut une option. L'outil calcule combien en traiter pour annuler le gamma du "
        "portefeuille, puis combien d'actions pour annuler le delta qui reste. Le scénario est "
        "celui des curseurs du premier onglet."
    )
    colonne_type, colonne_strike, colonne_maturite = st.columns(3)

    # Le strike proposé suit le cours saisi ; aucune borne n'est imposée, la validation explique.
    type_option = colonne_type.selectbox(
        "Option de couverture", ("call", "put"), key=CLE_TYPE_INSTRUMENT
    )
    strike = colonne_strike.number_input(
        "Strike de l'option de couverture (€)",
        value=round(float(cours_saisi) * RATIO_STRIKE_INSTRUMENT, 2),
        step=1.0,
    )
    maturite = colonne_maturite.number_input(
        "Maturité de l'option de couverture (années)",
        value=MATURITE_INSTRUMENT_PAR_DEFAUT,
        step=0.25,
        format="%.4f",
    )
    return {"type_option": type_option, "K": strike, "T": maturite}


def figure_pnl_delta_gamma(analyse_dg):
    """Compare le P&L de la couverture delta seule et de la couverture delta-gamma selon le cours."""
    figure = figure_de_base("Gain ou perte depuis la situation initiale (€)")
    spots = analyse_dg["spots"]

    # Deux stratégies posées dans la situation initiale puis conservées sans réajustement.
    ajouter_courbe(figure, spots, analyse_dg["pnl_delta_courbe"], "Couverture delta seule", COULEUR_ZONE)
    ajouter_courbe(figure, spots, analyse_dg["pnl_delta_gamma_courbe"], "Couverture delta-gamma", COULEUR_COURBE)
    ajouter_point(figure, analyse_dg["S1"], analyse_dg["pnl_delta"], "Scénario, delta seule", COULEUR_ZONE)
    ajouter_point(figure, analyse_dg["S1"], analyse_dg["pnl_delta_gamma"], "Scénario, delta-gamma", COULEUR_POINT)
    return figure


def zone_delta_gamma(analyse_dg):
    """Affiche les ordres de la couverture delta-gamma, les risques résiduels et la comparaison de P&L."""
    st.success(
        f"Pour couvrir en delta et en gamma : {analyse_dg['texte_options']}, "
        f"puis {analyse_dg['texte_actions']}."
    )
    afficher_indicateurs(INDICATEURS_DELTA_GAMMA, analyse_dg, None)
    afficher_indicateurs(INDICATEURS_RESIDUELS, analyse_dg, None)

    st.plotly_chart(figure_pnl_delta_gamma(analyse_dg))
    st.caption(
        "Couvert en delta seul, le portefeuille garde sa courbure : il gagne ou perd dès que le "
        "cours s'éloigne. Couvert aussi en gamma, sa courbe s'aplatit autour du cours initial. "
        "L'option de couverture apporte cependant son propre vega et son propre theta : avec une "
        "maturité égale à celle du portefeuille, ils se compensent presque aussi ; avec une "
        "maturité différente, un risque de volatilité subsiste."
    )


def zone_onglet_delta_gamma(portefeuille_initial, cours_saisi, scenario):
    """Enchaîne l'onglet delta-gamma : choix de l'instrument, validation, puis résultats."""
    instrument = zone_instrument(cours_saisi)

    # Même scénario que le premier onglet ; une saisie invalide n'arrête que cet onglet.
    marche_commun = portefeuille_initial[0]
    temps_ecoule = jours_vers_annees(scenario["jours"])
    erreurs = valider_instrument(
        instrument["type_option"], instrument["K"], instrument["T"],
        cours_saisi, marche_commun.r, marche_commun.sigma, temps_ecoule,
        marche_commun.rendement_dividende,
    )
    if erreurs:
        zone_erreurs(erreurs)
        return

    option_de_couverture = construire_instrument(
        instrument["type_option"], instrument["K"], instrument["T"],
        cours_saisi, marche_commun.r, marche_commun.sigma, marche_commun.rendement_dividende,
    )
    zone_delta_gamma(
        analyser_couverture_delta_gamma(
            portefeuille_initial,
            option_de_couverture,
            scenario["S"],
            pourcent_vers_decimal(scenario["sigma_pourcent"]),
            temps_ecoule,
        )
    )


# ---------------------------------------------------------------------------
# Zone : simulation de couverture discrète
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Simulation des trajectoires en cours…")
def simulation_en_cache(portefeuille, volatilite_realisee, horizon_jours, cout_relatif):
    """Lance la simulation, ou renvoie le résultat déjà calculé pour les mêmes paramètres."""
    # Streamlit réexécute la page à chaque clic : le cache évite de relancer 2 000 trajectoires pour rien.
    return simuler_couverture(
        portefeuille, volatilite_realisee, horizon_jours, cout_relatif=cout_relatif
    )


def texte_intervalle(nombre_jours):
    """Décrit une fréquence de réajustement en toutes lettres : tous les jours, tous les 7 jours."""
    return "Réajustée tous les jours" if nombre_jours == 1 else f"Réajustée tous les {nombre_jours} jours"


def tableau_statistiques(resultats):
    """Met en forme les statistiques de chaque fréquence de réajustement pour l'affichage."""
    return [
        {
            "Couverture": texte_intervalle(intervalle),
            "P&L moyen (€)": formater_nombre(resultats[intervalle]["moyenne"]),
            "Écart-type (€)": formater_nombre(resultats[intervalle]["ecart_type"]),
            "Pire 5 % (€)": formater_nombre(resultats[intervalle]["quantile_5"]),
            "Coût moyen (€)": formater_nombre(resultats[intervalle]["cout_moyen"]),
            "Erreur quadratique (€)": formater_nombre(resultats[intervalle]["erreur_quadratique"]),
        }
        for intervalle in INTERVALLES_REAJUSTEMENT
    ]


def classes_communes(resultats):
    """Bornes et largeur de classes partagées par tous les histogrammes de la simulation."""
    # Mise en forme seulement : sans classes communes, chaque histogramme découperait sa propre étendue
    # et les hauteurs ne seraient pas comparables d'une couleur à l'autre.
    plus_bas = min(float(resultats[intervalle]["pnl"].min()) for intervalle in resultats)
    plus_haut = max(float(resultats[intervalle]["pnl"].max()) for intervalle in resultats)
    largeur = (plus_haut - plus_bas) / NB_CLASSES_HISTOGRAMME
    return {"start": plus_bas, "end": plus_haut + largeur, "size": largeur}


def figure_histogramme_pnl(resultats):
    """Superpose les histogrammes du P&L simulé, une couleur par fréquence de réajustement."""
    figure = go.Figure()
    classes = classes_communes(resultats)

    # La distribution la plus large est dessinée en premier, la plus étroite par-dessus, pour rester visible.
    for intervalle in sorted(INTERVALLES_REAJUSTEMENT, reverse=True):
        figure.add_trace(
            go.Histogram(
                x=resultats[intervalle]["pnl"],
                name=texte_intervalle(intervalle),
                xbins=classes,
                marker_color=COULEURS_FREQUENCES[intervalle],
                opacity=OPACITES_FREQUENCES[intervalle],
            )
        )

    # Ligne verticale à zéro : à gauche, la couverture a perdu de l'argent ; à droite, elle en a gagné.
    figure.add_vline(x=0, line_width=1, line_dash="dot", line_color="grey")
    # Légende dans l'ordre de lecture naturel (tous les jours d'abord), malgré l'ordre de dessin inversé.
    figure.update_layout(
        barmode="overlay",
        xaxis_title="P&L de la position couverte à l'horizon (€)",
        yaxis_title="Nombre de trajectoires",
        height=HAUTEUR_GRAPHIQUE,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "traceorder": "reversed"},
    )
    return figure


def figure_nuage_pnl(resultat, strikes, nom_frequence):
    """Nuage du P&L de chaque trajectoire en fonction de son cours final, avec les strikes en repère."""
    figure = figure_de_base(
        "P&L de la position couverte à l'horizon (€)", "Cours du sous-jacent à l'horizon (€)"
    )

    # Un point par trajectoire, transparents pour que les zones denses ressortent.
    figure.add_trace(
        go.Scatter(
            x=resultat["cours_final"],
            y=resultat["pnl"],
            mode="markers",
            name=nom_frequence,
            marker={"color": COULEUR_COURBE, "size": TAILLE_POINT_NUAGE, "opacity": OPACITE_NUAGE},
            hovertemplate="Cours final = %{x:.2f} €<br>P&L = %{y:.2f} €<extra></extra>",
        )
    )
    ajouter_lignes_strikes(figure, strikes)
    return figure


def zone_nuage_pnl(resultats, portefeuille):
    """Affiche le nuage P&L / cours final pour la fréquence de réajustement choisie."""
    st.markdown("**D'où vient la forme de la distribution**")

    # Par défaut la fréquence la plus espacée, où la forme en V est la plus nette.
    intervalle = st.radio(
        "Fréquence de réajustement affichée",
        INTERVALLES_REAJUSTEMENT,
        index=len(INTERVALLES_REAJUSTEMENT) - 1,
        format_func=texte_intervalle,
        horizontal=True,
        key=CLE_FREQUENCE_NUAGE,
    )

    # Strikes distincts du portefeuille, pour repérer le creux du V.
    strikes = sorted({position.K for position in portefeuille})
    st.plotly_chart(figure_nuage_pnl(resultats[intervalle], strikes, texte_intervalle(intervalle)))
    st.caption(
        "Chaque point est une trajectoire. Pour un acheteur d'options, celles qui finissent loin du "
        "strike ont fait un grand mouvement, que le gamma transforme en gain. Celles qui finissent "
        "près du strike ont peu bougé : la couverture y a payé le theta sans rien gagner en échange, "
        "et c'est là que se trouvent les pires résultats. Pour un vendeur, la figure est inversée."
    )


def zone_simulation(portefeuille, volatilite_saisie_pourcent, jours_maximum):
    """Affiche l'onglet de simulation : paramètres, statistiques et histogramme du P&L de couverture."""
    st.subheader("Simulation de couverture discrète")
    st.caption(
        f"{NB_TRAJECTOIRES} trajectoires de cours sont simulées. Sur chacune, le portefeuille est "
        "couvert en delta, puis réajusté à intervalles réguliers. Avec une couverture continue et "
        "une volatilité réalisée égale à la volatilité saisie, le P&L serait nul : l'écart mesure "
        "le coût de ne réajuster que de temps en temps."
    )

    # Il faut au moins un jour de simulation sans faire expirer d'option.
    if jours_maximum < 1:
        st.info("La maturité la plus courte est trop proche pour simuler la couverture.")
        return

    colonne_volatilite, colonne_horizon, colonne_cout = st.columns(3)
    volatilite_realisee = curseur_recentre(
        colonne_volatilite,
        "Volatilité réalisée par le marché (%)",
        CLE_VOLATILITE_REALISEE,
        float(volatilite_saisie_pourcent),
        min(VOLATILITE_MIN_CURSEUR, float(volatilite_saisie_pourcent)),
        max(VOLATILITE_MAX_CURSEUR, float(volatilite_saisie_pourcent)),
        PAS_CURSEUR_VOLATILITE,
        "%.1f",
    )

    # Un curseur exige deux bornes distinctes : avec un seul jour possible, l'horizon est fixé.
    if jours_maximum >= 2:
        horizon = curseur_recentre(
            colonne_horizon, "Horizon (jours)", CLE_HORIZON, jours_maximum, 1, jours_maximum, 1, "%d"
        )
    else:
        colonne_horizon.caption("Horizon fixé à 1 jour.")
        horizon = 1

    # Coût de chaque ordre sur actions, payé à la mise en place et à chaque réajustement.
    cout_points_de_base = colonne_cout.slider(
        "Coût de transaction (points de base)", 0, COUT_MAX_POINTS_DE_BASE, 0, 1, key=CLE_COUT
    )

    resultats = simulation_en_cache(
        portefeuille,
        pourcent_vers_decimal(volatilite_realisee),
        horizon,
        points_de_base_vers_decimal(cout_points_de_base),
    )

    st.dataframe(tableau_statistiques(resultats), hide_index=True)
    st.success(
        f"Fréquence la plus efficace avec ces paramètres : "
        f"{texte_intervalle(frequence_optimale(resultats)).lower()}."
    )
    st.caption(
        "L'erreur quadratique combine le coût moyen, qui creuse le P&L, et la dispersion : c'est le "
        "critère retenu pour désigner la meilleure fréquence. Montez le coût de transaction pour "
        "voir le réajustement quotidien perdre sa place."
    )
    st.plotly_chart(figure_histogramme_pnl(resultats))
    st.caption(
        "Plus on réajuste rarement, plus le résultat est dispersé. Les trois histogrammes partagent "
        "les mêmes classes, donc leurs hauteurs se comparent directement. Placez la volatilité "
        "réalisée au-dessus de la volatilité saisie : un acheteur d'options couvert gagne en "
        "moyenne, un vendeur perd. C'est le pari d'un trader qui achète ou vend du gamma."
    )
    zone_nuage_pnl(resultats, portefeuille)


# ---------------------------------------------------------------------------
# Enchaînement des zones
# ---------------------------------------------------------------------------

def main():
    """Enchaîne les zones : saisie, validation, puis les deux onglets de l'outil."""
    configurer_page()
    zone_mode_emploi()
    marche = zone_marche()
    nom_exemple = zone_exemple()
    lignes = zone_portefeuille(nom_exemple)

    # On valide les valeurs brutes avant toute conversion, pour ne jamais calculer sur une saisie fausse.
    erreurs = valider_portefeuille(
        lignes, marche["S"], marche["r_pourcent"], marche["sigma_pourcent"], marche["q_pourcent"]
    )
    if erreurs:
        zone_erreurs(erreurs)
        st.stop()

    # Les pourcentages saisis sont convertis en taux décimaux attendus par les formules.
    portefeuille_initial = construire_portefeuille(
        lignes,
        marche["S"],
        pourcent_vers_decimal(marche["r_pourcent"]),
        pourcent_vers_decimal(marche["sigma_pourcent"]),
        pourcent_vers_decimal(marche["q_pourcent"]),
    )

    # L'option la plus courte borne le temps que l'on peut simuler, dans les deux onglets.
    jours_maximum = nombre_jours_simulables(portefeuille_initial)
    onglet_couverture, onglet_delta_gamma, onglet_simulation = st.tabs(
        ["Couverture du scénario", "Couverture delta-gamma", "Simulation de couverture discrète"]
    )

    with onglet_couverture:
        # Les curseurs définissent le scénario ; un seul appel au module de calcul fournit tout le reste.
        scenario = zone_curseurs(marche["S"], marche["sigma_pourcent"], jours_maximum, nom_exemple)
        analyse = analyser_scenario(
            portefeuille_initial,
            scenario["S"],
            pourcent_vers_decimal(scenario["sigma_pourcent"]),
            jours_vers_annees(scenario["jours"]),
        )
        zone_bandeau_scenario(analyse, scenario["jours"])
        zone_resultats(analyse)
        zone_graphique_couverture(analyse)
        zone_ajustement(analyse, scenario["jours"])
        zone_graphique_gamma(analyse)
        zone_graphique_pnl(analyse)

    with onglet_delta_gamma:
        zone_onglet_delta_gamma(portefeuille_initial, marche["S"], scenario)

    with onglet_simulation:
        zone_simulation(portefeuille_initial, marche["sigma_pourcent"], jours_maximum)


# Point d'entrée : Streamlit exécute le fichier de haut en bas à chaque interaction.
if __name__ == "__main__":
    main()
