"""Interface graphique de l'outil de gestion du risque d'un portefeuille d'options.

Lancement : streamlit run app.py

Ce fichier ne contient aucun calcul financier : il saisit, met en forme et
affiche. Tous les chiffres et toutes les courbes proviennent du module
options_risk, via la fonction analyser_scenario.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from options_risk import (
    AMPLITUDE_GRILLE,
    DECIMALES_ACTIONS,
    TYPES_POSITION,
    analyser_scenario,
    construire_portefeuille,
    decimal_vers_pourcent,
    est_ordre_nul,
    formater_nombre,
    jours_vers_annees,
    nombre_jours_simulables,
    pourcent_vers_decimal,
    valider_portefeuille,
)


# ---------------------------------------------------------------------------
# Constantes d'interface
# ---------------------------------------------------------------------------

# Titre affiché dans l'onglet du navigateur et en haut de la page.
TITRE_APP = "Couverture d'un portefeuille d'options"

# Conventions rappelées à l'écran, comme demandé dans le sujet.
TEXTE_CONVENTION = (
    "Convention : 1 option porte sur 1 action. Modèle de Black-Scholes sans dividende. "
    "Toutes les options portent sur le même sous-jacent."
)

# Contenu du tableau au premier chargement : la position d'exemple de l'énoncé.
LIGNES_PAR_DEFAUT = [{"type_position": "long call", "quantite": 100, "K": 100.0, "T": 0.5}]

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
CLE_EDITEUR = "editeur_portefeuille"
CLE_CURSEUR_SPOT = "curseur_spot"
CLE_CURSEUR_VOLATILITE = "curseur_volatilite"
CLE_CURSEUR_JOURS = "curseur_jours"

# Indicateurs de l'encart de résultats : libellé, clé du résumé, décimales, unité.
INDICATEURS_RESULTATS = (
    ("Valeur du portefeuille", "valeur", 2, " €"),
    ("Delta du portefeuille", "delta", 2, ""),
    ("Gamma du portefeuille", "gamma", 3, ""),
    ("Couverture (actions)", "couverture", DECIMALES_ACTIONS, ""),
)

# Mise en forme des graphiques.
HAUTEUR_GRAPHIQUE = 420
COULEUR_COURBE = "#1f77b4"
COULEUR_POINT = "#d62728"
COULEUR_REFERENCE = "#7f7f7f"
COULEUR_ZONE = "#ff7f0e"
OPACITE_ZONE = 0.18


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

def zone_marche():
    """Affiche dans la barre latérale les paramètres communs à toutes les options."""
    st.sidebar.header("Marché")
    st.sidebar.caption("Paramètres communs à toutes les options du portefeuille.")

    # Aucune borne min_value : c'est la validation qui refuse et explique.
    cours = st.sidebar.number_input("S₀ : cours actuel du sous-jacent (€)", value=100.0, step=1.0)
    taux = st.sidebar.number_input("r : taux sans risque (%)", value=3.0, step=0.25)
    volatilite = st.sidebar.number_input("σ : volatilité annuelle (%)", value=20.0, step=1.0)

    return {"S": cours, "r_pourcent": taux, "sigma_pourcent": volatilite}


def zone_portefeuille():
    """Affiche le tableau éditable des options et renvoie ses lignes sous forme de dictionnaires."""
    st.subheader("Portefeuille d'options")
    st.caption(
        "Une ligne par option. Cliquez sur une case pour la modifier, utilisez la dernière "
        "ligne du tableau pour en ajouter une, et sélectionnez une ligne puis la touche "
        "Suppr pour la retirer."
    )

    # Tableau éditable : chaque colonne a son type, le menu déroulant limite les positions aux quatre cas.
    tableau = st.data_editor(
        pd.DataFrame(LIGNES_PAR_DEFAUT),
        key=CLE_EDITEUR,
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

def curseur_recentre(conteneur, libelle, cle, valeur_reference, borne_basse, borne_haute, pas, format_affichage):
    """Affiche un curseur qui revient à sa valeur de référence dès que ses bornes changent."""
    cle_signature = f"{cle}_signature"
    signature = (valeur_reference, borne_basse, borne_haute)

    # Deux cas imposent de repositionner le curseur avant de l'afficher :
    # la saisie a changé (l'ancienne valeur peut sortir des nouvelles bornes),
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


def zone_curseurs(cours_saisi, volatilite_saisie_pourcent, jours_maximum):
    """Affiche les trois curseurs du scénario et renvoie les valeurs choisies."""
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
    )

    # Curseur de temps : masqué si l'option la plus courte expire dans moins de deux jours.
    if jours_maximum >= 1:
        jours = curseur_recentre(
            colonne_temps, "Temps écoulé (jours)", CLE_CURSEUR_JOURS, 0, 0, jours_maximum, 1, "%d"
        )
    else:
        colonne_temps.caption("Maturité trop courte pour simuler le passage du temps.")
        jours = 0

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
            "Delta": formater_nombre(ligne["delta"]),
            "Gamma": formater_nombre(ligne["gamma"], 3),
        }
        for ligne in detail
    ]


def zone_resultats(analyse):
    """Affiche les chiffres agrégés du scénario, l'ordre de couverture et le détail par ligne."""
    st.subheader("Résultats du scénario")
    resume = analyse["resume_scenario"]
    variation = analyse["variation"] if analyse["scenario_modifie"] else None

    # Quatre indicateurs ; sous chacun, son écart par rapport à la situation initiale si le scénario a bougé.
    colonnes = st.columns(len(INDICATEURS_RESULTATS))
    for colonne, (libelle, cle, decimales, unite) in zip(colonnes, INDICATEURS_RESULTATS):
        ecart = None if variation is None else formater_nombre(variation[cle], decimales) + unite
        colonne.metric(
            libelle,
            formater_nombre(resume[cle], decimales) + unite,
            delta=ecart,
            delta_color="off",
        )

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

def figure_de_base(titre_axe_y):
    """Crée une figure vide avec les axes titrés, la légende et la ligne du zéro."""
    figure = go.Figure()

    # Ligne horizontale à zéro : elle montre d'un coup d'œil le changement de signe.
    figure.add_hline(y=0, line_width=1, line_dash="dot", line_color="grey")

    # Axes titrés avec leurs unités et légende horizontale au-dessus du graphique.
    figure.update_layout(
        xaxis_title="Cours du sous-jacent S (€)",
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
            hovertemplate="S = %{x:.2f} €<br>" + nom + " : %{y:.2f}<extra></extra>",
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
            hovertemplate=nom + "<br>S = %{x:.2f} €<br>%{y:.2f}<extra></extra>",
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
    couverture_en_place = analyse["resume_initial"]["couverture"]
    couverture_cible = analyse["resume_scenario"]["couverture"]

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
    ajouter_niveau_en_place(figure, couverture_en_place)
    ajouter_point(figure, analyse["S0"], couverture_en_place, "Couverture en place", COULEUR_REFERENCE)
    ajouter_point(figure, analyse["S1"], couverture_cible, "Scénario courant", COULEUR_POINT)

    # Flèche d'ajustement, sauf si l'écart est inférieur à une demi-action.
    if not est_ordre_nul(analyse["ajustement"]):
        ajouter_fleche_ajustement(
            figure, analyse["S1"], couverture_en_place, couverture_cible, analyse["texte_ajustement"]
        )

    st.plotly_chart(figure)
    st.caption(
        "La ligne grise est la couverture détenue, la courbe bleue celle qu'il faudrait détenir. "
        "La flèche rouge mesure l'écart : c'est le nombre d'actions à échanger."
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
# Enchaînement des zones
# ---------------------------------------------------------------------------

def main():
    """Enchaîne les zones : saisie, validation, curseurs, résultats, graphiques, ajustement."""
    configurer_page()
    marche = zone_marche()
    lignes = zone_portefeuille()

    # On valide les valeurs brutes avant toute conversion, pour ne jamais calculer sur une saisie fausse.
    erreurs = valider_portefeuille(
        lignes, marche["S"], marche["r_pourcent"], marche["sigma_pourcent"]
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
    )

    # Les curseurs définissent le scénario ; l'option la plus courte borne le temps simulable.
    scenario = zone_curseurs(
        marche["S"], marche["sigma_pourcent"], nombre_jours_simulables(portefeuille_initial)
    )

    # Un seul appel au module de calcul fournit tout ce que les zones suivantes affichent.
    analyse = analyser_scenario(
        portefeuille_initial,
        scenario["S"],
        pourcent_vers_decimal(scenario["sigma_pourcent"]),
        jours_vers_annees(scenario["jours"]),
    )

    zone_resultats(analyse)
    zone_graphique_couverture(analyse)
    zone_graphique_gamma(analyse)
    zone_ajustement(analyse, scenario["jours"])


# Point d'entrée : Streamlit exécute le fichier de haut en bas à chaque interaction.
if __name__ == "__main__":
    main()