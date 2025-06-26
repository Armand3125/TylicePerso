import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import io

# =========================================
# Fonctionnalités Réutilisables
# =========================================

def process_image(image, num_clusters):
    """
    Processes the image by resizing and applying KMeans clustering.
    Returns the resized image array, labels, and sorted cluster indices.
    """
    width, height = image.size
    dim = 350  # Réduction à 350 pixels pour la plus grande dimension
    new_width = dim if width > height else int((dim / height) * width)
    new_height = dim if height >= width else int((dim / width) * height)

    resized_image = image.resize((new_width, new_height))
    img_arr = np.array(resized_image)

    pixels = img_arr.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(pixels)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    grayscale_values = np.dot(centers, [0.2989, 0.5870, 0.1140])
    sorted_indices = np.argsort(grayscale_values)  # Trier du plus sombre au plus clair

    return resized_image, img_arr, labels, sorted_indices, new_width, new_height

def recolor_image(img_arr, labels, sorted_indices, palette_colors):
    """
    Recolors the image array based on the provided palette colors.
    """
    recolored_img_arr = np.zeros_like(img_arr)
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            lbl = labels[i * img_arr.shape[1] + j]
            sorted_index = np.where(sorted_indices == lbl)[0][0]
            recolored_img_arr[i, j] = palette_colors[sorted_index]
    recolored_image = Image.fromarray(recolored_img_arr.astype('uint8'))
    return recolored_image

# =========================
# Dictionnaire des couleurs
# =========================
pal = {
    "NC": (0, 0, 0), "BJ": (255, 255, 255),
    "JO": (228, 189, 104), "BC": (0, 134, 214),
    "VL": (174, 150, 212), "VG": (63, 142, 67),
    "RE": (222, 67, 67), "VH": (97, 198, 128),
    "OM": (249, 153, 99), "VGa": (59, 102, 94),
    "BG": (163, 216, 225), "VM": (236, 0, 140),
    "GA": (166, 169, 170), "VB": (94, 67, 183),
    "BF": (4, 47, 86), "MC": (125, 101, 86),
    "BL": (211, 183, 167), "TC": (177, 85, 51),
}

# ====================================
# Listes de palettes fixes pour les Exemples
# ====================================
palettes_examples_4 = [
    ["NC", "TC", "JO", "BJ"],
    ["NC", "BC", "BG", "BJ"],
    ["NC", "RE", "VH", "BJ"],
    ["NC", "VB", "OM", "BJ"],
]

palettes_examples_6 = [
    ["NC", "VB", "RE", "OM", "JO", "BJ"],
    ["NC", "BF", "VB", "BC", "BG", "BJ"],
    ["NC", "VGa", "MC", "GA", "JO", "BJ"],
    ["NC", "MC", "TC", "OM", "BL", "BJ"],
]

# ====================================
# Configuration du titre et du style
# ====================================
st.title("Tylice")

css = """
    <style>
        .stRadio div [data-testid="stMarkdownContainer"] p { display: none; }
        .radio-container { display: flex; flex-direction: column; align-items: center; margin: 10px; }
        .color-container { display: flex; flex-direction: column; align-items: center; margin-top: 5px; }
        .color-box { border: 3px solid black; }
        .stColumn { padding: 0 !important; }
        .first-box { margin-top: 15px; }
        .percentage-container { margin-bottom: 0; }
        .button-container { margin-bottom: 20px; }
    </style>
"""
st.markdown(css, unsafe_allow_html=True)

# =========================================
# Section 1: Téléchargement de l'image
# =========================================
uploaded_image = st.file_uploader("Télécharger une image", type=["jpg", "jpeg", "png", "webp"])

# =========================================
# Sections conditionnelles après upload d'image
# =========================================
if uploaded_image is not None:
    # =========================================
    # Section 2: Sélection du nombre de couleurs
    # =========================================
    col1, col2 = st.columns([2, 5])

    with col1:
        if st.button("4 Couleurs", key="select_4"):
            st.session_state.num_selections = 4
    with col2:
        if st.button("6 Couleurs", key="select_6"):
            st.session_state.num_selections = 6

    # Initialisation du nombre de sélections si non défini
    if "num_selections" not in st.session_state:
        st.session_state.num_selections = 4  # Valeur par défaut

    num_selections = st.session_state.num_selections

    # =========================================
    # Section 3: Personnalisations
    # =========================================
    st.header("Personnalisations")

    rectangle_width = 80 if num_selections == 4 else 50
    rectangle_height = 20
    cols_personalization = st.columns(num_selections * 2)

    image_pers = Image.open(uploaded_image).convert("RGB")
    resized_image_pers, img_arr_pers, labels_pers, sorted_indices_pers, new_width_pers, new_height_pers = process_image(
        image_pers, num_clusters=num_selections
    )

    # Conversion de pixels à centimètres (350px = 14cm, soit 25px/cm)
    px_per_cm = 25
    new_width_cm = round(new_width_pers / px_per_cm, 1)  # Arrondi à 1 décimale (en cm)
    new_height_cm = round(new_height_pers / px_per_cm, 1)  # Arrondi à 1 décimale (en cm)

    if img_arr_pers.shape[-1] == 3:
        pixels_pers = img_arr_pers.reshape(-1, 3)
        kmeans_pers = KMeans(n_clusters=num_selections, random_state=0).fit(pixels_pers)
        labels_pers = kmeans_pers.labels_
        centers_pers = kmeans_pers.cluster_centers_

        centers_rgb_pers = np.array(centers_pers, dtype=int)
        pal_rgb = np.array(list(pal.values()), dtype=int)
        distances_pers = np.linalg.norm(centers_rgb_pers[:, None] - pal_rgb[None, :], axis=2)

        ordered_colors_by_cluster = []
        for i in range(num_selections):
            closest_colors_idx = distances_pers[i].argsort()
            ordered_colors_by_cluster.append([list(pal.keys())[idx] for idx in closest_colors_idx])

        cluster_counts_pers = np.bincount(labels_pers)
        total_pixels_pers = len(labels_pers)
        cluster_percentages_pers = (cluster_counts_pers / total_pixels_pers) * 100

        sorted_indices_pers = np.argsort(-cluster_percentages_pers)
        sorted_percentages_pers = cluster_percentages_pers[sorted_indices_pers]
        sorted_ordered_colors_by_cluster_pers = [ordered_colors_by_cluster[i] for i in sorted_indices_pers]

        selected_colors = []
        selected_color_names = []
        for i, cluster_index in enumerate(sorted_indices_pers):
            with cols_personalization[i * 2]:
                st.markdown("<div class='color-container'>", unsafe_allow_html=True)
                for j, color_name in enumerate(sorted_ordered_colors_by_cluster_pers[i]):
                    color_rgb = pal[color_name]
                    margin_class = "first-box" if j == 0 else ""
                    st.markdown(
                        f"<div class='color-box {margin_class}' style='background-color: rgb{color_rgb}; width: {rectangle_width}px; height: {rectangle_height}px; border-radius: 5px; margin-bottom: 4px;'></div>",
                        unsafe_allow_html=True
                    )
                st.markdown("</div>", unsafe_allow_html=True)

            with cols_personalization[i * 2 + 1]:
                selected_color_name = st.radio(
                    "", sorted_ordered_colors_by_cluster_pers[i],
                    key=f"radio_{i}_pers",
                    label_visibility="hidden"
                )
                selected_colors.append(pal[selected_color_name])
                selected_color_names.append(selected_color_name)

        # Recolorisation de l'image basée sur les sélections de l'utilisateur
        new_img_arr_pers = np.zeros_like(img_arr_pers)
        for i in range(img_arr_pers.shape[0]):
            for j in range(img_arr_pers.shape[1]):
                lbl = labels_pers[i * img_arr_pers.shape[1] + j]
                new_color_index = np.where(sorted_indices_pers == lbl)[0][0]
                new_img_arr_pers[i, j] = selected_colors[new_color_index]

        new_image_pers = Image.fromarray(new_img_arr_pers.astype('uint8'))

        # Affichage de l'image recolorée
        col1_pers, col2_pers, col3_pers = st.columns([1, 6, 1])
        with col2_pers:
            st.image(new_image_pers, use_container_width=True)

        # Affichage des dimensions de l'image
        st.markdown(f"<p style='text-align:center; font-size:16px;'><b>Dimensions : {new_width_cm} cm x {new_height_cm} cm</b></p>", unsafe_allow_html=True)

    # =========================================
    # Section 4: Exemples de Recoloration
    # =========================================
    st.header("Exemples de Recoloration")

    image = Image.open(uploaded_image).convert("RGB")
    resized_image, img_arr, labels, sorted_indices, new_width, new_height = process_image(image, num_clusters=num_selections)

    # Déterminer les palettes et le nombre de clusters
    if num_selections == 4:
        palettes = palettes_examples_4
        num_clusters = 4
    else:
        palettes = palettes_examples_6
        num_clusters = 6

    # Affichage de l'image recolorée pour chaque palette (2 par ligne)
    col_count = 0
    cols_display = st.columns(2)

    for palette in palettes:
        palette_colors = [pal[color] for color in palette]

        recolored_image = recolor_image(img_arr, labels, sorted_indices, palette_colors)

        with cols_display[col_count % 2]:
            st.image(recolored_image, caption=f"Palette: {' - '.join(palette)}", use_container_width=True, width=350)

        col_count += 1
