import streamlit as st
import pandas as pd
import joblib

model = joblib.load("final_mushroom_model.pkl")
le = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Mushroom Classifier", page_icon="🍄", layout="centered")
st.title("🍄 Mushroom Edibility Classifier")
st.write("Fill the mushroom characteristics below in plain English.")

st.write("---")

# ---------------------------
# HUMAN READABLE MAPPINGS
# ---------------------------

cap_shape_map = {
    "Bell": "b",
    "Conical": "c",
    "Convex": "x",
    "Flat": "f",
    "Knobbed": "k",
    "Sunken": "s"
}

cap_surface_map = {
    "Fibrous": "f",
    "Grooved": "g",
    "Scaly": "y",
    "Smooth": "s",
}

cap_color_map = {
    "Brown": "n",
    "Buff": "b",
    "Cinnamon": "c",
    "Gray": "g",
    "Green": "r",
    "Pink": "p",
    "Purple": "u",
    "Red": "e",
    "White": "w",
    "Yellow": "y",
    "Black": "k",
    "Orange": "o"
}

bruises_map = {
    "Bruises": "t",
    "No Bruises": "f"
}

gill_attachment_map = {
    "Attached": "a",
    "Free": "f",
    "Notched": "n"
}

gill_spacing_map = {
    "Close": "c",
    "Crowded": "w",
    "Distant": "d"
}

gill_color_map = {
    "Black": "k",
    "Brown": "n",
    "Buff": "b",
    "Chocolate": "h",
    "Gray": "g",
    "Green": "r",
    "Orange": "o",
    "Pink": "p",
    "Purple": "u",
    "Red": "e",
    "White": "w",
    "Yellow": "y"
}

stem_root_map = {
    "Bulbous": "b",
    "Club": "c",
    "Cup": "u",
    "Equal": "e",
    "Rhizomorphs": "z",
    "Rooted": "r"
}

stem_surface_map = {
    "Fibrous": "f",
    "Scaly": "y",
    "Silky": "k",
    "Smooth": "s"
}

stem_color_map = {
    "Brown": "n",
    "Buff": "b",
    "Cinnamon": "c",
    "Gray": "g",
    "Orange": "o",
    "Pink": "p",
    "Red": "e",
    "White": "w",
    "Yellow": "y"
}

veil_type_map = {
    "Universal": "u",
    "Partial": "p"
}

veil_color_map = {
    "Brown": "n",
    "Orange": "o",
    "White": "w",
    "Yellow": "y"
}

has_ring_map = {
    "Yes": "t",
    "No": "f"
}

ring_type_map = {
    "Evanescent": "e",
    "Flaring": "f",
    "Large": "l",
    "None": "n",
    "Pendant": "p",
    "Sheathing": "s",
    "Zone": "z",
    "Grooved": "g"
}

spore_print_color_map = {
    "Black": "k",
    "Brown": "n",
    "Buff": "b",
    "Chocolate": "h",
    "Green": "r",
    "Orange": "o",
    "Purple": "u",
    "White": "w",
    "Yellow": "y"
}

habitat_map = {
    "Woods": "d",
    "Paths": "p",
    "Grasses": "g",
    "Leaves": "l",
    "Urban": "u",
    "Waste": "w",
    "Meadows": "m"
}

season_map = {
    "Spring": "s",
    "Summer": "u",
    "Fall": "a",
    "Winter": "w"
}

# ---------------------------
# UI INPUTS
# ---------------------------

col1, col2 = st.columns(2)

with col1:
    cap_shape = st.selectbox("Cap Shape", list(cap_shape_map.keys()))
    cap_surface = st.selectbox("Cap Surface", list(cap_surface_map.keys()))
    cap_color = st.selectbox("Cap Color", list(cap_color_map.keys()))
    bruises = st.selectbox("Bruises / Bleeding", list(bruises_map.keys()))
    gill_attachment = st.selectbox("Gill Attachment", list(gill_attachment_map.keys()))
    gill_spacing = st.selectbox("Gill Spacing", list(gill_spacing_map.keys()))
    gill_color = st.selectbox("Gill Color", list(gill_color_map.keys()))
    stem_root = st.selectbox("Stem Root", list(stem_root_map.keys()))
    stem_surface = st.selectbox("Stem Surface", list(stem_surface_map.keys()))

with col2:
    stem_height = st.number_input("Stem Height (cm)", min_value=0.0, max_value=50.0, value=5.0)
    stem_width = st.number_input("Stem Width (cm)", min_value=0.0, max_value=50.0, value=5.0)
    stem_color = st.selectbox("Stem Color", list(stem_color_map.keys()))
    veil_type = st.selectbox("Veil Type", list(veil_type_map.keys()))
    veil_color = st.selectbox("Veil Color", list(veil_color_map.keys()))
    has_ring = st.selectbox("Has Ring?", list(has_ring_map.keys()))
    ring_type = st.selectbox("Ring Type", list(ring_type_map.keys()))
    spore_print_color = st.selectbox("Spore Print Color", list(spore_print_color_map.keys()))
    habitat = st.selectbox("Habitat", list(habitat_map.keys()))
    season = st.selectbox("Season", list(season_map.keys()))

cap_diameter = st.number_input("Cap Diameter (cm)", min_value=0.0, max_value=50.0, value=5.0)

# ---------------------------
# CONVERT TO DATASET CODES
# ---------------------------

sample = {
    "cap-diameter": cap_diameter,
    "cap-shape": cap_shape_map[cap_shape],
    "cap-surface": cap_surface_map[cap_surface],
    "cap-color": cap_color_map[cap_color],
    "does-bruise-or-bleed": bruises_map[bruises],
    "gill-attachment": gill_attachment_map[gill_attachment],
    "gill-spacing": gill_spacing_map[gill_spacing],
    "gill-color": gill_color_map[gill_color],
    "stem-height": stem_height,
    "stem-width": stem_width,
    "stem-root": stem_root_map[stem_root],
    "stem-surface": stem_surface_map[stem_surface],
    "stem-color": stem_color_map[stem_color],
    "veil-type": veil_type_map[veil_type],
    "veil-color": veil_color_map[veil_color],
    "has-ring": has_ring_map[has_ring],
    "ring-type": ring_type_map[ring_type],
    "spore-print-color": spore_print_color_map[spore_print_color],
    "habitat": habitat_map[habitat],
    "season": season_map[season]
}

# ---------------------------
# PREDICT
# ---------------------------

if st.button("Predict"):
    df = pd.DataFrame([sample])
    pred = model.predict(df)[0]
    label = le.inverse_transform([pred])[0]

    if label == "e":
        st.success("🍽️ **This mushroom is EDIBLE**")
    else:
        st.error("☠️ **This mushroom is POISONOUS**")
