#Kerakli kutubxonalar
import streamlit as st
from fastai.vision.all import *
import pathlib
from fasttransform.transform import Transform, Pipeline
import pathlib

plt = platform.system()
if plt =="Linux": pathlib.WindowsPath = pathlib.PosixPath
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Sarlavha
st.title("Bu model sizga mevalarni classifikatsiya qilib beradi..!")

# Rasm yuklash
file = st.file_uploader("Rasm yuklang", type=["jpg", "jpeg", "png","webp"])

bt = st.button("Raimni tekshirish !!")

if bt:
    img = PILImage.create(file)
    # model
    model = load_learner("model/food_mix.pkl")
    prediction, _, probs = model.predict(img)

    st.image(img)
    # Natijani ko‘rsatish
    st.success(f"Tasnif: **{prediction}** (Ehtimollik: {probs.max().item():.2%})")

    st.toggle("Qayta sinab ko'rish !")
bt2 = st.button("Sahifani yangilash !")
if bt2:
    st.rerun()
