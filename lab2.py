import requests
import streamlit as st
import io
from PIL import Image, ImageDraw
import random

# Funkcija siųsti užklausą į SentiSight.ai API objektų aptikimui
def predict_object_detection(image):
    token = "p3ndr41l10hk5l2esv0cajfohk"  # Pritaikymo raktas, reikalingas autentifikacijai
    project_id = "60738"  # Projektas, kuriam norima atlikti objektų aptikimą
    model = "object-detection-model-1"  # Modelis, skirtas objektų aptikimui
    
    headers = {"X-Auth-token": token, "Content-Type": "application/octet-stream"}  # Nustatomos HTTP antraštės

    image_bytes = image.read()  # Paveikslėlio baitai perskaitomi į atmintį

    r = requests.post(f'https://platform.sentisight.ai/api/predict/{project_id}/{model}/last/', headers=headers, data=image_bytes)  # Siunčiama užklausa į SentiSight.ai API

    return r  # Grąžinamas užklausos atsakymas

# Funkcija, skirta nubrėžti ribines dėžutes ant paveikslėlio
def draw_boxes_on_image(image_bytes, boxes):
    try:
        img = Image.open(io.BytesIO(image_bytes))  # Atidaromas paveikslėlis naudojant PIL biblioteką
        draw = ImageDraw.Draw(img)  # Sukuriamas paveikslėlio objektas

        # Priskiriama atsitiktinė spalva kiekvienai klasės žymei
        class_colors = {}
        for box in boxes:
            class_name = box['label']
            if class_name not in class_colors:
                class_colors[class_name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        for box in boxes:
            left = box['x0']
            top = box['y0']
            right = box['x1']
            bottom = box['y1']
            class_name = box['label'] # gautas pavadinimas
            score = box['score'] #gauta tikimybė
            color = class_colors[class_name] #nustatoma spalva pagal klasės žymę
            draw.rectangle([left, top, right, bottom], outline=color, width=3)  # Brėžiama ribinė dėžė
            draw.text((left+10, top), f"{class_name}: {score:.2f}", fill=color)  # Pridedami tekstiniai žymėjimai
        return img  # Grąžinamas paveikslėlis su nubrėžtomis ribinėmis dėžėmis
    except Exception as e:
        st.error(f'Error opening image: {e}')  # Klaida, jei nepavyksta atverti paveikslėlio
        return None

# Streamlit programos pradžia
def main():
    st.title("SentiSight.ai Object Detection")  # Antraštė

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg"])  # Įkeliamas paveikslėlis

    if uploaded_image is not None:
        response = predict_object_detection(uploaded_image)  # Siunčiama užklausa į objektų aptikimo API

        if response.status_code == 200:  # Jei užklausa sėkminga
            response_data = response.json()  # Gaunama atsakymo informacija
            # surenka visus duomenis ir objektus į vieną sąrašą
            detected_objects = []
            for obj in response_data:
                detected_objects.append({
                    'label': obj['label'],
                    'score': obj['score'],
                    'x0': obj['x0'],
                    'y0': obj['y0'],
                    'x1': obj['x1'],
                    'y1': obj['y1']
                })  

            if detected_objects:  # Jei yra aptiktų objektų
                st.write("Objects detected")  # Parodomas pranešimas apie aptiktus objektus
                image_with_boxes = draw_boxes_on_image(uploaded_image.getvalue(), detected_objects)  # Paveikslėlis su aptiktų objektų ribinėmis dėžėmis
                if image_with_boxes:
                    st.image(image_with_boxes, caption='Image with detected objects', use_column_width=True)  # Paveikslėlis su aptiktų objektų ribinėmis dėžėmis
                else:
                    st.warning("Failed to draw bounding boxes on the image.")  # Klaida, jei nepavyko nubrėžti ribinių dėžių ant paveikslėlio
            else:
                st.warning("No objects detected in the image.")  # Pranešimas, jei paveikslėlyje nėra aptiktų objektų
        else:
            st.error('Error performing prediction. Status code:', response.status_code)  # Klaida, jei nepavyko atlikti prognozės
            st.error('Error message:', response.text)  # Klaidos pranešimas

            
if __name__ == '__main__':
    main()