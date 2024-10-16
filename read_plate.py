import cv2
import easyocr

img = cv2.imread('plate.jpg') # pylint: disable=no-member

if img is None:
    print("Erreur : Impossible de lire l'image de la plaque.")
    exit()

reader = easyocr.Reader(['en'])
result = reader.readtext(img)

if result:
    for (bbox, text, prob) in result:
        print(f"Plaque détectée : {text.strip()} (Précision : {prob:.2f})")
else:
    print("Aucune plaque détectée.")
