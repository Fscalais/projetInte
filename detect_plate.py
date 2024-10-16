import cv2
import numpy as np

cascade_path = r"C:\Users\User\OneDrive\Documents\integration\Resources\haarcascade_russian_plate_number.xml"
nplateCascade = cv2.CascadeClassifier(cascade_path) # pylint: disable=no-member

if nplateCascade.empty():
    print(f"Erreur : le fichier Haarcascade n'a pas pu être chargé depuis {cascade_path}.")
    exit()

min_area = 500

img = cv2.imread('voiture.jpg') # pylint: disable=no-member

if img is None:
    print("Erreur : Impossible de lire l'image voiture.jpg.")
    exit()

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # pylint: disable=no-member

numberplates = nplateCascade.detectMultiScale(img_gray, 1.1, 4)

img_roi = None
for (x, y, w, h) in numberplates:
    area = w * h
    if area > min_area:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) # pylint: disable=no-member
        cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2) # pylint: disable=no-member
        img_roi = img[y:y + h, x:x + w]

if img_roi is not None:
    cv2.imwrite("plate.jpg", img_roi) # pylint: disable=no-member
    print("Plaque enregistrée sous 'plate.jpg'.")
else:
    print("Aucune plaque détectée.")

cv2.imshow("Image", img) # pylint: disable=no-member
cv2.waitKey(0) # pylint: disable=no-member
cv2.destroyAllWindows() # pylint: disable=no-member
