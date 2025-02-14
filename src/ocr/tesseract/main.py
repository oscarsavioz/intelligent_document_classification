import pytesseract
from pytesseract import Output
from PIL import Image
import os
import time

paths = []

for dossier, sous_dossiers, fichiers in os.walk("../../data/images"):
    for fichier in fichiers:
        if fichier.endswith(".tif"):
            paths.append(os.path.join(dossier, fichier))

removed_chars = "°!@#$%^&*()_+<>?,./:\"‘'{}[]~`‘“"
config = '--psm 3 --oem 3 -c tessedit_create_txt=1 tessedit_char_blacklist={}"'.format(removed_chars)

start = time.time()

for p in paths:
    print(f"Processing {p}")
    img = Image.open(p)
    ratio = 3300 / max(img.size)

    img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)))

    img.save(p + ".png")

    with open(os.path.splitext(p)[0] + "_ocr.txt", 'w', encoding='utf-8') as f:
        f.write(pytesseract.image_to_string(img, output_type=Output.STRING, config=config))

fin = time.time()

print(f"Durée d'exécution pour {len(paths)} images : {fin-start} secondes")
