import easyocr
from PIL import Image
import os
import time

paths = []
for dossier, sous_dossiers, fichiers in os.walk("../../data/images"):
    for fichier in fichiers:
        if fichier.endswith(".tif"):
            paths.append(os.path.join(dossier, fichier))

reader = easyocr.Reader(['en'], gpu=False)

start = time.time()

for p in [paths[1]]:
    print(f"Processing {p}")
    img = Image.open(p).convert('RGB')
    ratio = 3300 / max(img.size)

    img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)))

    with open(os.path.splitext(p)[0] + "_ocr_easy.txt", 'w', encoding='utf-8') as f:
        result = reader.readtext(p, detail=0, workers=2)
        for r in result:
            f.write(r + " ")

fin = time.time()

print(f"Durée d'exécution pour {len(paths)} images : {fin-start} secondes")
