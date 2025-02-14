from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
from utils import classes_list

def parse_file(filename, replicas=True):
	with open(filename, 'r') as file:
		lines = file.readlines()

	if replicas:
		lines = lines[::2]
	pairs = []

	for line in lines:
		pair = line.strip().split(',')
		pairs.append([int(x) for x in pair])

	preds, trues = zip(*pairs)

	return np.array(preds), np.array(trues)

if __name__ == "__main__":
	classes = classes_list()

	preds, trues = parse_file('donut_out.txt', False)

	n_correct = 0
	for i, d in enumerate(trues):
		if trues[i] == preds[i]:
			n_correct += 1

	acc = n_correct/len(trues)
	print("Accuracy : ", acc)

	cm = confusion_matrix(trues, preds)

	# Afficher la matrice de confusion
	plt.figure(figsize=(10, 10))
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
	plt.title('Matrice de confusion (Donut)')
	plt.xlabel('Classe prédite')
	plt.ylabel('Classe réelle')
	plt.savefig("donut_confusion_matrice.png")
	plt.show()
