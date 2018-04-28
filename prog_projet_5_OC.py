
""" Importation des librairies standard """
import pandas as pd
import numpy as np
import datetime
from datetime import datetime, date, timedelta
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score, precision_score

""" Importation des librairies personnelles """
import auxiliaire as aux
""" Definition des paramètres """
MAX_DAY = 180

"""Importation des tables de données initiales"""

table_prepa = pd.read_csv("table_preparation.csv", sep=",", 
                          engine="python", encoding='utf-8')
table_prepa.drop("Unnamed: 0", axis=1, inplace=True)
table_prepa["InvoiceDate"] = \
    table_prepa["InvoiceDate"].apply(lambda x: 
                                     datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
table_rfm = pd.read_csv("table_rfm.csv", sep=",", 
                        engine="python", encoding='utf-8')
table_rfm.set_index(["CustomerID"], inplace=True)
table_label = table_rfm["Label"]

"""Importation des fonctions de standardisation
et de prédiction """


infile = open(f"clf_prediction_{MAX_DAY}.pyc",'rb')
clf_vot = pickle.load(infile)
infile.close()
infile = open(f"std_scale_model2_{MAX_DAY}.pyc",'rb')
std_scale_model2 = pickle.load(infile)
infile.close()

""" Génération d'une liste aléatoire de clients"""

max_clients = len(table_prepa["CustomerID"].unique())
nb_clients = np.random.randint(0, max_clients - 1)
table_rfm.reset_index(inplace=True)
list_clients = list(table_rfm["CustomerID"].sample(nb_clients))
list_col = ['Is_UK', 'Product_nb', 'UnitPriceMin', 'UnitPriceMax', 
				   'UnitPriceMean', 'AmountSum', 'AmountMin',
				   'AmountMax', 'AmountMean', 'QuantitySum', 
				   'QuantityMin', 'QuantityMax', 'QuantityMean',
				   'Discount', "Manual", "POST", "Frequency", 
				   "Periode", "TotalAmount"]
len(list_clients)
cm = 0
class_names = [0, 1, 2, 3, 4]

""" definition du coeur du programme """

if __name__ == '__main__':
	print(f"Selection aleatoire de {len(list_clients)} clients")
	table_extracted = aux.extract_clients(table_prepa, 
										  list_clients)
	print(f"Selection des donnees clients et transformation....")
	table_prediction = aux.transform_extraction(table_prepa, 
												MAX_DAY, 
												list(table_label.index))
	print(f"Standardisation des donnees....")
	table_prediction = aux.standardize_table(table_prediction, 
											 std_scale_model2,
											 list_col)
	y = table_prediction["Label"].values
	table_prediction.drop(["Label"], axis=1, inplace=True)
	X = table_prediction.values
	y_pred = clf_vot.predict(X)
	cm = confusion_matrix(y, y_pred)
	float_formatter = lambda x: "%.2f" % x
	np.set_printoptions(formatter={'float_kind':float_formatter})
	print(f"########## Resultats du score de prediction :######")
	print("\n")
	print(f"{round(clf_vot.score(X,y) * 100, 2)} %")
	print("\n")
	print(f"#############  Matrice de confusion : #############")
	print("\n")
	print(f"{aux.plot_confusion_matrix(cm, class_names, normalize=False)}")
	print("\n")
	print(f"#############  Score de rappel par classe : #########")
	print("\n")
	print(f"{recall_score(y, y_pred, labels=class_names, average=None)}")
	print("\n")
	print(f"#############  Score de precision par classe : #########")
	print("\n")
	print(f"{precision_score(y, y_pred, labels=class_names, average=None)}")
	print("\n")
	print(f"#############  Score de F1: ############################")
	print("\n")
	print(f"{f1_score(y, y_pred, labels=class_names, average=None)}")
	print("\n")
	