import pandas as pd
import numpy as np
import datetime
import itertools
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
import pickle
from sklearn.metrics import confusion_matrix


def transform_extraction(table_p, max_d, list_client):
    """ Fonction qui transforme les séquences temporelles
    entrées clients en tableau de caractéristiques clients
    à l'horizon temporelle max_day"""
    
    max_day = max_d
    
    # Table des caractéristiques des premiers achats clients
    tab_feat = \
    table_p.groupby(["InvoiceDate", 
                    "CustomerID"]) \
           .agg({"UnitPrice": ["min", "max", "mean"], 
                 "Amount": ["sum", "mean", "min", "max"],
                 "Quantity": ["sum", "mean", "min", "max"], 
                 "Discount": ["sum"], "Manual": ["sum"], 
                 "POST": ["sum"]})
    tab_feat.reset_index(inplace=True)
    tab_feat.drop_duplicates(["CustomerID"], inplace=True)
    tab_feat.set_index(["CustomerID"], inplace=True)
    
    # Table du nombre de produits achetés au premier achat
    tab_hash_product_qty = \
        table_p.groupby(["InvoiceDate", 
                     "CustomerID"])["Description"].unique().reset_index()
    tab_hash_product_qty["Product_nb"]  = \
    tab_hash_product_qty["Description"].apply(lambda x: len(x))
    tab_hash_product_qty.drop(["Description"], inplace=True, axis=1)
    tab_hash_product_qty.drop_duplicates(["CustomerID"], inplace=True)
    tab_hash_product_qty.set_index(["CustomerID"], inplace=True)
    
    # Table du mois du premier achat et de la variable Is_UK
    tab_hash_Month_UK = \
    table_p.drop_duplicates(["CustomerID"])[["Is_UK",
                                             "InvoiceDate",
                                             "CustomerID"]]
    tab_hash_Month_UK["Month"] = \
    tab_hash_Month_UK["InvoiceDate"].apply(lambda x: x.month)
    tab_hash_Month_UK.drop(["InvoiceDate"], inplace=True, axis=1)
    tab_hash_Month_UK.set_index(["CustomerID"], inplace=True)
    
    # Table contenant la feature Latency du modèle 3 du premier achat   
    date_ref = datetime(2011, 12, 9)
    tab_hash_dist = \
        table_p.drop_duplicates(["CustomerID"])[["InvoiceDate",
                                               "CustomerID"]]
    tab_hash_dist["Latency"] = tab_hash_dist["InvoiceDate"]
    tab_hash_dist["Latency"] = \
    tab_hash_dist["Latency"].apply(lambda x: abs((x - date_ref).days))
    tab_hash_dist = tab_hash_dist.set_index(["CustomerID"])
    tab_hash_dist = tab_hash_dist["Latency"] 
    
    # Table contenant les labels
    
    tab_hash_label = table_p.groupby(["CustomerID"]).mean()["Label"]
    
    ##############################################################################
    
    # Table contenant la feature Recency du dernier achat sur la séquence fournie
    temp_tab = table_p[table_p["Dist_first_trans"] < max_day].copy()
    temp_tab.sort_values(["InvoiceDate"], ascending=False, inplace=True)
    tab_hash_eten = temp_tab.drop_duplicates(["CustomerID"])[["InvoiceDate",
                                                              "CustomerID"]]
    tab_hash_eten["Recency"] = tab_hash_eten["InvoiceDate"]
    tab_hash_eten["Recency"] = \
        tab_hash_eten["Recency"].apply(lambda x: abs((x - date_ref).days))
    tab_hash_eten.reset_index(inplace=True)
    tab_hash_eten = tab_hash_eten.set_index(["CustomerID"])
    tab_hash_eten = tab_hash_eten["Recency"]
    
    # Table contenant la feature Periode représentant le temps entre le premier
    # achat et le dernier achat sur la période considérée
    tab_hash_period = pd.DataFrame(index=tab_hash_eten.index)
    tab_hash_period["Periode"] = tab_hash_period.index
    tab_hash_period = \
        tab_hash_period["Periode"].apply(lambda x: 
                                         tab_hash_dist[x] - tab_hash_eten[x]) 
    
    # Table contenant la fréquence d'achat client sur la période considérée
    table_essai_freq = \
    table_p[table_p["Dist_first_trans"] < max_day]
    table_essai_freq = \
        table_essai_freq[["InvoiceDate",
                          "CustomerID"]].drop_duplicates() \
                                        .groupby(["CustomerID"]) \
                                        .count()
    table_essai_freq = table_essai_freq["InvoiceDate"]
    
    # Table contenant le montant des achats sur la période considérée
    temp_tab = table_p[table_p["Dist_first_trans"] < max_day] 
    tab_hash_total_amount = \
        temp_tab.groupby(["CustomerID"]).agg({"Amount":["sum"]})
    tab_hash_total_amount.reset_index(inplace=True)
    tab_hash_total_amount.set_index(["CustomerID"],inplace = True)
    
    # CREATION DE LA TABLE DE TRANSACTION CLIENT
    table_trans = pd.DataFrame(index=list_client)
    
    dict_product_qty = tab_hash_product_qty.to_dict(orient='index')
    dict_month_uk = tab_hash_Month_UK.to_dict(orient='index')
    table_trans["Key"] = list_client
    
    table_trans["Label"] = \
        table_trans["Key"].apply(lambda x: int(tab_hash_label[x]))  
    table_trans["Month"] = \
    table_trans["Key"].apply(lambda x: dict_month_uk[x]["Month"])
    table_trans["Is_UK"] = \
    table_trans["Key"].apply(lambda x: dict_month_uk[x]["Is_UK"])
    table_trans["Product_nb"] = \
    table_trans["Key"].apply(lambda x: dict_product_qty[x]['Product_nb'])
    
    table_trans["UnitPriceMin"] = \
        table_trans["Key"].apply(lambda x: 
                                 tab_feat["UnitPrice", "min"][x])
    table_trans["UnitPriceMax"] = \
        table_trans["Key"].apply(lambda x: 
                                 tab_feat["UnitPrice", "max"][x])
    table_trans["UnitPriceMean"] = \
        table_trans["Key"].apply(lambda x: 
                                 tab_feat["UnitPrice", "mean"][x])
    table_trans["AmountSum"] = \
        table_trans["Key"].apply(lambda x: 
                                 tab_feat["Amount", "sum"][x])
    table_trans["AmountMin"] = \
        table_trans["Key"].apply(lambda x: 
                                 tab_feat["Amount", "min"][x])
    table_trans["AmountMax"] = \
        table_trans["Key"].apply(lambda x: 
                                 tab_feat["Amount", "max"][x])
    table_trans["AmountMean"] = \
        table_trans["Key"].apply(lambda x: 
                                 tab_feat["Amount", "mean"][x])
    table_trans["QuantitySum"] = \
        table_trans["Key"].apply(lambda x: 
                                 tab_feat["Quantity", "sum"][x])
    table_trans["QuantityMin"] = \
        table_trans["Key"].apply(lambda x: 
                                 tab_feat["Quantity", "min"][x])
    table_trans["QuantityMax"] = \
        table_trans["Key"].apply(lambda x: 
                                 tab_feat["Quantity", "max"][x])
    table_trans["QuantityMean"] = \
        table_trans["Key"].apply(lambda x: 
                                 tab_feat["Quantity", "mean"][x])
    table_trans["Discount"] = \
        table_trans["Key"].apply(lambda x: 
                                 tab_feat["Discount", "sum"][x])
    table_trans["Manual"] = \
        table_trans["Key"].apply(lambda x: 
                                 tab_feat["Manual", "sum"][x])
    table_trans["POST"] = \
        table_trans["Key"].apply(lambda x: 
                                 tab_feat["POST", "sum"][x])

    table_trans["Frequency"] = \
        table_trans["Key"].apply(lambda x: table_essai_freq[x])
    table_trans["Periode"] = \
    table_trans["Key"].apply(lambda x: tab_hash_period[x])
    table_trans["TotalAmount"] = \
        table_trans["Key"].apply(lambda x: tab_hash_total_amount["Amount","sum"][x])
    table_trans.drop(["Key"], axis=1, inplace=True)
    
    return table_trans

def standardize_table(table_p, std_scale, list_col):
    """ Fonction qui standarise les entrees """
	
    table_p2 = pd.get_dummies(table_p, columns = ["Month"])
    X_norm = std_scale.transform(table_p2[list_col])
    table_p2[list_col] = X_norm
    return table_p2
	
def extract_clients(tab, list_cl):
    """ Fonction qui renvoie un slice du 
    dataframe initial à partir d'un nombre de clients
    aléatoirement choisi"""
    
    df_sliced = pd.DataFrame()
    for id_c in list_cl:
        df_sliced = df_sliced.append(tab[tab["CustomerID"] == id_c])

    return df_sliced


def plot_confusion_matrix(cm, classes, normalize=False):
	""" Fonction qui affiche la matrice de confusion """
	
	if normalize:
		cm = cm.astype("float") / cm.sum(axis=1)[:,np.newaxis]
	return cm