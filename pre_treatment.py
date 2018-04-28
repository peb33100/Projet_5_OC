import pandas as pd
import numpy as np
import datetime
from datetime import datetime, date, timedelta
import pickle


from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix

def fill_label(x, table_rfm):
    """ Fonction permettant d'attribuer une classe à 
    un client en fonction du traitement effectué 
    sur la table_rfm """
    try:
        return table_rfm["Label"][x]
    except KeyError :
        return np.nan  
    
    
def create_feature_occ_dist(table_travail):
    """ Fonction qui complète la table_travail
    avec l'occurence de chaque transaction et
    la distance en jours à la premier transaction"""
    
    tab_hash_freq = table_travail[["InvoiceDate", 
                               "CustomerID"]].drop_duplicates() \
                                             .groupby(["CustomerID"]) \
                                             .count()
    tab_hash_freq["Compteur"] = 0
    tab_compteur = tab_hash_freq["Compteur"].copy()
    tab_distance = tab_hash_freq["Compteur"].copy()
    tab_hash_compteur = table_travail.groupby(["InvoiceDate", 
                                               "CustomerID"]).mean()
    tab_hash_compteur.drop(["UnitPrice", "Amount", "Discount", 
                            "POST", "Manual", "Is_UK", "Label"], 
                           axis=1, inplace=True)
    tab_hash_compteur.reset_index(inplace=True)
    tab_hash_compteur["Date"] = tab_hash_compteur["InvoiceDate"]
    tab_hash_compteur["Distance"] = 0
    tab_hash_compteur.set_index(["InvoiceDate", "CustomerID"], 
                                inplace=True)
    
    for ind, val in tab_hash_compteur.iterrows():
        tab_compteur[ind[1]] = tab_compteur[ind[1]] + 1
        tab_hash_compteur.at[ind,"Occurence"] = tab_compteur[ind[1]]
        
    table_travail["Occurence"] = [(x, y) for x, y in 
                                  zip(table_travail["InvoiceDate"], 
                                      table_travail["CustomerID"])]
    table_travail["Occurence"] = \
        table_travail["Occurence"].apply(lambda x: 
                                         tab_hash_compteur["Occurence"][(x[0], 
                                                                         x[1])])
        
    for ind, val in tab_hash_compteur.iterrows():
        if tab_distance[ind[1]] == 0 :
            tab_distance[ind[1]] = ind[0]
            tab_hash_compteur.at[ind, "Distance"] = 0
        else:
            tab_hash_compteur.at[ind, "Distance"] = \
                int((ind[0] - tab_distance[ind[1]]).days)
        
    table_travail["Dist_first_trans"] = \
        [(x, y) for x, y in zip(table_travail["InvoiceDate"], 
                                table_travail["CustomerID"])]
    table_travail["Dist_first_trans"] = \
        table_travail["Dist_first_trans"].apply(lambda x: 
                                                tab_hash_compteur["Distance"][(x[0], 
                                                                               x[1])])
    return table_travail
        
    
    
def transform_extraction(table_p, max_day, list_client):
    """ Fonction qui transforme les séquences temporelles
    entrées clients en tableau de caractéristiques clients
    à l'horizon temporelle max_day"""
    
    MAX_DAY = max_day
    
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
    temp_tab = table_p[table_p["Dist_first_trans"] < MAX_DAY].copy()
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
    table_p[table_p["Dist_first_trans"] < MAX_DAY]
    table_essai_freq = \
        table_essai_freq[["InvoiceDate",
                          "CustomerID"]].drop_duplicates() \
                                        .groupby(["CustomerID"]) \
                                        .count()
    table_essai_freq = table_essai_freq["InvoiceDate"]
    
    # Table contenant le montant des achats sur la période considérée
    temp_tab = table_p[table_p["Dist_first_trans"] < MAX_DAY] 
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
    
    #table_trans["Distance"] = table_trans["Key"].apply(lambda x: 
    #                                                   tab_hash_dist[x])
    
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

def standardize_table(table_p, list_c, MAX_DAY):
    """Renvoie la table_p preparee et standardisee"""
    table_p2 = pd.get_dummies(table_p, columns = ["Month"])
    std_scale = preprocessing.StandardScaler() \
                             .fit(table_p2[list_c])
    X_norm = std_scale.transform(table_p2[list_c])
    outfile = open(f"std_scale_model2_{MAX_DAY}.pyc", 'wb')
    pickle.dump(std_scale, outfile)
    outfile.close()
    
    
    table_p2[list_c] = X_norm
    return table_p2


def evaluate_model_max_day(table_p, list_m):
    y = table_p["Label"].values
    table_p.drop(['Label'], axis=1, inplace=True)
    X = table_p.values
    
    X_train, X_test, y_train, y_test = \
    model_selection.train_test_split(X, y, test_size=0.3, stratify=y)
    
    
    """ Validation croisée de la régression logistique"""
    lr = LogisticRegression(max_iter=500)
    params_lr = {'C': np.logspace(-3, 1, 10) , 
             'penalty': ['l1', 'l2'], 'solver': ['liblinear']}
    log_reg = GridSearchCV(lr, params_lr, cv=5, scoring="accuracy",
                           return_train_score=True)
    log_reg.fit(X_train,y_train)
    
    """ Validation croisée du classifier support vecteur"""
    svc = LinearSVC(dual=False)
    params_svc = {'C': np.logspace(-3, 1, 8), 
              'penalty': ['l1','l2'], 'loss': ['squared_hinge']}
    sv_classifier = GridSearchCV(svc, params_svc, cv=5,
                                 scoring="accuracy",
                                 return_train_score=True)
    sv_classifier.fit(X_train,y_train)
    
    """ Validation croisée du classifier gradient boosting"""
    
    gbc = GradientBoostingClassifier()
    params_gbc = {'learning_rate': np.logspace(-3, 0, 4), 
                  'n_estimators': [10, 100, 300],
                  'max_depth': [2,3]}
    gbc_classifier = GridSearchCV(gbc, params_gbc, cv=5, 
                              scoring = "accuracy",
                              return_train_score = True)
    gbc_classifier.fit(X_train,y_train)
    
    
    gbc_clf = GradientBoostingClassifier(**gbc_classifier.best_params_)
    lr_clf = LogisticRegression(max_iter = 500, **log_reg.best_params_)
    svm_clf = LinearSVC(dual=False, **sv_classifier.best_params_)
    
    vot_clf = VotingClassifier(estimators = [('gbc',gbc_clf),
                                         ('lr',lr_clf),
                                         ('svm',svm_clf)], 
                               voting = 'hard')
    vot_clf.fit(X_train,y_train)
    y_pred = vot_clf.fit(X_train, y_train).predict(X_test)
    list_m.append(confusion_matrix(y_test, y_pred))
    
    
    return vot_clf.score(X_test, y_test)