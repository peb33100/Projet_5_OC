import itertools
import matplotlib.pyplot as plt
import numpy as np



def plot_resultats(a, b, c, cl_names):
    """Fonctions retournant la synthèse des résultats"""
    class_names = cl_names
    matrice = np.zeros((3,5))
    matrice[0,:] = a
    matrice[1,:] = b
    matrice[2,:] = c
    plt.figure(figsize=(12, 8))
    plt.imshow(matrice, interpolation='nearest', 
               cmap=plt.cm.Blues)
    plt.title("Synthèse des résultats", fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    tick_h = np.arange(3)
    plt.xticks(tick_marks, class_names,
               rotation=45, fontsize=16)
    plt.yticks(tick_h, ["Recall", "Precision",
                        "F1_score"], fontsize=16)

    fmt = '.2f'
    thresh = 0.7
    for i, j in itertools.product(range(matrice.shape[0]), 
                                  range(matrice.shape[1])):
        plt.text(j, i, format(matrice[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if matrice[i, j] > thresh else "black",
                 fontsize=16)

    plt.tight_layout()
    plt.xlabel('Classe de prédiction', fontsize=16)
    plt.ylabel("Critères d'evaluation", fontsize=16)
    plt.show()



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=16)
    plt.yticks(tick_marks, classes, fontsize=16)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)
    
    
def afficher_3D_representation(X, cluster, method, c1, c2, c3):
    """ Affiche sous forme 3D les clusters 
    définis par la méthode method des points 
    contenus dans X """
    
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=X[:, 0], ys=X[:, 1], zs=X[:, 2], c=cluster.labels_)
    ax.set_xlabel(c1)
    ax.set_ylabel(c2)
    ax.set_zlabel(c3)
    plt.title("Représentation 3D des clusters définis par " + method)
    plt.show()
    
    
def afficher_2D_representation(X, cluster, method, algo, c1, c2, c3):
    """Affiche sous forme de 3 représentations 2D les clusters définis 
    par la méthode method des points contenus dans X suivant les dimensions 
    réduites otenues en utilisant l'algorithme algo"""
    
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(131)
    ax.scatter(X[:, 0], X[:, 1], c=cluster.labels_)
    plt.xlabel(algo + c1, fontsize=16)
    plt.ylabel(algo + c2, fontsize=16)
    plt.title(method)

    ax = fig.add_subplot(132)
    ax.scatter(X[:, 1], X[:, 2], c=cluster.labels_)
    plt.xlabel(algo + c2, fontsize=16)
    plt.ylabel(algo + c3, fontsize=16)
    plt.title(method)

    ax = fig.add_subplot(133)
    ax.scatter(X[:, 0], X[:, 2], c=cluster.labels_)
    plt.xlabel(algo + c1, fontsize=16)
    plt.ylabel(algo + c3, fontsize=16)
    plt.title(method)
    
    plt.show()