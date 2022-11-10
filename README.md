# OC-Projet7-Scoring_App-API_Part
Projet 7 du parcours Data Scientist de OpenClassroom - Partie serveur web API

Nous souhaitons mettre en place un outil de « scoring credit » pour la société « Prêt à dépenser », afin de déterminer la probabilité de remboursement d’un client, dans l’objectif de classifier la demande en crédit en accordé ou refusé. Pour définir le modèle de classification, la société nous a fourni une base de données de plus de 300 000 clients anonymisés, comportant entre autres des informations comportementales, les historiques de demande de crédits dans diverses institutions financières. Nous nous retrouverons avec plus de 300 variables. 

Ce repository contient les éléments nécessaires au fonctionnement d’un serveur dashboard faisant appel aux modèles de classification entrainé disponible via des requetes api définit dans le repository OC-Projet7-Scoring_App-API_Part.

Le repository contient les fichiers suivants :
    - main.py : lance le serveur dashboard
    - dashboard/app.py : gère les différentes pages du dashboard
    - dashboard/pages/info_indiv.py : gère l'interface d'affichage des informations d'un individu.
    - dashboard/pages/prediction.py : gère l'interface d'affichage des résultats des prédictions d'un individu.
    - dashboard/pages/model.py : gère l'interface d'affichage des performances du modèle de classification choisi.
    - dashboard/pages/callbacks.py : gère toutes les requêtes internes et externes du serveur.