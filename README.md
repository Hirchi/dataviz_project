# dataviz_project
 Analyse brivement:
La base DVF enregistre les données des transactions immobilières de tous les communes en frances métropoles (hors Alsace et Moselle) ainsi que les département d'Outre-Mer (hors Mayotte).
À partir de ce jeux de données, nous pourons faire des analyses pour la valeur foncières en france selon différentes critères (typologie de biens, la nature de mutation,créer des outils cartographique interactive avec la valeur foncière par chaque communes et par le département (par exemple, comme les outils cartographiques que MeilleursAgents a utilisé, nous pourrons faire la carte de prix par m² des bureaux/commerces/entrepôts par communes en France).
Nous pourrons savoir aussi à partir de ce jeu de données l'évoution selon le temps ( entre 2017 et 2020 ) pour chaque type de biens.
Le point faible de ce jeu, c'est qu'il n'y a pas mal des informations manquantes (comme les caractéristiques du bien, le nombre d'étage, ascenseur, année de construction ...) qui nous permet de faire une estimation la valeur du bien en utilisant la méthode hédonique et en prendre en compte l'impacte de la localisation spatial du bien.
De plus,le jeu de données contient aussi beaucoup des données manquants.
Le jeux de données est très lourd donc je l'ai enregistrer dans un bucket s3 sur aws
L'application streamlit n'as que 10 mb de ram donc je charge que 900000 ligne et à cause de ça la les résultats sur l'application sont pas complets 
Le nettoyage ansi que mes différente interprétation de données est dur l'application
