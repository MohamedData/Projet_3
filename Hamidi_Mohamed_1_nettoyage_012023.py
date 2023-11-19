#!/usr/bin/env python
# coding: utf-8

# In[185]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[186]:


data = pd.read_csv('fr.openfoodfacts.org.products.csv', sep='\t')


# In[187]:


data.info()


# In[188]:


data.shape #320 772 lignes pour 162 colonnes


# In[189]:


data.head() 


# In[190]:


data.columns


# In[191]:


data.dtypes.value_counts() #106 colonnes de type float pour 56 de type objets


# In[192]:


round(data.isna().sum().sum()/(data.shape[0]*data.shape[1]),2) #76% de valeurs NaN


# ![valeurNa.png](attachment:valeurNa.png)

# On observe beaucoup de vide, certaines colonnes le sont quasiment totalement, on décide de sélectionner les colonnes avec plus de valeurs.

# In[193]:


data['countries_tags'].unique()


# In[194]:


mask = data['countries_fr'].str.contains('France', na=False)
df = data[mask].drop(['countries', 'countries_tags', 'countries_fr'], axis=1)


# In[195]:


df.shape #Les produits français sont au nombre de 98440 dans notre data frame


# In[196]:


df.isna().mean().hist() #beaucoup de colonnes avec un nombre de valeurs NaN élevé


# In[197]:


df = df.loc[:, df.isna().mean() < 0.6] #Sélection des colonnes ayant moins de 60% de valeurs NaN


# In[198]:


df.shape #Il reste 43 indicateurs pour 98440 produits


# In[199]:


df.columns #Indicateurs toujours disponible


# In[200]:


df.isna().mean().sort_values()


# Avec les colonnes dont on dispose, on veut trouver une idée d'application, on va définir chacune des colonnes pour avoir une vue d'ensemble : 
# - code : Code barre
# - url : Url de la page produit sur Open Foot Facts
# - creator : Premier contributeur qui ajoute en premier le produit
# - created_t : Date à laquelle le produit a été ajouté (format UNIX)
# - created_datetime : Data à laquelle le produit a été ajouté (format iso8601)
# - last_modified_t : Date à laquelle la page produit a été modifié (format UNIX)
# - last_modified_datetime : Date à laquelle la page produit a été modifié (format iso8601)
# - product_name : Nom du produit
# - quantity : Quantité et unité
# - packaging : Matière de l'emballage (Verre, plastique, carton)
# - packaging_tags : Matière de l'emballage (Verre, plastique, carton)
# - brands : Marque du produit
# - brands_tags : Marque du produit
# - categories : Catégories des produits ('Filet de Boeuf', 'Bonbons', 'Sodas au cola', ...)
# - categories_tags : Catégories des produits (En anglais ici)
# - categories_fr : Catégories des produits ('Filet de Boeuf', 'Bonbons', 'Sodas au cola', ...)
# - purchase_places : Lieux d'achats du produit
# - ingredients_text : Liste d'ingrédients du produit
# - additives_n : Nombres d'additifs 
# - additives : Additifs -> Produits ajoutés aux denrées alimentaires
# - ingredients_from_palm_oil_n : Nombre d'ingrédients issus de l'huile de palme
# - ingredients_that_may_be_from_palm_oil_n : Nombre d'ingrédients pouvant provenir de l'huile de palme
# - nutrition_grade_fr : Score nutrition
# - pnns_groups_1 : Catégorie produit ('Beverages', 'Fat and sauces', 'Salty snacks', ...)
# - pnns_groups_2 : Sous catégories produit ('Legumes', 'Meat', 'Nuts', ...)
# - states : Etats de remplissage de la fiche produit dans OpenFoodFacts
# - states_tags : Etats de remplissage de la fiche produit dans OpenFoodFacts
# - states_fr : Etats de remplissage de la fiche produit dans OpenFoodFacts (en Français)
# - main_category : Catégorie principal du produit ('Filet-de-boeuf', 'en:Cremes-vegetales-a-base-de-coco-pour-cuisiner', ...) 
# - main_category_fr : Catégorie principal du produit
# - image_url :  Adresse internet de l'image
# - image_small_url : Adresse internet de l'image réduite
# - energy_100g : Energie apportée par le produit (en Kj)
# - fat_100g : Lipide
# - saturated-fat_100g : Acide gras saturé
# - carbohydrates_100g : Glucide
# - sugars_100g : Sucre
# - fiber_100g : Fibre
# - proteins_100g : Protéine
# - salt_100g : Sel
# - sodium_100g : Sodium
# - nutrition-score-fr_100g : Nutri score français
# - nutrition-score-uk_100g : Nutri score anglais

# ![valeurNa2.png](attachment:valeurNa2.png)

# Idée d'application : Le client scan le produit, l'application lui donne (si elle existe) la plus grande qualité nutritionnelle du produit (Exemple : 'Très protéiné !'). Si elle n'en a pas renvoie simplement 'Pas de qualité nutritionnelle spécifique'. Ensuite, l'application suggère des produits de la même catégorie ayant de meilleurs qualités nutritionnelles. (Par catégorie : 'Plus protéiné !', 'Meilleurs graisses !', 'Meilleurs sucres !').

# ## Sélection des variables pertinentes

# Au regard de notre idée, nous choisissons de prendre les variables suivantes :
# - code : Essentiel pour récupérer nos informations produits
# - product_name
# - Variables de catégorie : categories, categories_tags, categories_fr, pnns_groups_1, pnns_groups_2, main_category, main_category_fr
# - Variables nutritionnelles : energy_100g, fat_100g, saturated-fat_100g, carbohydrates_100g, sugars_100g, fiber_100g, proteins_100g, salt_100g, sodium_100g
# - Nutri Score : nutrition-score-fr_100g, nutrition_grade_fr

# In[201]:


#Création d'une liste repertoriant nos indicateurs
var_to_select = ['code', 'product_name', 'categories', 'categories_tags', 'categories_fr', 'pnns_groups_1','pnns_groups_2', 
                 'main_category', 'main_category_fr', 'energy_100g', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g',
                 'sugars_100g', 'fiber_100g', 'proteins_100g', 'salt_100g', 'sodium_100g', 'nutrition-score-fr_100g', 'nutrition_grade_fr']


# In[202]:


#Sélection des variables
df = df[var_to_select]


# ## Nettoyage de données

# ### Identification et suppression des doublons

# In[203]:


#Nombre de doublons ayant un code similaire
df.duplicated(['code']).sum()


# In[204]:


#Affichage des doublons
df.loc[df[['code']].duplicated(keep=False),:].sort_values(by='code')


# On veut supprimer ces doublons. Etant donné leurs petits nombre on se décide de récupérer les index de ceux qu'on veut supprimer et de les supprimer à la main.

# In[205]:


#Liste des index des doublons à supp
doublon_a_supp = [9892, 519, 67371, 80034]


# In[206]:


#Suppression des doublons sélectionnés
df.drop(doublon_a_supp, inplace=True)


# ### Identification et traitement des valeurs aberrantes

# In[207]:


df.describe()


# On sait que les variables fat_100g, saturated-fat_100g, carbohydrates_100g, sugars_100g, fiber_100g, proteins_100g, salt_100g et sodium_100g doivent être comprisent entre 0 et 100. Or, on remarque que la plupart d'entre elles ont des valeurs qui sortent de cet intervalle. Remplaçons ces valeurs par la valeur NaN.

# In[208]:


#On créer une liste des colonnes que l'on veut modifier
macro = ['fat_100g', 'saturated-fat_100g',
       'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g',
       'salt_100g', 'sodium_100g']


# In[209]:


#On remplace l'ensemble des valeurs > 100 ou < 0
for mac in macro : 
    df.loc[(df[mac] < 0) | (df[mac] > 100), mac] = np.NaN


# In[210]:


#On retrouve des valeurs comprisent entre 0 et 100
df.describe()


# Sachant qu'un produit est globalement composé de glucide, lipide, protéine et de sel. Ainsi pour 100g de produit la somme fat_100g	+ carbohydrates_100g + proteins_100g + salt_100g + fiber_100g ne peut être supérieure à 100g sinon on peut les considérer comme des valeurs aberrantes.

# In[211]:


#On créer le sous-dataframe de df que l'on nomme valeur_nutri composé des 5 colonnes qui nous intéressent :
valeur_nutri = df[['fat_100g',
       'carbohydrates_100g', 'proteins_100g', 'salt_100g', 'fiber_100g']]


# In[212]:


#Produits ayant une somme fat_100g + carbohydrates_100g + proteins_100g + salt_100g + fiber_100g aberrantes, on en répertorie 468
df[valeur_nutri.sum(axis=1) > 100][['product_name', 'fat_100g',
       'carbohydrates_100g', 'proteins_100g', 'salt_100g', 'fiber_100g']]


# In[213]:


#Création d'une liste répertoriant l'ensemble des indexs des observations à supprimer
a_supp = list(df[valeur_nutri.sum(axis=1) > 100].index)

#Suppression des éléments aberrants
df.drop(a_supp, inplace=True)


# Le sucre est un élément faisant partit des glucides, il ne peut donc exceder la quantité de glucide. Par exemple, pour un produit composé de 27g de glucides il ne peut y avoir plus de 27g de sucre. Vérifions qu'aucune valeurs n'est aberrante en ce sens.

# In[214]:


df[df['sugars_100g'] > df['carbohydrates_100g']][['sugars_100g', 'carbohydrates_100g']]


# 99 produits ont une valeur pour le sucre supérieure que la valeur des glucides, la raison est parfois liée à l'arondissement (comme sur la première ligne), cependant on trouve parfois des différences aberrantes (20.5g de sucre pour 0g de glucide).

# In[215]:


#Création d'une liste répertoriant l'ensemble des valeurs aberrantes
sugar_a_supp = list(df[df['sugars_100g'] > df['carbohydrates_100g']].index)

#Suppression des valeurs aberrantes
df.drop(sugar_a_supp, inplace=True)


# On applique cette même logique aux acides gras saturés qui sont une composante des lipides:

# In[216]:


df[df['saturated-fat_100g'] > df['fat_100g']][['saturated-fat_100g', 'fat_100g']]


# 80 produits sont concernés par un écart aberrant entre le poids mesuré des lipides et celui d'une de ses sous catégories que sont les acides gras saturés.

# In[217]:


#Création d'une liste répertoriant les indices des valeurs aberrantes
gras_a_supp = list(df[df['saturated-fat_100g'] > df['fat_100g']].index)

#Suppression des valeurs aberrantes
df.drop(gras_a_supp, inplace=True)


# Si on considère que les macros nutriments les plus caloriques sont les lipides (avec 1g de lipide = 9kcal, 1g glucide = 4kcal, 1g protéine = 4kcal) alors pour 100g de produit ce dernier ne peut exceder 9 x 100 = 900kcal de produits. Si on passe ensuite des kcal aux kjoules alors on a qu'un produit ne peut pas fournir plus de 3 765,6 kjoules pour 100g. Mettons en évidence les outliers.

# In[218]:


#On récupère les observations ayant des valeurs non nulles pour l'énergie
energy_prod = df[~ df['energy_100g'].isna()]['energy_100g'].sort_values()


# In[219]:


#On observe ainsi 102 observations superieures à la limite théorique 
df.loc[energy_prod[energy_prod > 4.184*900 + 2 ].index, ['product_name', 'energy_100g']]


# In[220]:


#Création d'une liste répertoriant les indices des valeurs aberrantes
energy_a_supp = list(energy_prod[energy_prod > 4.184*900 + 2 ].index)

#Suppression des valeurs aberrantes
df.drop(energy_a_supp, inplace=True)


# ### Identification et traitement des valeurs NaN

# In[221]:


#Barplot de la proportion de valeurs NaN
df.isna().mean().sort_values().plot.bar()
plt.show()


# On observe 2 variables en dessous de 10% de valeurs NaN, un bloc de 13 autres ayant entre 30% et 40% de valeurs NaN et un dernier bloc de 3 variables ayant entre 50% et 60% de valeurs NaN.

# In[222]:


#On impute aux valeurs vide de sel ayant une valeur pour le sodium : val_sel = 2.5 * val_sodium
df.loc[df['salt_100g'].isna(), 'salt_100g'] = (df.loc[df['salt_100g'].isna(), 'sodium_100g'] * 2.5)


# In[223]:


#On impute aux valeurs vide de sodium ayant une valeur pour le sel : val_sodium = val_sel / 2.5
df.loc[df['sodium_100g'].isna(), 'sodium_100g'] = (df.loc[df['sodium_100g'].isna(), 'salt_100g'] / 2.5)


# In[224]:


#On observe que deux valeurs aberrantes se sont ajoutées après notre traitement.
df[(df['salt_100g'] > 100) | (df['sodium_100g'] > 40)]


# In[225]:


#On change à la main les valeurs du produit
df.loc[df['product_name'] == 'Keeny Bio', 'sodium_100g'] = 0.18
df.loc[df['product_name'] == 'Keeny Bio', 'salt_100g'] = 0.18*2.5


# In[226]:


#On change à la main les valeurs du produit
df.loc[df['product_name'] == 'Sel de Guérande Label Rouge', 'sodium_100g'] = 40
df.loc[df['product_name'] == 'Sel de Guérande Label Rouge', 'salt_100g'] = 100


# In[227]:


#Plus de 30 000 valeurs sont totalement vides pour nos colonnes numériques
df[['fat_100g', 'saturated-fat_100g',
       'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g',
       'salt_100g', 'sodium_100g']].isna().mean(axis=1).hist()


# In[228]:


#Supprimons l'ensemble des lignes n'ayant aucune valeur numérique
index_a_supp = list(df[df[['fat_100g', 'saturated-fat_100g',
       'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g',
       'salt_100g', 'sodium_100g']].isna().mean(axis=1) == 1].index)


# In[229]:


#Suppression des lignes totalement vide
df.drop(index_a_supp, inplace=True)


# In[230]:


#Barplot de la proportion de valeurs NaN
df.isna().mean().sort_values().plot.bar()
plt.show()


# In[231]:


#On réecrit certaines valeurs de pnns_groups_1 en y ajoutant une majuscule pour que 
#ces valeurs rentrent dans la bonne catégorie
df.loc[df['pnns_groups_1'] == 'cereals-and-potatoes', 'pnns_groups_1'] = 'Cereals and potatoes'
df.loc[df['pnns_groups_1'] == 'sugary-snacks', 'pnns_groups_1'] = 'Sugary snacks'
df.loc[df['pnns_groups_1'] == 'fruits-and-vegetables', 'pnns_groups_1'] = 'Fruits and vegetables'
df.loc[df['pnns_groups_1'].isna(), 'pnns_groups_1'] = 'unknown'


# In[232]:


#On réecrit certaines valeurs de pnns_groups_2 en y ajoutant une majuscule pour que 
#ces valeurs rentrent dans la bonne catégorie
df.loc[df['pnns_groups_2'] == 'cereals', 'pnns_groups_2'] = 'Cereals'
df.loc[df['pnns_groups_2'] == 'vegetables', 'pnns_groups_2'] = 'Vegetables'
df.loc[df['pnns_groups_2'] == 'fruits', 'pnns_groups_2'] = 'Fruits'
df.loc[df['pnns_groups_2'].isna(), 'pnns_groups_2'] = 'unknown'


# In[233]:


from sklearn.impute import KNNImputer
#On créer une liste repertoriant l'ensemble des sous-catégories
groupes = df['pnns_groups_2'].unique()

for group in groupes:
    X = np.array(df.loc[df['pnns_groups_2'] == group, ['fat_100g', 'saturated-fat_100g',
       'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g',
       'salt_100g', 'sodium_100g']])
    
    imputer = KNNImputer(n_neighbors=5)
    
    X_transfo = imputer.fit_transform(X)
    
    df.loc[df['pnns_groups_2'] == group, ['fat_100g', 'saturated-fat_100g',
       'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g',
       'salt_100g', 'sodium_100g']] = X_transfo


# In[234]:


#Utilisons le KNNImputer afin de remplir les valeurs restantes (NaN et unknown)
from sklearn.impute import KNNImputer
X = np.array(df[['fat_100g', 'saturated-fat_100g',
       'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g',
       'salt_100g', 'sodium_100g']])
imputer = KNNImputer(n_neighbors=5)
X_transfo = imputer.fit_transform(X)


# In[235]:


#On modifie les colonnes du DataFrame 
df.loc[:, ['fat_100g', 'saturated-fat_100g',
       'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g',
       'salt_100g', 'sodium_100g']] = X_transfo


# In[236]:


#Barplot de la proportion de valeurs NaN
df.isna().mean().sort_values().plot.bar()
plt.show()


# In[237]:


#On répertorie dans une liste les index des observations dont on veut modifier certaines valeurs
huile_a_modifier = list(df[df['carbohydrates_100g']*4 + df['proteins_100g']*4 + df['fat_100g']*9 + df['fiber_100g']*2 > 900].index)

#On sait que les glucides, sucres, fibres, protéines, sel et sodium des huiles sont égals à 0 on change donc les valeurs
df.loc[huile_a_modifier, ['carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g',
       'salt_100g', 'sodium_100g']] = np.NaN


# On sait que saturated-fat_100g ne peut execeder la valeur de fat_100g et que sugars_100g ne peut exceder la valeur de carbohydrates_100g car ils en sont les composés. Sachant cela, on décide de réajuster les valeurs des colonnes.

# In[238]:


#On observe les valeurs aberrantes pour le rapport glucide/sucre et lipide/acide gras saturés
df[(df['sugars_100g'] > df['carbohydrates_100g']) | (df['saturated-fat_100g'] > df['fat_100g'])]


# In[239]:


#Pour chaque valeurs de 'sugars_100g' superieurs à sa valeur dans 'carbohydrates_100g', on mets à égalité
df.loc[df['sugars_100g'] > df['carbohydrates_100g'], 'sugars_100g'] = df.loc[df['sugars_100g'] > df['carbohydrates_100g'], 'carbohydrates_100g']


# In[240]:


#Pour chaque valeurs de 'saturated-fat_100g' superieurs à sa valeur dans 'fat_100g', on mets à égalité
df.loc[df['saturated-fat_100g'] > df['fat_100g'], 'saturated-fat_100g'] = df.loc[df['saturated-fat_100g'] > df['fat_100g'], 'fat_100g']


# Les indicateurs nutritifs sont tous remplit à l'exception de l'energie, pour ce faire on peut l'estimer nous même, en effet : 
# - 1g de glucide = 4kcal
# - 1g de protéine = 4kcal
# - 1g de lipide = 9kcal
# - 1g de fibre = 2kcal

# In[241]:


#Estimation de l'énergie à la main
df.loc[df['energy_100g'].isna(), 'energy_100g'] = df.loc[df['energy_100g'].isna(), 'carbohydrates_100g']*4 + df.loc[df['energy_100g'].isna(), 'proteins_100g']*4+ df.loc[df['energy_100g'].isna(), 'fat_100g']*9 + df.loc[df['energy_100g'].isna(), 'fiber_100g']*2 


# In[242]:


#Suppression des valeurs non remplit
df = df.loc[df[['energy_100g', 'fat_100g', 'saturated-fat_100g',
       'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g',
       'salt_100g', 'sodium_100g']].isna().mean(axis=1)==0,:]


# In[243]:


#Barplot de la proportion de valeurs NaN
df.isna().mean().sort_values().plot.bar()
plt.show()


# In[244]:


#On cherche la présence d'outliers
df.describe()


# On a plus aucune valeurs NaN et aucune valeurs aberrantes n'apparait lorsqu'on fait un describe(). On a donc terminé notre nettoyage.

# In[245]:


#On exporte notre dataframe df en 'df_final.csv'
df.to_csv("df_final.csv", index=False)

