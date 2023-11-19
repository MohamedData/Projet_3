#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[3]:


data = pd.read_csv('df_final.csv')


# In[4]:


data.shape


# In[5]:


#On vérifie que le dataframe est bien chargé
data.head()


# In[6]:


#On sélectionne les observations n'ayant aucune valeur manquante pour les colonnes'pnns_groups_1' et 'pnns_groups_2'
df = data[data[['pnns_groups_1', 'pnns_groups_2']].isna().sum(axis=1) == 2]


# In[7]:


#Nombre de valeurs manquantes dans toutes les autres colonnes "catégories"
df[['categories', 'categories_tags', 'categories_fr', 'main_category', 'main_category_fr']].isna().mean()


# L'ensemble des autres colonnes "catégories" sont entièrement vide lorsqu'on retire les observations déjà informé par 'pnns_groups_1' ou 'pnns_groups_2'. Ainsi, elles ne nous sont d'aucune utilité et on préfère les retirer.

# In[8]:


#Suppression des colonnes 'categories', 'categories_tags', 'categories_fr', 'main_category', 'main_category_fr'
data.drop(['categories', 'categories_tags', 'categories_fr', 'main_category', 'main_category_fr'], axis=1, inplace=True)


# In[9]:


#Suppression des observations n'ayant pas de valeurs pour le nutriscore
data = data.loc[data[['nutrition-score-fr_100g', 'nutrition_grade_fr']].isna().mean(axis=1) == 0, :]


# ## Analyse univariée

# ### Variables qualitatives

# In[10]:


#Description générale
data.describe(exclude = [np.number]).T


# On voit que 'pnns_groups_1' représente 10 catégories tandis que 'pnns_groups_2' représente 37 sous_catégories pour chacun des aliments. Nous étudierons simplement 'pnns_groups_1' dans l'analyse univariée dans un soucis de clarté.

# In[11]:


#On commence par afficher le tableau des effectifs/fréquences de pnns_groups_1 

effectifs = data['pnns_groups_1'].value_counts()
modalites = effectifs.index # l'index de effectifs contient les modalités

tab = pd.DataFrame(modalites, columns = ["Categories"]) # création du tableau à partir des modalités
tab["n"] = effectifs.values
tab["f"] = tab["n"] / len(data) # len(data) renvoie la taille de l'échantillon
val_nan = pd.DataFrame([['NaN', data['pnns_groups_1'].isna().sum(), data['pnns_groups_1'].isna().mean()]], columns=['Categories', 'n', 'f'], index=[10])
tab = tab.append(val_nan)
tab = tab.sort_values("n") # tri des valeurs de la variable X (croissant)
tab["F"] = tab["f"].cumsum() # cumsum calcule la somme cumulée
tab


# In[12]:


#A l'aide d'un countplot on affiche le diagramme en baton du nombre de produit par catégorie
sns.countplot(data=data, y='pnns_groups_1', order=data['pnns_groups_1'].value_counts().index)


# On remarque que beaucoup de catégories dites "mauvaise pour la santé" sont très représentées (snacks sucrés, produits laitieres, boissons etc), on veut alors voir ce qu'en dit le NutriScore.

# In[13]:


#On continue en affichant le tableau des effectifs/fréquences de nutrition_grade_fr

effectifs = data['nutrition_grade_fr'].value_counts()
modalites = effectifs.index # l'index de effectifs contient les modalités

tab = pd.DataFrame(modalites, columns = ["Categories"]) # création du tableau à partir des modalités
tab["n"] = effectifs.values
tab["f"] = tab["n"] / len(data) # len(data) renvoie la taille de l'échantillon
tab = tab.sort_values("n") # tri des valeurs de la variable X (croissant)
tab["F"] = tab["f"].cumsum() # cumsum calcule la somme cumulée
tab


# On observe que la fréquence cumulée des deux meilleurs notes (a et b) est quasiment équivalente à la fréquence de l'avant dernière note (d). Cela signifie que beaucoup de produit sont mal notés.

# In[14]:


#A l'aide d'un countplot on affiche le diagramme en baton du nombre de produit par catégorie
sns.countplot(data=data, y='nutrition_grade_fr', order=['a','b','c','d','e'], palette=['g', 'yellowgreen', 'gold', 'darkorange', 'r'])


# Les 3 notes ayant la plus grande proportion de produit sont également les plus mauvaises avec dans l'ordre d, c et e. Mettons en valeur cette différence significative entre les produits bien notés et les produits mal notés. Considérons que :
# - Bonnes notes : a et b.
# - Mauvaises notes : c, d et e.

# In[97]:


#Création de nos 2 catégories avec leurs pourcentages respectives
proportion = [round(data['nutrition_grade_fr'].isin(['a', 'b']).mean(), 2)* 100, round(data['nutrition_grade_fr'].isin(['c', 'd', 'e']).mean(), 2)* 100]
labels = ['Bonnes notes', 'Mauvaises notes']

#Palette Seaborn
colors = sns.color_palette('pastel')[0:5]

#Création du diagramme en camembert
plt.pie(proportion, labels = labels, colors = colors, autopct='%.0f%%')
plt.show()


# On observe donc que 2 produits sur 3 sont mal notés, cela montre la nécessité pour les clients de choisir des produits de meilleurs qualités. L'étiquette 'Protéiné', 'Bons sucres', 'Bonnes graisses' ou 'Pas de qualités nutritionnelles particulières' permet de donner rapidement une indication sur les bienfaits du produits. De plus, les suggestions vont l'enjoindre à se tourner vers de meilleurs produits.

# ### Variables quantitatives

# In[15]:


data.describe().T


# Analyse des macronutriments : 
# - Moyenne : Tout d'abord on observe que les lipides, les glucides et le sucre sont du même ordre de grandeur $(10^{1})$. D'une autre part, on observe que le sel, les protéines, les fibres ainsi que les acides gras saturés sont d'une ordre de grandeur inférieure $(10^{0})$.
# - Ecart type : On observe que les  lipides, les glucides ainsi que le sucre sont du même ordre de grandeur $(10^{1})$. Ils ont donc tendances à être plus dispersé que les protéines, sel, sodium, fibres et acide gras saturés ayant un ordre de grandeur inférieur. $(10^{0})$
# - Mediane : On observe que les protéines, fibres, sucres, lipides et acide gras saturés ont une médiane avec un ordre de grandeur plus élevé $(10^{0})$ que le sel et le sodium $(10^{1})$. Cependant, la variable glucide a le plus grand ordre de grandeur $(10^{1})$.

# In[16]:


for element in ['energy_100g',
       'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g',
       'fiber_100g', 'proteins_100g', 'salt_100g', 'sodium_100g']:
        plt.figure(figsize=(2,2))
        sns.displot(data=data,
            x=element,
            kind="kde",
            log_scale=False);
        plt.axvline(x=round(data[element].mean(),2), color='r') #On mets en relief la moyenne
        plt.axvline(x=round(data[element].median(),2), color='g') #On mets en relief la médiane
        plt.show()
        print('Moyenne :', round(data[element].mean(),2),'|','Mediane :', round(data[element].median(),2), 'Ecart Type :','|', round(data[element].std(),2))


# Le résultat ci-dessus nous apprends que : 
# - Le sodium, le sel, les fibres, et les acides gras saturés ont des profils statistiques très proches, en effet tout d'abord ils ont une médiane très proche de la moyenne, de plus ces deux mesures sont très petites (Entre 0 et 5 pour la moyenne et 2 pour la médiane). 
# - Pour l'ensemble des variables la moyenne est plus grande que la médiane, cela signifique que la distribution est étalée à droite.

# In[17]:


order_box = list(data[['fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g',
       'fiber_100g', 'proteins_100g', 'salt_100g', 'sodium_100g']].median().sort_values().index)

sns.boxplot(showfliers=False, data=data[['fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g',
       'fiber_100g', 'proteins_100g', 'salt_100g', 'sodium_100g']], orient="h", order=order_box);


# Les boxplots produit ci-dessus nous renseignent avec plus de détails quand à nos variables quantitatives. On observe déjà que les 3 macros nutriments principaux (glucide, protéine, lipide) ont la plus grande médiane. De plus, à toute proportion gardé on observe que la forme de l'ensemble de nos variables numériques est assez similaire (distance entre q1 et q2 inferieure à la distance entre q2 et q3, étalée sur la droite). De nos macronutriments, proteins_100g est le moins dispersée tandis que les glucides le sont énormément avec un q1 < 10 et un q3 > 50.

# In[18]:


plt.figure(figsize=(2,2))
sns.displot(data=data, x='nutrition-score-fr_100g', kind="kde", log_scale=False);
plt.axvline(x=round(data['nutrition-score-fr_100g'].mean(),2), color='r') #On mets en relief la moyenne
plt.axvline(x=round(data['nutrition-score-fr_100g'].median(),2), color='g') #On mets en relief la médiane
plt.show()
print('Moyenne :', round(data['nutrition-score-fr_100g'].mean(),2),'|','Mediane :', round(data['nutrition-score-fr_100g'].median(),2), 'Ecart Type :','|', round(data['nutrition-score-fr_100g'].std(),2))


# On rappelle que plus le nutrition-score est petit, meilleur est la qualité du produit. Ainsi, on remarque qu'à 9 il y a une séparation entre les bons et moins bon produits. On rappelle qu'entre 3 et 10 le score est de c, ainsi la courbe à droite (avec un pic) correspond aux produits avec un score de d et e. A gauche, on retrouve une grande partie des produits notés c, puis b et enfin a. 

# In[19]:


#Boite à moustache améliorée
sns.boxenplot(data=data, x='nutrition-score-fr_100g', showfliers=False)


# Cette boite à moustache particulière nous confirme l'intuition que l'on avait dans le précédent histogramme. En effet, on a une symétrie quasi parfaite de part et d'autre de la médiane pour ce qui est de la proportion des valeurs et ce sur des tranches de valeurs quasiment de même longueur.

# ## Analyse bivariée

# ### Variables quantitatives/quantitatives

# Commençons l'analyse bivariée avec un tableau de corrélation : 

# In[20]:


data[['nutrition-score-fr_100g', 'energy_100g', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g',
       'fiber_100g', 'proteins_100g', 'salt_100g', 'sodium_100g']].corr()


# Mettons en valeur les corrélations significatives à l'aide d'une heatmap.

# In[21]:


# Compute the correlation matrix
corr = data[['nutrition-score-fr_100g', 'energy_100g', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g',
       'fiber_100g', 'proteins_100g', 'salt_100g', 'sodium_100g']].corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True);


# On observe des corrélations positives :
# - Le sel et le sodium  car les deux sont liés par un rapport (1g de sodium = 2.5g de sel)
# - Les glucides et le sucre car le sucre est un composé des glucides
# - Les lipides et les acides gras saturés car les acides gras saturés sont un composés des lipides
# - L'énergie et les lipides car les lipides sont les macronutriments les plus energétiques, leurs présences (ou non d'ailleurs) peut vite faire varier la quantité d'énergie du produit.
# - Le nutri score, les acides gras saturés et le sucre simple. Cela s'expliquant simplement par le fait que ce sont ces éléments (en partie) qui pondèrent le nutri score. 

# In[22]:


#Affichons sur un pairplot les nuages de points des variables protéines, lipides, glucides et energie
sns.pairplot(data[['nutrition-score-fr_100g', 'energy_100g', 'fat_100g', 'carbohydrates_100g', 'proteins_100g']], diag_kind="kde")


# Pour l'ensemble des relations entre ces variables on observe pas de corrélation linéaire intéressante sauf pour le couple energy_100g et fat_100g décrit précedemment dans lequel on observe une tendance linéaire sur le nuage de point.

# ### Variables qualitatives/qualitatives

# Voyons si 'pnns_groups_2' représente les sous-catégories de 'pnns_groups_1'.

# In[23]:


#Tableau de contingence de 'pnns_groups_1' et 'pnns_groups_2'
pd.crosstab(data['pnns_groups_2'], data['pnns_groups_1'])


# On observe bien que 'pnns_groups_2' représente bien des sous catégories des catégories contenuent dans 'pnns_groups_1'. Par exemple la colonne Beverages (Boissons) se décompose en 5 sous catégories : Sweetened beverages (Boissons sucrées), Non-sugared beverages (Boissons non sucrées), Fruit nectars (Nectars de fruit), Fruit juices (Jus de fruit), Artificially sweetened beverages (Boissons sucrées artificielles).

# In[24]:


#Test de Khi-2
from scipy.stats import chi2_contingency as chi2_contingency

khi2, pval , ddl , contingent_theorique = chi2_contingency(pd.crosstab(data['pnns_groups_2'], data['pnns_groups_1']))
print(pval)


# La p-value étant très inférieure à 0.05 (= 0) on considère donc qu'il y a dépendance entre les 2 variables.

# In[25]:


sns.set(rc = {'figure.figsize':(10,8)})
sns.heatmap(pd.crosstab(data['pnns_groups_1'], data['nutrition_grade_fr'], normalize='index'), annot=True)


# Le résultat ci-dessus nous apprends beaucoup de chose concernant les catégories de produits avec une répartition plutôt uniforme entre toutes les notes ('Fish Meat Eggs', 'unknown', 'composite foods', 'beverages'), les catégories plutôt bien noté ('cereals and potatoes' et 'fruits and vegetables' avec une grosse proportion noté a) et celles mal notées ('Fat and sauces', 'Milk and dairy products', 'salty snacks' avec une grosse proportion noté d ainsi que 'sugary snacks' avec une grosse proportion noté d et e).

# In[26]:


sns.set(rc = {'figure.figsize':(10,8)})
sns.heatmap(pd.crosstab(data['pnns_groups_2'], data['nutrition_grade_fr'], normalize='index'))


# On peut affiner notre analyse en prenant cette fois-ci les sous catégories. On observe alors que :
# - Les légumes, pomme de terre, légumineuses, fruits, oeufs et céréales ont toutes une grande proportion de produit ayant la note a
# - Les soupes, lait et yaourt, fruits secs, les boissons édulcorée artificiellements ayant une grande proportion de produit ayant la note de b
# - Les jus de fruits, desserts lactés, céréales du petit déjeuner ayant une grande proportion de produit ayant la note de c
# - Les patisseries, plats à base de tripes, bonbons, sandwich, produits salés et gras, pizzas et quich, noisettes, glaces, graisses, fromage, appéritifs et boissons alcoolisés ayant une grande proportion de produits ayant la note de d. On remarque que beaucoup de ces produits sont composés de sucre simple et de gras. 
# - Les boissons sucrées, viandes transformées, nectars de fruits, produits chocolatiers, et les cakes/biscuits ont une grande proportion de produit ayant la note de e. On fait la même remarque que précedemment, ces catégories ont des produits composées soit de sucre simple, soit de gras, soit des deux.

# In[27]:


#Observons avec un countplot
plt.figure(figsize=(10, 10))
ordre = data[data['nutrition_grade_fr'] == 'a'].groupby('pnns_groups_1').count()['nutrition_grade_fr'].sort_values().index
sns.countplot(y='pnns_groups_1', hue='nutrition_grade_fr', data=data, order=ordre, hue_order=['a','b','c','d','e'], palette=['g', 'yellowgreen', 'gold', 'darkorange', 'r']);


# Pour finir notre analyse croisée entre le score nutritif et les catégories/sous-catégories d'aliments, on aimerait savoir si les couples de variables score nutritif/catégories et score nutritif/sous-catégories sont liées ou indépendantes.

# In[28]:


#Test de Khi-2 entre 'pnns_groups_1' et 'nutrition_grade_fr'
from scipy.stats import chi2_contingency as chi2_contingency

khi2, pval , ddl , contingent_theorique = chi2_contingency(pd.crosstab(data['pnns_groups_1'], data['nutrition_grade_fr']))
print(pval)


# On observe que la p-value est de 0. On peut donc conclure sur la dépendance des deux indicateurs produits.

# In[29]:


#Test de Khi-2 entre 'pnns_groups_2' et 'nutrition_grade_fr'
from scipy.stats import chi2_contingency as chi2_contingency

khi2, pval , ddl , contingent_theorique = chi2_contingency(pd.crosstab(data['pnns_groups_2'], data['nutrition_grade_fr']))
print(pval)


# Ici aussi la pvalue est très petite, cela signifie que 'pnns_groups_2' et 'nutrition_grade_fr' sont liées.

# ### Variables qualitatives/quantitatives

# In[30]:


#ANOVA
def eta_squared(x,y):
    moyenne_y = y.mean()
    classes = []
    for classe in x.unique():
        yi_classe = y[x==classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj-moyenne_y)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
    return SCE/SCT


# Les glucides, protéines, lipides et énergies ont-elles un impact sur le nutri score ?

# In[31]:


list_nutri = ['energy_100g', 'fat_100g', 'carbohydrates_100g', 'proteins_100g']
for nutri in list_nutri:
    sns.boxenplot(data=data, x=nutri, y ='nutrition_grade_fr', showfliers=False, order=['a','b','c','d','e'], palette=['g', 'yellowgreen', 'gold', 'darkorange', 'r'])
    plt.show()


# Lorsqu'on observe les 4 graphes produit ci-dessus, on remarque que :
# - L'énergie a 5 boites à moustaches assez distincte quant au positionnement de la médiane et à l'étalement (parfois à droite, parfois à gauche, parfois symétrique). On peut donc penser que l'énergie a un impact sur la note attribuée.
# - Les lipides ont une répartition par note assez similaire au détail près que les boites à moustache ont des médianes et des proportions significativement différentes (d et e ayant de grande médiane et une plus grande proportion que les autres). La aussi, cela nous laisse penser que les lipides ont un impact sur la note attribuée.
# - Les glucides ont une répartition/proportion assez proche à vue d'oeil au détail près que les médianes de d et e sont ici aussi significativement différente des autres. On peut penser que l'impact s'il est réel doit être assez minime.
# - Les protéines ayant une répartition des produits par note assez similaire, on peut penser que les protéines ne jouent aucune rôle sur la note.
# On se rappelle que le nutri score est calculé à partir des acides gras saturés, des sucres simples, du niveau d'énergie, des fibres... Cela nous laisse penser que notre analyse est juste, on vérifie cela à travers un ANOVA.

# In[32]:


for nutri in list_nutri:
    print(nutri,'/','nutrition_grade_fr :', round(eta_squared(data['nutrition_grade_fr'],data[nutri]),2))


# On voit donc que l'energie et les lipides ont un impact significatif sur la note tandis que les glucides et les protéines n'en ont aucun. Cela confirme donc notre intuition précédente.

# Les catégories d'aliments ont-elles un impact sur la quantité de glucide, protéine, lipide et énergie ?

# In[33]:


list_nutri = ['energy_100g', 'fat_100g', 'carbohydrates_100g', 'proteins_100g']
for nutri in list_nutri:
    sns.boxenplot(data=data, x=nutri, y ='pnns_groups_1', showfliers=False)
    plt.show()


# On observe pour l'énergie, les lipides, les glucides et les protéines une répartition par catégorie très hétérogène, cela nous laisse penser que les catégories ont un impact significatif sur leurs quantités et leurs répartitions. Quantifions cette observation à l'aide d'un modèle ANOVA.

# In[34]:


list_nutri = ['energy_100g', 'fat_100g', 'carbohydrates_100g', 'proteins_100g']
for nutri in list_nutri:
    print(nutri,'/','catégories :', round(eta_squared(data['pnns_groups_1'],data[nutri]),2))


# Les résultats de l'ANOVA nous permettent de confirmer notre analyse. En effet, pour l'ensemble des macro nutriments principaux ainsi que de l'énergie on note un lien évident entre les catégories d'aliments et la répartition/proportion d'énergie/protéine/lipide/glucide dans les produits. On remarque d'ailleurs que les glucides représentent le macronutriment le plus sensible au changement de catégorie. Voyons ce que cela donne si on approfondit notre analyse en prenant cette fois les sous-catégories :

# In[35]:


list_nutri = ['energy_100g', 'fat_100g', 'carbohydrates_100g', 'proteins_100g']
for nutri in list_nutri:
    print(nutri,'/','catégories :', round(eta_squared(data['pnns_groups_2'],data[nutri]),2))


# On remarque ici que pour chacun des couples une dépendance plus élevé que dans le cas des catégories. Cela signifie qu'approfondir des catégories aux sous-catégories à un intérêt lorsqu'on veut rendre nos groupes encore plus hétérogènes du point de vue des macronutriments et de l'énergie.

# Les catégories d'aliments ont-elles un impact sur le nutriscore ?

# In[36]:


sns.boxenplot(data=data, x='nutrition-score-fr_100g', y ='pnns_groups_1', showfliers=False)
plt.show()


# Les boites à moustaches que l'on voit ci-dessus sont assez différentes, on observe que les médianes sont assez diffuses (Basses avec les fruits/céréales, moyenne avec les boissons et hautes avec les collations sucrées/salées) et des répartitions changeantes (équilibrées, étandues sur la droite, étandues sur la gauche). Ces éléments nous laisse penser qu'ici aussi il existe un lien entre les catégories et la note attribuée. On vérifie cette observation par un ANOVA.

# In[37]:


print('nutrition-score-fr_100g','/','catégories :', round(eta_squared(data['pnns_groups_1'],data['nutrition-score-fr_100g']),2))


# On observe bien un dépendance entre la catégorie des aliments et leurs notes. Voyons maintenant ce que cela donne si on approfondit notre analyse par sous-catégories :

# In[38]:


print('nutrition-score-fr_100g','/','catégories :', round(eta_squared(data['pnns_groups_2'],data['nutrition-score-fr_100g']),2))


# Ici aussi, le lien décrit par l'ANOVA entre les sous-catégories et le score nutritif est significatif et ce, plus encore qu'avec les catégories. On déduit donc ici aussi que l'approfondissement en sous-catégorie ajoute de la variance inter-groupe, ce qui augmente la dépendance groupe/score nutrition.

# ## Analyse Multivariée

# ### Analyse exploratoire générale

# Dans cette partie on va :
# - Mettre en évidence les corrélations entre variable ainsi que les grandes tendances.
# - Observer le nuage de points sur les 2 plans factorielles.

# In[39]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[40]:


#Sélection des colonnes lipides, acide gras saturé, glucide, sucre, fibre et protéine du dataframe data
X = data[[
       'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g',
       'fiber_100g', 'proteins_100g'
       ]]


# In[42]:


#Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[43]:


#On vérifie que nos données sont bien normalisées
idx = ["mean", "std"]

#moyenne de 0 et ecart type de 1
pd.DataFrame(X_scaled).describe().round(2).loc[idx, :] 


# In[44]:


#On veut 4 axes principaux pour notre analyse 
n_components = 4
pca = PCA(n_components=n_components)

#On produit un ACP sur les données normalisées
pca.fit(X_scaled)


# In[45]:


#Combien de variance est expliquée par chacun de nos 4 axes ?
pca.explained_variance_ratio_


# In[46]:


#On observe ce que chaque axe principal explique comme inertie
scree = (pca.explained_variance_ratio_*100).round(2)
scree_cum = scree.cumsum().round()
x_list = range(1, n_components+1)

plt.bar(x_list, scree)
plt.plot(x_list, scree_cum,c="red",marker='o')
plt.xlabel("rang de l'axe d'inertie")
plt.ylabel("pourcentage d'inertie")
plt.title("Eboulis des valeurs propres")
plt.show(block=False)


# Il est également intéressant d'observer comment nos axes principaux (F1, F2, F3 et F4) sont construit.
# Pour cela on observe les combinaisons linéaires de nos variables initiales permettant leurs productions.

# In[47]:


pcs = pca.components_ 
pcs = pd.DataFrame(pcs)
pcs.columns = X.columns
pcs.index = [f"F{i}" for i in x_list]
pcs.round(2)


# On observe par exemple dans le tableau ci-dessus que F1 est très corrélée aux lipides (0.59), acides gras saturés (0.56), protéine (0.38) etc... mais pas corrélée aux fibres (-0.01).

# In[48]:


#On met en évidence les variables initiales les plus corrélées à nos 4 axes
fig, ax = plt.subplots(figsize=(20, 6))
sns.heatmap(pcs.T, vmin=-1, vmax=1, annot=True, cmap="coolwarm", fmt="0.2f")


# In[49]:


#fonction pour le graphe de corrélation
def correlation_graph(pca, 
                      x_y, 
                      features) : 
    """Affiche le graphe des correlations

    Positional arguments : 
    -----------------------------------
    pca : sklearn.decomposition.PCA : notre objet PCA qui a été fit
    x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2
    features : list ou tuple : la liste des features (ie des dimensions) à représenter
    """

    # Extrait x et y 
    x,y=x_y

    # Taille de l'image (en inches)
    fig, ax = plt.subplots(figsize=(10, 9))

    # Pour chaque composante : 
    for i in range(0, pca.components_.shape[1]):

        # Les flèches
        ax.arrow(0,0, 
                pca.components_[x, i],  
                pca.components_[y, i],  
                head_width=0.07,
                head_length=0.07, 
                width=0.02, )

        # Les labels
        plt.text(pca.components_[x, i] + 0.05,
                pca.components_[y, i] + 0.05,
                features[i])
        
    # Affichage des lignes horizontales et verticales
    plt.plot([-1, 1], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-1, 1], color='grey', ls='--')

    # Nom des axes, avec le pourcentage d'inertie expliqué
    plt.xlabel('F{} ({}%)'.format(x+1, round(100*pca.explained_variance_ratio_[x],1)))
    plt.ylabel('F{} ({}%)'.format(y+1, round(100*pca.explained_variance_ratio_[y],1)))

    # J'ai copié collé le code sans le lire
    plt.title("Cercle des corrélations (F{} et F{})".format(x+1, y+1))

    # Le cercle 
    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale

    # Axes et display
    plt.axis('equal')
    plt.show(block=False)


# In[50]:


#fonction pour afficher les projections des variables dans R^n 
def display_factorial_planes(   X_projected, 
                                x_y, 
                                pca=None, 
                                labels = None,
                                clusters=None, 
                                alpha=1,
                                figsize=[10,8], 
                                marker="." ):
    """
    Affiche la projection des individus

    Positional arguments : 
    -------------------------------------
    X_projected : np.array, pd.DataFrame, list of list : la matrice des points projetés
    x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2

    Optional arguments : 
    -------------------------------------
    pca : sklearn.decomposition.PCA : un objet PCA qui a été fit, cela nous permettra d'afficher la variance de chaque composante, default = None
    labels : list ou tuple : les labels des individus à projeter, default = None
    clusters : list ou tuple : la liste des clusters auquel appartient chaque individu, default = None
    alpha : float in [0,1] : paramètre de transparence, 0=100% transparent, 1=0% transparent, default = 1
    figsize : list ou tuple : couple width, height qui définit la taille de la figure en inches, default = [10,8] 
    marker : str : le type de marker utilisé pour représenter les individus, points croix etc etc, default = "."
    """

    # Transforme X_projected en np.array
    X_ = np.array(X_projected)

    # On définit la forme de la figure si elle n'a pas été donnée
    if not figsize: 
        figsize = (7,6)

    # On gère les labels
    if  labels is None : 
        labels = []
    try : 
        len(labels)
    except Exception as e : 
        raise e

    # On vérifie la variable axis 
    if not len(x_y) ==2 : 
        raise AttributeError("2 axes sont demandées")   
    if max(x_y )>= X_.shape[1] : 
        raise AttributeError("la variable axis n'est pas bonne")   

    # on définit x et y 
    x, y = x_y

    # Initialisation de la figure       
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # On vérifie s'il y a des clusters ou non
    c = None if clusters is None else clusters
 
    # Les points    
    # plt.scatter(   X_[:, x], X_[:, y], alpha=alpha, 
    #                     c=c, cmap="Set1", marker=marker)
    sns.scatterplot(data=None, x=X_[:, x], y=X_[:, y], hue=c)

    # Si la variable pca a été fournie, on peut calculer le % de variance de chaque axe 
    if pca : 
        v1 = str(round(100*pca.explained_variance_ratio_[x]))  + " %"
        v2 = str(round(100*pca.explained_variance_ratio_[y]))  + " %"
    else : 
        v1=v2= ''

    # Nom des axes, avec le pourcentage d'inertie expliqué
    ax.set_xlabel(f'F{x+1} {v1}')
    ax.set_ylabel(f'F{y+1} {v2}')

    # Valeur x max et y max
    x_max = np.abs(X_[:, x]).max() *1.1
    y_max = np.abs(X_[:, y]).max() *1.1

    # On borne x et y 
    ax.set_xlim(left=-x_max, right=x_max)
    ax.set_ylim(bottom= -y_max, top=y_max)

    # Affichage des lignes horizontales et verticales
    plt.plot([-x_max, x_max], [0, 0], color='grey', alpha=0.8)
    plt.plot([0,0], [-y_max, y_max], color='grey', alpha=0.8)

    # Affichage des labels des points
    if len(labels) : 
        for i,(_x,_y) in enumerate(X_[:,[x,y]]):
            plt.text(_x, _y+0.05, labels[i], fontsize='14', ha='center',va='center') 

    # Titre et display
    plt.title(f"Projection des individus (sur F{x+1} et F{y+1})")
    plt.show()


# In[51]:


#On affiche le cercle de corrélation
x_y = (0,1)
correlation_graph(pca, x_y, pcs.columns)


# Le cercle des corrélations nous montre que :
# - Le premier axe principal (31.1% de variance expliquée) est corrélé positivement aux lipides, acides gras saturés ainsi qu'aux protéines. De plus elle est anticorrélée aux glucides et aux sucres simples. Enfin, elle n'est pas corrélée aux fibres.
# - Le deuxième axe principal (28% de variance expliquée) est corrélé positivement aux glucides, sucres simples, lipides et acide gras saturés.
# Ainsi le premier axe principal aura tendance à scinder les produits sucrées des produits gras tandis que le deuxième axe principal nous apprendra surtout sur la quantité de nutriments que le produit possède.

# In[52]:


#Projection des variables 
X_proj = pca.transform(X_scaled)


# In[53]:


x_y = [0,1]
display_factorial_planes(X_proj, x_y)


# On observe que :
# - Les aliments très sucrées sont plus nombreux que les aliments gras (Le nuage de points est très dense à gauche mais moins dense à droite).
# - Les quantités de nutriments dans chaque produit peuvent être très variable (très sucrée, très gras, peu sucrée, peu gras, moyennement sucrée, moyennement gras etc...)

# Pour aller plus loin, on décide d'afficher le cercle de corrélation du 3ème et 4ème axes principales ainsi que leurs plans factorielles associées.

# In[54]:


#On affiche le cercle de corrélation
x_y = (2,3)
correlation_graph(pca, x_y, pcs.columns)


# On observe que :
# - Le 3ème axe principal (18.4% de variance expliquée) est fortement corrélé aux fibres et aux protéines.
# - Le 4ème axe principal (12.1% de variance expliquée) est fortement corrélé aux fibres et fortement anticorélé aux protéines.
# \
# On en déduit que le 3ème axe principal mesure la quantité de fibre/de protéine tandis que le 4ème axe principal mesure la proportion de fibre et de protéine (plus protéiné et moins fibreux ou plus fibreux et moins protéiné).   

# In[55]:


x_y = [2,3]
display_factorial_planes(X_proj, x_y)


# Ainsi on observe que les produits ont tendance à avoir une composition plutôt fibreuse que protéiné. De plus la quantité de fibre est très variable et étalée tandis que les protéines restent plus concentrée.

# On a vu dans l'analyse bivariée que les catégories pouvaient être (en partie) expliquées par leurs quantités de macronutriments. Ainsi, on peut approfondir notre ACP en sélectionnant un candidat par catégories.

# # Approfondissement

# On aimerait trouver une répartition des différentes catégories ('pnns_groups_1') sur nos plans factoriels. Cela nous permettrait d'observer les groupes proches et les regrouper en cluster afin de conclure sur la variabilité des macronutriments en fonction de la catégorie. Pour cela, on décide de construire un candidat par catégorie en prenant la moyenne de chaque macro nutriment. 

# In[56]:


#Pour chaque catégorie de 'pnns_groups_1' on construit un candidat cen prenant la moyenne par 
#catégorie de chaque macro nutriment
X = data.groupby('pnns_groups_1').mean()


# In[57]:


#On sélectionne les lipides, acides gras saturés, glucides, sucre simple, fibre et protéine
X = X[[
       'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g',
       'fiber_100g', 'proteins_100g']]


# In[58]:


#On normalise nos données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[59]:


#On vérifie que nos données sont bien normalisées
idx = ["mean", "std"]

#Moyenne de 0 et ecart type de 1
pd.DataFrame(X_scaled).describe().round(2).loc[idx, :]


# In[60]:


#On effectue une ACP à 4 composantes
n_components = 4
pca = PCA(n_components=n_components)
pca.fit(X_scaled)


# In[61]:


#On observe le pourcentage d'inertie expliquée par chaque axe
pca.explained_variance_ratio_


# In[62]:


x_y = (0,1)
correlation_graph(pca, x_y, pcs.columns)


# Ce premier plan factoriel est très interessant, en effet on observe que :
# - Le premier axe principal (35.4% d'inertie expliquée) est très corrélé aux glucides, aux fibres et aux sucres simple. 
# - Le deuxième axe principal (31.1% d'inertie expliquée) est très corrélé aux lipides et acide gras saturés.
# \
# Les glucides, fibres et sucre simple étant tous de la catégorie "sucre" on déduit que le premier axe principal mesure la quantité de sucre toute catégorie confondue. De plus, on sait que les acides gras saturés composent les lipides ainsi le deuxième axe principal mesure la quantité de gras (bon ou mauvais).

# In[63]:


#fonction pour afficher les projections des variables dans R^n 
def display_factorial_planes(   X_projected, 
                                x_y, 
                                pca=None, 
                                labels = None,
                                clusters=None, 
                                alpha=1,
                                figsize=[10,8], 
                                marker="." ):
    """
    Affiche la projection des individus

    Positional arguments : 
    -------------------------------------
    X_projected : np.array, pd.DataFrame, list of list : la matrice des points projetés
    x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2

    Optional arguments : 
    -------------------------------------
    pca : sklearn.decomposition.PCA : un objet PCA qui a été fit, cela nous permettra d'afficher la variance de chaque composante, default = None
    labels : list ou tuple : les labels des individus à projeter, default = None
    clusters : list ou tuple : la liste des clusters auquel appartient chaque individu, default = None
    alpha : float in [0,1] : paramètre de transparence, 0=100% transparent, 1=0% transparent, default = 1
    figsize : list ou tuple : couple width, height qui définit la taille de la figure en inches, default = [10,8] 
    marker : str : le type de marker utilisé pour représenter les individus, points croix etc etc, default = "."
    """

    # Transforme X_projected en np.array
    X_ = np.array(X_projected)

    # On définit la forme de la figure si elle n'a pas été donnée
    if not figsize: 
        figsize = (7,6)

    # On gère les labels
    if  labels is None : 
        labels = []
    try : 
        len(labels)
    except Exception as e : 
        raise e

    # On vérifie la variable axis 
    if not len(x_y) ==2 : 
        raise AttributeError("2 axes sont demandées")   
    if max(x_y )>= X_.shape[1] : 
        raise AttributeError("la variable axis n'est pas bonne")   

    # on définit x et y 
    x, y = x_y

    # Initialisation de la figure       
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # On vérifie s'il y a des clusters ou non
    c = None if clusters is None else clusters
 
    # Les points    
    # plt.scatter(   X_[:, x], X_[:, y], alpha=alpha, 
    #                     c=c, cmap="Set1", marker=marker)
    sns.scatterplot(data=None, x=X_[:, x], y=X_[:, y], hue=c, palette =['black', 'red', 'green', 'yellow', 'magenta'])

    # Si la variable pca a été fournie, on peut calculer le % de variance de chaque axe 
    if pca : 
        v1 = str(round(100*pca.explained_variance_ratio_[x]))  + " %"
        v2 = str(round(100*pca.explained_variance_ratio_[y]))  + " %"
    else : 
        v1=v2= ''

    # Nom des axes, avec le pourcentage d'inertie expliqué
    ax.set_xlabel(f'F{x+1} {v1}')
    ax.set_ylabel(f'F{y+1} {v2}')

    # Valeur x max et y max
    x_max = np.abs(X_[:, x]).max() *1.1
    y_max = np.abs(X_[:, y]).max() *1.1

    # On borne x et y 
    ax.set_xlim(left=-x_max, right=x_max)
    ax.set_ylim(bottom= -y_max, top=y_max)

    # Affichage des lignes horizontales et verticales
    plt.plot([-x_max, x_max], [0, 0], color='grey', alpha=0.8)
    plt.plot([0,0], [-y_max, y_max], color='grey', alpha=0.8)

    # Affichage des labels des points
    if len(labels) : 
        # j'ai copié collé la fonction sans la lire
        for i,(_x,_y) in enumerate(X_[:,[x,y]]):
            plt.text(_x, _y+0.05, labels[i], fontsize='14', ha='center',va='center') 

    # Titre et display
    plt.title(f"Projection des individus (sur F{x+1} et F{y+1})")
    plt.show()


# In[64]:


#Projection des variables (dimensions)
X_proj = pca.transform(X_scaled)


# In[65]:


#Affichage des catégories sur le premier plan factoriel
x_y = [0,1]
display_factorial_planes(X_proj, x_y, pca, labels=X.index, figsize=(20,16), marker="o")


# On observe que notre plan peut se lire en 4 parties : 
# - Les produits gras et sucrés : Sugary Snacks et Salty Snacks.
# - Les produits gras et pas sucrés : Fat and sauces et Milk and dairy Products
# - Les produits pas gras et sucrés : Cereals and potatoes
# - Les produits pas gras et pas sucrés : Beverages et Composite foods.
# \
# Avec un minimum de connaissance du domaine de l'alimentation on se rends compte que ces catégories sont cohérente. Allons plus loin en utilisant 2 méthodes de clustering : Le K-means et la CAH.

# ## Kmeans

# In[66]:


from sklearn.cluster import KMeans


# Afin d'utiliser l'algorithme du K-means efficacement, il est important de sélectionner le bon nombre de cluster. Pour ce faire, on créer une boucle testant ce que chaque cluster explique en terme d'inertie et on s'intéresse à la valeur de coude. 

# In[67]:


# Une liste vide pour enregistrer les inerties :  
intertia_list = [ ]

# Notre liste de nombres de clusters : 
k_list = range(1, 10)

# Pour chaque nombre de clusters : 
for k in k_list : 
    
    # On instancie un k-means pour k clusters
    kmeans = KMeans(n_clusters=k)
    
    # On entraine
    kmeans.fit(X_proj)
    
    # On enregistre l'inertie obtenue : 
    intertia_list.append(kmeans.inertia_)


# In[68]:


#Affichage de la courbe du nombre de cluster en fonction de l'inertie
fig, ax = plt.subplots(1,1,figsize=(12,6))

ax.set_ylabel("intertia")
ax.set_xlabel("n_cluster")

ax = plt.plot(k_list, intertia_list)


# On remarque qu'à n_cluster = 5 il y a un coude, c'est donc la valeur qui nous intéresse afin de composer nos clusters de manière optimal.

# In[69]:


#On sélectionne 5 clusters
#On instancie notre Kmeans avec 5 clusters : 
kmeans = KMeans(n_clusters=5)

#On l'entraine : 
kmeans.fit(X_proj)

#On peut stocker nos clusters dans une variable labels : 
labels = kmeans.labels_
labels


# In[70]:


#On convertit X_proj en dataframe
X_proj = pd.DataFrame(X_proj, columns = ["PC1", "PC2", "PC3", "PC4"])


# In[71]:


[0,1]
display_factorial_planes(X_proj, x_y, pca, labels=X.index, figsize=(20,16), marker="o", clusters=labels)


# On observe sur notre premier plan factoriel les 5 clusters crées en fonction des macronutriments sélectionnés :
# - Les produits laitiers ainsi que les poissons, viandes et oeuf sont regroupés ensemble.
# - Les nourritures composées, boissons et légumes/fruits sont regroupés ensemble.
# - Les snacks salés et les céréales/potatoes sont regroupés ensemble.
# - Les snacks sucrés forment une catégorie seule.
# - Les sauces et graisses forment une catégorie seule.
# \
# Pour avoir une meilleure idée des proximités entre les groupes on décide d'observer nos clusters sur un affichage 3D en affichant les 3 premiers axes principaux.

# In[72]:


#ACP 3D

# On définit notre figure et notre axe différemment : 
fig= plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

# On affiche nos points : 
ax.scatter(
    X_proj.iloc[:, 0],
    X_proj.iloc[:, 1],
    X_proj.iloc[:, 2],
    c=labels, cmap="Set1", edgecolor="k", s=40)

# On spécifie le nom des axes : 
ax.set_xlabel("F1")
ax.set_ylabel("F2")
ax.set_zlabel("F3")


# Le 3ème axe principal ajoute 20% de variance expliquée (soit au total environ 86% de variance expliquée par la représentation 3D). Cela nous permet de mieux comprendre la composition des clusters.

# ## CAH

# In[73]:


from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


# On veut maintenant faire un clustering par Classification Ascendante Hiérarchique. On utilise la méthode de Ward.

# In[74]:


#CAH avec méthode de Ward
Z = linkage(X_proj, method="ward")


# In[75]:


#Affichage du dendrogramme
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

_ = dendrogram(Z, ax=ax, labels = X.index, orientation='left')

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.ylabel("Distance.")
plt.show()


# Le dendrogramme nous affiche un clustering très proche de celui produit par le K-means. Affichons nos points en prenant un nombre de cluster identique qu'au K-means.

# In[76]:


from sklearn.cluster import AgglomerativeClustering


# In[77]:


#Création de l'algorithme du CAH à 5 clusters
cah = AgglomerativeClustering(n_clusters=5, linkage="ward")

#Fit du CAH avec nos données projetées
cah.fit(X_proj)

#Stockage des labels dans la variable labels2
labels2 = cah.labels_


# In[78]:


#Affichage de nos clusters sur le premier plan factoriel
[0,1]
display_factorial_planes(X_proj, x_y, pca, labels=X.index, figsize=(20,16), marker="o", clusters=labels2)


# Le clustering semble être identique à celui produit par le K-means, vérifions le en comparant nos valeurs avec un diagramme de Sankey.

# ![sankeydiagram.png](attachment:sankeydiagram.png)

# On observe à gauche le clustering produit par le K-means et à gauche celui produit par la CAH. Les valeurs coincident parfaitement. On peut donc penser que nos résultats sont plutôt robuste.

# ### Conclusion de l'analyse exploratoire

# L'ACP nous a permit d'identifier les grandes tendances de nos macronutriments. Ces tendances sont :
# - Le 'Gras' : composé des lipides et des acides gras saturés.
# - Les 'Sucres' : composé des glucides, sucres simples et des fibres.
# - Les Protéines.
# \
# On a également observé que l'appartenance à une des catégories (de pnns_groups_1) avait une incidence direct sur la composition des aliments.

# ## Analyse des variables 'Bons sucres', 'Bonnes graisses' et 'Protéiné'

#  Afin de conclure et de vérifier si notre idée d'application est réalisable, on décide de créer trois variables 'Bons sucres', 'Bonnes graisses', et 'Protéiné' puis avec une ACP et un clustering chercher à observer des catégories/groupes d'individus ayant une de ces qualités nutritionnelles (ou aucune d'entre elles).

# Commençons par créer nos variables : 
# - 'Bons sucres' va être la somme des glucides et des fibres (considérés comme de bons sucres) auxquelles on soustrait les sucres simples (considérés comme de mauvais sucres)
# - 'Bonnes graisses' va être les lipides auxquelle on soustrait les acides gras saturés (mauvais "gras")
# - 'Protéiné' reprenant simplement la variable Protéine

# In[79]:


#Création de nos variables
var_1 = data['carbohydrates_100g'] - data['sugars_100g'] + data['fiber_100g']
var_2 = data['fat_100g'] - data['saturated-fat_100g']
var_3 = data['proteins_100g']

#Concaténation des Series pour former un dataframe
new_var = pd.concat([var_1, var_2, var_3], axis = 1)

#On renomme les colonnes
new_var.rename(columns = {0:'Sucre', 1:'Graisses', 'proteins_100g':'Protéine'}, inplace = True)


# In[80]:


#On créer la variable X et on normalise les données
X = new_var
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[81]:


#On vérifie que nos données sont bien normalisées
idx = ["mean", "std"]
pd.DataFrame(X_scaled).describe().round(2).loc[idx, :] #moyenne de 0 et ecart type de 1


# In[82]:


#ACP à 3 composantes
n_components = 3
pca = PCA(n_components=n_components)
pca.fit(X_scaled)


# In[83]:


#Variance expliquée par chacun des axes
pca.explained_variance_ratio_


# In[84]:


#fonction pour afficher les projections des variables dans R^n 
def display_factorial_planes(   X_projected, 
                                x_y, 
                                pca=None, 
                                labels = None,
                                clusters=None, 
                                alpha=1,
                                figsize=[10,8], 
                                marker=".",
                                ):
    """
    Affiche la projection des individus

    Positional arguments : 
    -------------------------------------
    X_projected : np.array, pd.DataFrame, list of list : la matrice des points projetés
    x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2

    Optional arguments : 
    -------------------------------------
    pca : sklearn.decomposition.PCA : un objet PCA qui a été fit, cela nous permettra d'afficher la variance de chaque composante, default = None
    labels : list ou tuple : les labels des individus à projeter, default = None
    clusters : list ou tuple : la liste des clusters auquel appartient chaque individu, default = None
    alpha : float in [0,1] : paramètre de transparence, 0=100% transparent, 1=0% transparent, default = 1
    figsize : list ou tuple : couple width, height qui définit la taille de la figure en inches, default = [10,8] 
    marker : str : le type de marker utilisé pour représenter les individus, points croix etc etc, default = "."
    """

    # Transforme X_projected en np.array
    X_ = np.array(X_projected)

    # On définit la forme de la figure si elle n'a pas été donnée
    if not figsize: 
        figsize = (7,6)

    # On gère les labels
    if  labels is None : 
        labels = []
    try : 
        len(labels)
    except Exception as e : 
        raise e

    # On vérifie la variable axis 
    if not len(x_y) ==2 : 
        raise AttributeError("2 axes sont demandées")   
    if max(x_y )>= X_.shape[1] : 
        raise AttributeError("la variable axis n'est pas bonne")   

    # on définit x et y 
    x, y = x_y

    # Initialisation de la figure       
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # On vérifie s'il y a des clusters ou non
    c = None if clusters is None else clusters
 
    # Les points    
    # plt.scatter(   X_[:, x], X_[:, y], alpha=alpha, 
    #                     c=c, cmap="Set1", marker=marker)
    sns.scatterplot(data=None, x=X_[:, x], y=X_[:, y], hue=c, palette = ['black', 'red', 'green', 'grey'])

    # Si la variable pca a été fournie, on peut calculer le % de variance de chaque axe 
    if pca : 
        v1 = str(round(100*pca.explained_variance_ratio_[x]))  + " %"
        v2 = str(round(100*pca.explained_variance_ratio_[y]))  + " %"
    else : 
        v1=v2= ''

    # Nom des axes, avec le pourcentage d'inertie expliqué
    ax.set_xlabel(f'F{x+1} {v1}')
    ax.set_ylabel(f'F{y+1} {v2}')

    # Valeur x max et y max
    x_max = np.abs(X_[:, x]).max() *1.1
    y_max = np.abs(X_[:, y]).max() *1.1

    # On borne x et y 
    ax.set_xlim(left=-x_max, right=x_max)
    ax.set_ylim(bottom= -y_max, top=y_max)

    # Affichage des lignes horizontales et verticales
    plt.plot([-x_max, x_max], [0, 0], color='grey', alpha=0.8)
    plt.plot([0,0], [-y_max, y_max], color='grey', alpha=0.8)

    # Affichage des labels des points
    if len(labels) : 
        # j'ai copié collé la fonction sans la lire
        for i,(_x,_y) in enumerate(X_[:,[x,y]]):
            plt.text(_x, _y+0.05, labels[i], fontsize='14', ha='center',va='center') 

    # Titre et display
    plt.title(f"Projection des individus (sur F{x+1} et F{y+1})")
    plt.show()


# In[85]:


x_y = (0,1)
correlation_graph(pca, x_y, new_var.columns)


# In[86]:


#Projection des variables (dimensions)
X_proj = pca.transform(X_scaled)


# In[87]:


# Une liste vide pour enregistrer les inerties :  
intertia_list = [ ]

# Notre liste de nombres de clusters : 
k_list = range(1, 10)

# Pour chaque nombre de clusters : 
for k in k_list : 
    
    # On instancie un k-means pour k clusters
    kmeans = KMeans(n_clusters=k)
    
    # On entraine
    kmeans.fit(X_proj)
    
    # On enregistre l'inertie obtenue : 
    intertia_list.append(kmeans.inertia_)


# In[88]:


fig, ax = plt.subplots(1,1,figsize=(12,6))

ax.set_ylabel("intertia")
ax.set_xlabel("n_cluster")

ax = plt.plot(k_list, intertia_list)


# Le coude se situe à 4 clusters, on choisit donc ce nombre.

# In[89]:


#On sélectionne 4 clusters
#On instancie notre Kmeans avec 4 clusters : 
kmeans = KMeans(n_clusters=4)

#On l'entraine : 
kmeans.fit(X_proj)

#On peut stocker nos clusters dans une variable labels : 
labels = kmeans.labels_
labels


# ![t%C3%A9l%C3%A9charger%20%2813%29.png](attachment:t%C3%A9l%C3%A9charger%20%2813%29.png)

# On observe nos 4 clusters :
# - 'Bons sucres !' en rouge.
# - 'Très Protéiné !' en noir.
# - 'Bonnes graisses !' en gris.
# - 'Pas de qualités nutritionnelles particlulières !' en vert.

# # Conclusion

# On a pu voir que :
# - Il y avait une nécéssité pour les clients de sélectionner des produits de bonne qualité. (Grosse proportion de mauvaise note).
# - Les produits ont tendance à avoir une quantité de glucide > lipide > protéine. Cela montre que l'accès au macronutriments est désequilibré dans le monde de la nutrition (mais chacune est tout autant indispensable). Ce résultat est est moins vrai si on considère les "bons sucres" plutôt que les glucides ou les "bonnes graisses" plutôt que les lipides. (En ce sens, trouver des protéines est presque aussi compliqué que de trouver des bonnes graisses ou de bon lipides).
# - Le choix de "Bons sucres", "Bonnes graisses" et "Protéiné" est pertinent (car les 3 sont anticorrélés et les variables ont du sens dans le cadre d'une meilleure alimentation).
# - On a pu mettre en évidence l'existence de 4 clusters schematisant bien nos différentes catégories.
# \
# Au regard de tous ces éléments on peut dire que le projet a du sens et qu'il est réalisable.
