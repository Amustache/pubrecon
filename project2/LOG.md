# LOGBOOK

## 2019.12.19
- Bla bla rush final okay bon.
- For the record: j'ai essayé talos et hyperas mais ça marche moyen avec mon implémentation.
- Faut explorer https://github.com/fizyr/keras-retinanet aussi.

## 2019.12.18
- Test du modèle sur un autre ordinateur. On se rend compte que si tensorflow-gpu est disponible, le modèle ne fonctionne pas.
    - cf. https://github.com/matterport/Mask_RCNN/wiki notamment.
- Nettoyage des notebooks pour ne garder qu'un exemple utile.
- Fix de dependencies et de typos, ainsi qu'une GROSSE erreur dans le modèle.
- On laisse tourner le learning "pour voir".
- Prochaines étapes importantes :
    - Cross-validation, hyperparameters, toussa.
    - Tester et retester que le package fonctionne sur un ordinateur lambda.
    - (Optionnel) Implémenter le GPU (ja-mais on aura le temps).
    - Dump des liens qui trainent, à lire et à trier : https://pastebin.com/jBDxGhhH (oui c'est long)
    - ! Terminer le rapport !
    - Mettre quelques datas disponibles en ligne pour que les gens puissent tester.
    - Rappel des trucs à améliorer pour le PDF/code d'après le projet 1:
        - cross-validation is used to tune the hyperparameters
        - you do not have any test vs train leanring curves
        - you do not tune the hyperparameters
        - [do preprocessing]
        - you do not compare any models
        - you do not have baselines (=What is the hypothesis?)
- Ajout du mode batch pour accélérer le modèle.

## 2019.12.17
- Travail sur le rapport.
- Travail sur le packaging.
    - Il devrait être terminé.
- Résultats du modèle pour 100 samples et 10 epochs, 76.6% d'accuracy.
- Question sur la vitesse.
    - Il faudrait voir pour optimiser le selective search.
- Travail sur le côté "tuto".

## 2019.12.16
- Travail sur le rapport.
- Travail sur le packaging.

## 2019.12.15
- "Je _crois_ que j'ai un truc qui fonctionne."
- (Petit/Gros) Résumé :
    - Je commence par récupérer toutes les photos/labels et je les mets dans une dataframe pandas. Je sauvegarde le tout pour pas se faire ***** à chaque fois.
    - Pour chaque image, je récupère des "areas of interest" avec le selective search de cv2.
    - A partir de ces aoi, je compare avec le ground truth ("les vrais labels") pour voir si ça colle suivant un threeshold. Si c'est le cas, on l'ajoute à notre liste de labels, sinon, c'est du background.
    - Ensuite, je construis mon modèle. VGG16 pretrained avec des poids de ImageNet, puis je pars du principe que mon réseau est déjà entrainé correctement (flaw: on a pas mal de texte, donc pas forcément que des images), donc je ne réentraine que les derniers layers.
    - Ensuite, ça part en entrainement, avec 10% de tests.
    - "Pour être honnête j'utilise un RCNN car j'ai réussi à le faire fonctionner, et VGG dans la structure du RCNN car c'est disponible ^^" - https://www.quora.com/What-is-the-VGG-neural-network
- Totalité des données disponibles: environ 4Go.
    - Par contre du coup c'est la mort pour le learning, y'a bien trop de données et pas assez de RAM.
    - Réflexion pour utiliser Google Collab ou Google Cloud.
    - Utiliser du Python "pur" au lieu du Notebook.
    - Optimiser la structure, particulièrement le selective search.
- Début du packaging.

## 2019.12.13
- Une présentation "finale" est possible, cela peut être cool d'y aller.

## 2019.12.10
- Objectifs :
    - Terminer le modèle, de l'entraîner sur Gen4.
    - Avancer le rapport.

## 2019.12.09
- Gen4 entièrement labélisé et disponible.
- ResNet fonctionnel disponible, mais à tester.
- "Je note ça là sinon je vais oublier : il faudra parler du fait que nos données brutes n'ont pas de texte intégré, que des solutions existent, mais que ce n'était pas le but"

## 2019.12.06
- More data available thanks to Magalie.
- On va restreindre notre étude sur Gen4, car pas le temps de labéliser plus de trucs.

## 2019.12.04
- Les classes sont disponibles.
    - Il faut faire attention de ne pas avoir un truc dépendant du texte, mais seulement du format du texte, car on fait de la reconnaissance d'image, pas de la contextualisation de texte.

## 2019.12.03
- More data available thanks to Magalie.

## 2019.12.02
- Il peut être intéressant de contacter Boris Krywicki (Université de Liège).
        
## 2019.12.01
- Travail sur l'implémentation de keras-frcnn.
- Il nous reste moins de trois semaines, y'a du taf.
- Je vais tenter de terminer l'implémentation du modèle pré-existant. Derrière, il faudra le réimplémenter par nous-même si ça fonctionne.
    - Ajout de `https://github.com/kbardool/keras-frcnn.git` comme submodule.
- **Note** Selon les implémentations, les bndboxes ne sont pas définies de la même manières (soit par position relative (pixels) soit c'est mappé sur [0;1]).
- A priori, notre travail est """exclusif""", dans le sens on n'a pas connaissance de travail similaire.
    - https://www.konbini.com/fr/inspiration-2/30ans-publicites-jeux-video/.
- Le rapport a été bien avancé avec les "grosses parties" présentes.
- **Note** Lors de la comparaison des résultats, il faudra bien faire attention d'utiliser les mêmes méthodes d'évaluation que dans le paper d'origine.
    - Paper en question: https://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf
- On a le p'tit script de conversion pour adapter les XML à nos modèles.
- On travaille sur le FerRCNN, mais c'est CHAUD.
- On travaille sur le rapport.
- Réflexion au niveau des classes.
    - Au niveau des classes, j'ai p't'être un peu peur que ce soit trop large, mais aucune idée
    - Like, on va peut-être devoir sélectionner un sous-groupe des classes que tu as proposé
    - Plus la redondance, si t'as une classe trop "large" (=publicité pleine page) sachant que le but c'est reconnaître une publicité, ben, le learning risque d'avoir du mal
    - Le but c'est "Ah, y'a une grosse image et peu de texte => proba(pub) élevée" vs. "Ah, y'a du texte et peu d'image => proba(pub) basse"
    - Alors que si on a toute la page classifiée comme "c'est une pub", j'ai aucune idée de l'influence
    - Après, c'est totalement une hypothèse !
    - Une autre solution qui existe, c'est effectivement de dire "ça c'est pub" "ça c'est pas pub" et de laisser le modèle faire ses propres classes
    - Idéalement on tente les deux

## 2019.11.30
- Exploration des datas.
- Les inputs nous sont données sous format XML. Il faut donc réaliser une conversion préliminaire XML > Format que l'on souhaite.
- L'idée est de réaliser une dataframe pandas pour faciliter l'exploitation, puis de sauvegarder le fichier sous forme de pickle réutilisable. Il faut ainsi créer un script de conversion.
    - Script de conversion créé sous la forme d'un notebook.
- Une fois les datas exploitables, il faut... les exploiter.
    - En gros, utiliser https://github.com/kbardool/keras-frcnn.
        - **Note** Un trick que j'ai appris c'est d'exploiter un modèle déjà existant, de "couper" les derniers layers, et de retrain juste sur ces derniers layers. 'peut nous faire gagner du temps.

## 2019.11.26
### Discussion avec Yannick
- Faire attention au niveau de la chronographie
- Pas même classification suivant la période
- Regarder dans le zip préliminaire
- Ce que l'on a c'est une hypothèse préliminaire, changement de direction car pas obligation
- Attente mutuelle
- Prochaines étapes :
    - Reconnaissance des datas
    - Identifier le corpus
    - Toy modèle
    - Voir ce que ça donne
- Prévoir un calendrier des étapes
- Vérifier ce que l'on a le droit de réutiliser dans le cadre du cours
- Avoir un premier proto
- Iterer sur les erreurs

## 2019.11.25
- Réception des premières datas 🎉

## 2019.11.24
- Objectifs de la semaine prochaine (proposé) :
    - Commencer et bien avancer le rapport.
    - Travailler sur le modèle proposé cf. https://www.analyticsvidhya.com/blog/2018/11/implementation-faster-r-cnn-python-object-detection/.
    - Etudier et comparer d'autres modèles.

## 2019.11.21
- Il faut réduire le corpus.
- Magalie: "Si on prend par exemple Tilt, Joystick hebdo, Joystick (que j'avais très bizarremet oublié dans la bdd) et Gen4 sur 88-98 sans les hors-séries on a 326 numéros numérisés (il manque 3 Gen4), avec une trentaine de numéros à labelliser avec 15/20 classes en comptant 120 pages en moyenne  j'aurai 3600 pages à labelliser, ce qui devrait me prendre une semaine je pense"
- 

## 2019.11.20
- Arrêt de la problématique technique: Classification du contenu de pages de magazines.
- Arrêt de la problématique de recherche: La _place_ (physique et conceptuelle) de la publicité dans les magazines de jeu vidéo.
    - Possibilité d'ouverture : La place du contenu éditorial par rapport à la publicité.

## 2019.11.19
### Yannick
- "Donc en effet, vous allez construire la problématique en croisant les idées de Magalie avec les miennes avec ce qui vous branche en général. Pour ça, je vous recommande d'aller flâner un peu dans les pages de ces magazines, de vous y perdre quelques heures. Certains magazines sont beaucoup plus célèbres que d'autres, mais ce peut aussi être l'occasion de faire une analyse de la presse de seconde zone. Tout est possible, depuis l'exploratoire (où la problématique devient surtout axée méthodo) à la véritable question de recherche en sciences humaines (où la méthodologie n'est plus une finalité)."
- "Essayez d'avoir une piste aujourd'hui ou demain, et d'avoir une idée claire d'ici la fin de la semaine ;-)"
- Lecture obligatoire: https://orbi.uliege.be/bitstream/2268/215516/1/DozoKrywickiArmandColin.pdf
- Lecture recommandée: https://goutarcade.hypotheses.org/80
- Attention, pas tous les titres sont correctement scannés/trouvables.

### Stache
- Après m'être un peu baladé sur AbandonWareMag, il s'est avéré que pamal de "morceaux choisis" n'étaient pas forcément complets... Donc, pas forcément facile à traiter
- En revanche, parmi certaines collection, c'est chouette, car il est possible de bien s'amuser !
- Il sera par contre peut-être difficile de réaliser des études sur des magazines "récents", dans le sens que, soit le copyhammer est présent, soit le magazine n'est pas encore disponible en ligne
- Quelque chose qui resort, c'est qu'au niveau des publicités, ce qui prime, c'est la pleine page
- Un truc qui pourrait être intéressant à étudier, c'est de voir la "place" de la publicité. Plutôt au début ? Plutôt à la fin ? La quantité ? etc
- Donc voilà, est-ce que quelque chose dans la veine de "De l'évolution de la place de la publicité dans la presse spécialisée des années 90 aux années 2010" vous irait ?
- (Place littérale comme figurée, donc)
- *Pourquoi cette période ?*
    - Compliqué de trouver des références que l'on peut suivre.
    - Après 2010, soit copystrike, soit pas d'exemplaire.
    - Comme cas d'études, on peut par exemple utiliser "Player One" ou "Joystick".
- *Pourquoi la place ?*
    - "Type" de publicité est assez standardisé, avec une (deux) pleine(s) page(s), pas régulièrement des encarts spécifiques, et semble typé.
- Notre partie "ML" serait la classification "Pub"/"Pas pub", et notre partie "ADA" serait la place de ladite pub.

## 2019.11.18
- Choix du format à définir (xml ? json ?)
- *Discussion avec Yannick*
    - Comment faire un "paper":
        - Poser une question de recherche;
            - -> Intro
        - State-of-the-art, qu'est-ce qui existe
            - -> Parler de ce que c'est
            - -> Décrire les éléments (ML, DeepL, etc.) et analyser les discours, remettre en contexte, etc.
        - Méthodologie par rapport à la problématique
        - Analyse
        - Résultats
        - Conclusion
    - Profiter du site https://abandonware-magazines.org/
        - Scans de magazines
        - URLs et noms cohérents, mais pas toujours
            - -> Besoin d'un peu de data concierge
        - Possibilité de réaliser un lien avec les "collecteurs amateurs"
    - Réfléchir à quoi faire :
        - Toute l'histoire d'un magazine ?
        - D'une époque ?
        - Histoire de la publicité ?
        - Idée éditoriale ?
        - Signature de l'évolution d'un magazine ?
        - Classifier les magazines en fonction de l'époque ?
    - Comment classifier tout le contenu ?
        - Extraire du texte (OCA) ou pas ? (cf. Google Vision)
    - Prochaines étapes :
        - Il faut explorer le site AbandonMag (https://abandonware-magazines.org/) pour explorer quelques magazines, et trouver le type de corpus que l'on souhaite.
        - Il faut ensuite réfléchir à notre corpus, puis à notre problématique.
- Mise en place structure initiale repo.

## 2019.11.15
- "But du jeu technique" : "classifier tous les éléments d'un magazine en étant le plus fin possible"
    - "L'idée est d'avoir une organisation hiérarchique de classes pour aller de plus en plus finement"

## 2019.11.13
- Newcommer: Magalie Vetter, qui va nous supervier avec Yannick.
- "Une problématique plus large et connexe qui pourrait peut-être être intéressante consisterait à trouver des moyens de distinguer les différentes parties d'un magazines automatiquement par exemple"
- But du mémoire de Magalie : "détecter la publicité pleine page de la publicité encart dans les magazines de jeu vidéo, afin de récupérer des données quantitatives"
- Potentialités :
    - "détection des différentes parties d'un magazine" ?
    - L'évolution de la pub ou de la ligne éditoriale sur une magazine pour toute sa durée ?
    - De plusieurs magazines sur un temps réduit ?
    - Juste des images ?
    - Extraire et classifier les images ?
    - "étudier l'impact des annonceurs dans des magazines de 1988 à 1998" (pbmt de Magalie)

## 2019.11.11
- Rencontre préliminaire avec Yannick Rochat.
- Discussion des différents axes possibles.
    - Classification publicités jeux vidéo;
    - Détecter type d'articles;
    - Presse en générale;
    - DHLab: journaux ou écriture manuscrite;
    - Détecter séquences;
    - Speedrun TAS ou pas ?
