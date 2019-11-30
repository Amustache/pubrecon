# LOGBOOK

## 2019.11.30
- Exploration des datas.
- Les inputs nous sont données sous format XML. Il faut donc réaliser une conversion préliminaire XML > Format que l'on souhaite.
- L'idée est de réaliser une dataframe pandas pour faciliter l'exploitation, puis de sauvegarder le fichier sous forme de pickle réutilisable. Il faut ainsi créer un script de conversion.
    - Script de conversion créé sous la forme d'un notebook.
- Une fois les datas exploitables, il faut... les exploiter.

**TODO**

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
