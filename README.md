# projet ML partie 2

![](https://wtf.roflcopter.fr/pics/5KbcRaep/fR5K7CrV)


## Installation et lancement

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16jhYvrql9nC64A-nEGbjp2hJ4PK2Juw6#scrollTo=R7r5LOl4CprB)

### sur colab

Avant de continuer s'assurer d'avoir installer les libraries suivantes et pour `matplotlib` la version `3.1.1` qui semble la seule fonctionner chez moi pour générer les graphiques.

### en local

[requirements](./requirements.txt)

si projet importé depuis le lien github. `git clone https://github.com/uNouss/projetML; cd projetML`.

L'installation se fait comme suit:

1. création d'un nouvelle environnement virtuelle  nommé `.venv` (facultative): `python3 -m venv .venv`; suivi de son activation: `source .venv/bin/activate`.
2. installation des libraries: `pip install -r requirements.txt` si un nouvelle environnement est crée sinon `python3 -m pip install -r requirements.txt`.

3. une fois les installations faites lancer **NNI** comme suit: `nnictl create --config src/config.yml` et ouvrir le lien depuis en sortie de cette commande. Il lancera le script python situé dans `src/mnist-fashion.py` avec le fichier contenant l'espace de recherche `src/search_space.json` dans lequel sont défini les hyperparamètres.


## Définition de l'espace de recherche
[espace de recherche](src/search_space.json)
## Fichier de configuration
[fichier de configuration](src/config.yml)
## Code python
[script python](src/mnist-fashion.py)
