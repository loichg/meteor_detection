
# Meteorix

## Motion Vector Extractor (MVE) - Détection de Météores
 
### Introduction
Ce projet implémente un algorithme de détection des météores basé sur l'extraction et le filtrage des vecteurs de mouvement à partir de vidéos compressées en **H.264** ou **MPEG-4**. Il s'inscrit dans la mission **Meteorix**, un CubeSat universitaire dédié à l'étude des météoroïdes et des débris spatiaux.

L'objectif est d'utiliser l'information contenue dans la compression vidéo pour identifier les météores de manière efficace.

### Fonctionnalités
- Extraction des **vecteurs de mouvement** à partir de vidéos compressées (H.264, MPEG-4)
- Filtrage des vecteurs pour isoler les météores
  - Suppression du bruit par un **seuil de norme**
  - Regroupement des vecteurs proches avec **Nearest Neighbors**
  - Sélection des trajectoires rectilignes
- Utilisation de **FFMPEG** et de l'outil **Motion Vector Extractor (MVE)**
- Encodage des vidéos avec **accélération GPU**

### Utilisation
#### Filtrage des Vecteurs
Appliquez les filtres pour isoler les météores :
```bash
python3 src/mvextractor/__main2__.py input.mp4 --dump
```
#### Compression Vidéo
Pour compresser des images ppm en une vidéo mp4 :
```bash
ffmpeg -hwaccel cuda -framerate ? -i dossier/image%d.ppm -c:v ? -preset ? -pix_fmt ? -b:v ? -g ? output.mp4 -loglevel verbose
```
- -c:v h264_nvenc (H.264) -c:v hevc_nvenc (H.265)
- -framerate : Nombre d'image par seconde (par défaut, 25)
- -preset : Modifie la compression mais change la qualité en conséquent (plus le preset est faible, meilleure la compression sera)
- -g : Change la fréquence des images clés (par défaut, framerate*10)
- -b:v : Change le bitrate (constant)
- -cq 23 -rc vbr : Adapte la compression selon la qualité de l'image et de donner plus de bits aux parties en mouvements
- -pix_fmt : format des pixels (yuv420p, yuv444p, gray)

## Exemples de Résultats
<p align="center">
  <img src="mve/images/pasfiltre.jpg" width="45%" />
  <img src="mve/images/filtre.jpg" width="45%" />
</p>

## FlowNet sur em780 (Python 3.12.3)

Ce projet implémente FlowNet pour le calcul et la visualisation du flot optique à l'aide de PyTorch.

### 🛠 I. Installation et Configuration

#### 1. Clonage et Modèles
**Code source :** Cloner le projet depuis [https://github.com/paul-pp/flownet](https://github.com/paul-pp/flownet) dans le répertoire `FlowNetPytorch`.
**Modèles pré-entraînés :** Télécharger les fichiers `.pth` depuis [ce Google Drive](https://drive.google.com/drive/folders/16eo3p9dOvmssxRoZCmWkTpNjKRzJzn5) et les placer dans `trained_model`.

#### 2. Préparation des données
* Créer un répertoire `emtest` pour vos images.
* Les images d'une paire doivent provenir de la même vidéo, avoir les mêmes dimensions et être nommées ainsi : `<nom>1.<ext>` et `<nom>2.<ext>`.

#### 3. Environnement Virtuel
Depuis votre répertoire de travail sur la machine :

```bash
python3 -m venv venv-flownet
source venv-flownet/bin/activate
pip3 install -r FlowNetPytorch/requirements.txt
```
*(Nécessite Python 3.12.3, PyTorch 2.5.1, et les librairies listées dans le document).

### II. Lancement de l'Inférence

Pour calculer le flot optique, utilisez le script `run_inference.py`:

```bash
# Commande standard (GPU par défaut)
python3 FlowNetPytorch/run_inference.py /chemin/vers/emtest trained_model/flownets_EPE1.951.pth

# Forcer l'utilisation du CPU
python3 FlowNetPytorch/run_inference.py /chemin/vers/emtest trained_model/flownets_EPE1.951.pth -c
```

#### Options disponibles :
* `-v (--output-value)` : `raw` (fichier .npy), `viz` (image png couleur), ou `both` (par défaut).
* `-g / -c` : Force l'utilisation du GPU ou du CPU.

### III. Visualisation par Vecteurs

Une fois l'inférence terminée en mode `raw`, vous pouvez visualiser le flot avec des flèches:

```bash
python3 FlowNetPytorch/visu_vect.py -f /chemin/vers/emtest --arrow_size 1 --arrow_segmentation 2
```
* `--arrow_size` : Règle la taille des flèches.
* `--arrow_segmentation` : Règle l'espacement entre les flèches.

### IV. Génération de Dataset de Synthèse

Il est possible de générer un jeu de données d'entraînement à partir de banques d'images (`bg` et `motif`):
```bash
python3 FlowNetPytorch/gen_met.py --nb_paires 1000
```
Le résultat sera stocké dans `dataset_syn` avec une arborescence compatible `mpi_sintel_clean`.

### V. Entraînement du Modèle

Pour entraîner votre propre modèle sur un dataset de type Sintel:
```bash
python3 FlowNetPytorch/main.py /chemin/vers/dataset_syn -b8 -j8 -a flownets --dataset mpi_sintel_clean --epochs 60
```

* `-b8` : Taille du batch (8 paires).
* `-j8` : Nombre de workers pour la parallélisation.
* `--epochs 60` : Nombre d'itérations complètes sur le dataset.

## Auteurs
Projet réalisé par Loïc Huang, Anès Abdou, Cyprien Renaut, Alex Faucheux et Paul Poupeau dans le cadre du cursus MAIN4 à Polytech Sorbonne.

