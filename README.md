<h1 align="center">⚠️ This repository is not maintained anymore. ⚠️</h1>

# **pubrecon** - The _place_ (physical and conceptual) of advertising in video game magazines

From the course CS-433 - Machine Learning - EPFL — Martin Jaggi & Rüdiger Urbanke, cleaned for public publication.

Made by Stache (me!)

Supervised by [Yannick Rochat](yannick.rochat@unil.ch) & [Magalie Vetter](magalie.vetter@chartes.psl.eu)

Available on [GitHub](https://github.com/Amustache/pubrecon), original from [GitHub](https://github.com/Amustache/ML-2019/edit/master/project2/).

---

## 0. Introduction

Our project consists in the recognition of advertisment in French videogame magazines between the 80s and the 2010s. We implemented a R-CNN and worked on the Gen4 magazine for that purpose.

More information can be found in the projet report (on demand).

Please follow this `README.md` to make the package work.

## 1. Installation

We assume that you have `python==3.7` and `pip>=19.3` installed and working. You may experience issues if it is not the case.

### 1a. Structure

We recommend the following structure for your project, which should be the basic example when pulling the repo:
```
project/
    data/
        in/
        out/
    notebooks/
    scripts/
```

### 1b. Packages

1. (Optional) Create a new environment (e.g. `conda create --name pubrecon python=3.7 -y`, cf. [Conda](https://conda.io)) and activate it (e.g. `conda activate pubrecon`).
2. Run `pip install --upgrade pip`.
3. Clone this repo.
3. Install the package using `pip install -e /root/of/the/repo/`.
    - This will install several packages, namely `matplotlib`, `tensorflow`, `opencv-python`, `opencv-contrib-python`, `keras`, `numpy`, `tqdm`, `pandas`, `sklearn`.
4. Grab  some coffee.
5. You can now use **pubrecon**.

## 2. Preparation and configuration

### 2a. Data preparation

The annotation tool used is [LabelImg](https://github.com/tzutalin/labelImg). They are saved as XML files in PASCAL VOC format, the format used by ImageNet, which coincidentally is the basis for our weights.

Images must be JPEG, and an additional XML file with at least the following fields should be found as well:
```
<annotation verified="yes">
        <filename>FILENAME</filename>
        <object>
                <name>CLASS_NAME</name>
                <bndbox>
                        <xmin>XMIN</xmin>
                        <ymin>YMIN</ymin>
                        <xmax>XMAX</xmax>
                        <ymax>YMAX</ymax>
                </bndbox>
        </object>
        ...
</annotation>
```
Where each annotation (bbox) should have its own `<object>`.

Put all the data in `data/in`. That`s it.

**Tip**: You can find some labelled data [here](https://drive.google.com/drive/folders/1pi9p1SybvlIZ3qT85SlPLmxzexusPgNT?usp=sharing)!

### 2b. Configuration

Configuration happens in the [`config.cfg`](./config.cfg) file. ~~You can load it using ???~~ Simply copy/paste it at the beginning of your project if needed.

## 3. How to use?

Select one of the following options to use the package.

### 3a. Jupyter Notebook

- (Optional) Install [Jupyter](https://jupyter.org/) with `pip install jupyter` for quick prototyping.
- You can use [this worked example](./notebooks/3.%20Worked%20example.ipynb).

### 3b. `run.py`

Not working for now.

### 3c. Go wild

Use the files of your choices and follow these steps:
1. Import the revelant classes using `from pubrecon.data import DataFrame, ImagesData` and `from pubrecon.model import RCNN`.
3. Define a [`DataFrame`](./pubrecon/data.py#L11), then use `prepare_data`.
4. Define a [`ImagesData`](./pubrecon/data.py#L123), then use `prepare_images_and_labels`.
5. Define a [`RCNN`](./pubrecon/model.py#L20), `train` it and check its metrics with `history`.
6. Finally, use the model's `test_image` to labelize a new page.

## Contact

Basically [contact me on Telegram](https://t.me/Stache).

## Known bugs / TODO

### Bugs

- If `tensorflow-gpu` is installed on the same environment, the model does not work.
- Suboptimal utilisation of RAM.

### TODOs

- Save `imagesdata.pickle`.
- Use GPU.
- Real Config thing-ly.

## Acknowledgments

Thanks Yannick Rochat and Magalie Vetter for supervising our project. Thanks Johan Paratte and Yoann Ponti for their input. Thanks Louis Vialiar and Camille Montemagni for the proofreading and testing.

## Changelog

[LOG.md](./LOG.md)
