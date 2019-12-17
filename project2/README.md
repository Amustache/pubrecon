# **pubrecon** - The _place_ (physical and conceptual) of advertising in video game magazines

CS-433 - Machine Learning - EPFL — Martin Jaggi & Rüdiger Urbanke

Project by [Hugo "Stache" Hueber](mailto:hugo.hueber@epfl.ch) & [Florine Réau](mailto:florine.reau@epfl.ch)

Supervised by [Yannick Rochat](yannick.rochat@unil.ch) & [Magalie Vetter](magalie.vetter@chartes.psl.eu)

---

## Introduction

Our project consists in the recognition of advertisment in French videogame magazines between the 80s and the 2010s. We implemented a R-CNN and worked on the Gen4 magazine for that purpose.

More information can be found in the [projet report](./report/report/report.pdf).

## Installation

### Structure

We recommand the following structure for your project, which should be the basic example when pulling the repo:
```
project/
    data/
        in/
        out/
    notebooks/
    scripts/
```

### Packages

1. Create a new environment (e.g. `conda create --name pubrecon python=3.7 -y`, cf. [Conda](conda.io)) and activate it (e.g. `conda activate pubrecon`).
2. Run `pip install --upgrade pip`.
3. Install the package using `pip install -e /root/of/the/repo/`.
4. Grab  some coffee.
5. You can now use **pubrecon**.

## Preparation and configuration

### Data preparation

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

Put all the data in `data/in`. That's it.

### Configuration

TODO

## How to use it?

(Optional) Install [Jupyter](https://jupyter.org/) with `pip install jupyter` for quick prototyping.

## Contact

TODO

## Known bugs

TODO

## Acknowledgments

Thanks to Yannick Rochat and Magalie Vetter for supervising our project. Thanks to Johan Paratte for the precious help.

## Changelog

[LOG.md](./LOG.md)