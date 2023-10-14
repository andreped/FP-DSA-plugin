# FastPathology Digital Slide Archive (FP-DSA) extension

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8124068.svg)](https://doi.org/10.5281/zenodo.8124068)

**DISCLAIMER:** This is a work in progress. When I have the plugin properly working and stable, I will make a public docker image, and make a release here.

Note that this setup has been tested against Ubuntu 18.04 and 20.04. It should likely work on Windows 10, but on macOS there is a conflict between OpenGL/OpenCL resulting in a `RuntimeError: clGetPlatformIDs`.

Click `watch` in the top right if this project interests you and want to be updated when it is ready to be tested.

<p style="text-align: center;">
  <img src="assets/snapshot_nuclei.png" width="45%" style="background-color:black">
  <img src="assets/snapshot_classification.png" width="45%" style="background-color:black">
</p>


## 🎊 Features

The software is still in development, but some key features have been added such as:

* Uses pyFAST backend to run FAST pipelines (FPLs)
* Developed generic backend tool for running FPLs through the UI and convert predictions to the JSON format
* Ability to run patch-wise classification and segmentation models
* Render classification predictions as heatmaps and segmentation objects as boundaries
* Store predictions in database, access, download, and modify these through the UI


## 🐳 Requirements

DSA needs to be installed. Follow the instructions [here](https://github.com/DigitalSlideArchive/digital_slide_archive/tree/master/devops/dsa) on how to do so.

In addition, docker need to be setup such that it works with pyFAST. For that I strongly recommend installing Docker desktop. You might also need to install the nvidia docker to make it work properly:

```
sudo apt update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```


## 💻 Installation

Clone the repository:
```
git clone https://github.com/andreped/FP-dsa-plugin.git
```

Build the docker image for the plugin:
```
cd dsa/
docker build -t fastpathology .
```

To add the plugin to DSA, choose `Upload new Task` under `Slicer CLI Web Tasks` in the DSA web UI, and write `fastpathology:latest` and click `Import image`. The plugin can then be used from the Analysis Page.


## 👏 Acknowledgements

The core was built based on [pyFAST](https://github.com/smistad/FAST), and the plugin was inspired by the plugins made for [MONAILabel](https://github.com/Project-MONAI/MONAILabel/tree/main/plugins/dsa) and [HistomicsTK](https://github.com/DigitalSlideArchive/HistomicsTK/tree/master/histomicstk/cli). Conversion of pyFAST's pyramidal TIFF annotations to HistomicsTK's JSON annotations was enabled using [OpenCV](https://github.com/opencv/opencv).

The plugin was made for the [Digital Slide Archive](https://github.com/DigitalSlideArchive/digital_slide_archive) which has developed an open and extremely robust and user-friendly web solution for archiving, visualizing, processing, and annotating large microscopy images. Building our methods on top of DSA was done with ease and credit to the developers such as [manthey](https://github.com/manthey) and [dgutman](https://github.com/dgutman) for addressing any issue and concerns we had at impressive speed!


## ✨ License

The plugin has [MIT-License](https://github.com/andreped/FP-dsa-plugin/blob/main/LICENSE).

Note that the different components used have their respective licenses. However, to the best of our knowledge, all dependencies used have permissive licenses with no real proprietary limitations.


## 🔬 Citation

If you found this project relevant for your research, consider citing it by:
```
@software{pedersen2023fp_dsa_plugin,
  author       = {André Pedersen},
  title        = {andreped/FP-DSA-plugin: v0.0.1},
  month        = jul,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v0.0.1},
  doi          = {10.5281/zenodo.8124068},
  url          = {https://doi.org/10.5281/zenodo.8124068}
}
```
