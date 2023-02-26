# FAST-Pathology Digital Slide Archive (DSA) extension

**DISCLAIMER:** This is a work in progress. When I have the plugin properly working and stable, I will make a public docker image, and make a release here.

Click `watch` in the top right if this project interests you and want to be updated when it is ready to be tested.


## Requirements

DSA needs to be installed. Follow the instructions [here](https://github.com/DigitalSlideArchive/digital_slide_archive/tree/master/devops/dsa) on how to do so.

In addition, docker need to be setup such that it works with pyFAST. For that I strongly recommend installing Docker desktop. You might also need to install the nvidia docker to make it work properly:

```
sudo apt update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```


## Installation

Clone the repository:
```
git clone https://github.com/andreped/FP-dsa-plugin.git
```

Build the docker image for the core fastpathology:
```
cd dsa/fastpathology/
docker build -t fastpathology .
```

Build the docker image for the plugin:
```
cd ../
docker build -t fastpathology-dsa-plugin .
```

To add the plugin to DSA, choose `Upload new Task` under Slicer CLI Web Tasks in the DSA web UI. The plugin can then be used from the Analysis Page.


## Acknowledgements

The core was built based on [pyFAST](https://github.com/smistad/FAST), and the plugin was inspired by the plugins made for [MONAILabel](https://github.com/Project-MONAI/MONAILabel/tree/main/plugins/dsa) and [HistomicsTK](https://github.com/DigitalSlideArchive/HistomicsTK/tree/master/histomicstk/cli).

The plugin was made for the [Digital Slide Archive](https://github.com/DigitalSlideArchive/digital_slide_archive) which have developed an open and extremely robust and user-friendly archive web solution for large microscopy images. Building our methods on top of DSA was done with ease and credit to the developers such as [manthey](https://github.com/manthey) for addressing any issue and concerns we had at impressive speed!


## License

The plugin has [MIT-License](https://github.com/andreped/FP-dsa-plugin/blob/main/LICENSE).

Note that the different components used have their respective licenses. However, to the best of our knowledge, they are all dependencies used have permissive licenses with no real proprietary limitations.
