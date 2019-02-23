+++
title = "Conda Essentials Notes"
date = 2019-02-18T08:53:39-05:00
draft = false

# Tags and categories
# For example, use `tags = []` for no tags, or the form `tags = ["A Tag", "Another Tag"]` for one or more tags.
tags = ["Software Development", "Virtual Environments"]
categories = ["Data Science"]

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
[image]
  # Caption (optional)
  caption = ""

  # Focal point (optional)
  # Options: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight
  focal_point = ""
+++

{{< figure library="1" src="conda/conda.png" >}}

<h2 style="font-family: Georgia; font-size:1.5em;color:purple; font-style:bold">
Introduction</h2>

**Conda** in an open source package management system that works on all platforms. It is a tool that helps manage packages and environments for different programming languages. Develop a high level understanding of how Conda works helped me at so many levels especially when it comes to managing environments and make my work more reproducable. Below are the notes that I wrote down during my journey of learning Conda and I always refere back to them:

<h2 style="font-family: Georgia; font-size:1.5em;color:purple; font-style:bold">
General</h2>

- Conda packages are files and executables that can in principle contain images, data, noteboeeks, files, etc.
- Conda mainly used in Python ecosystem; however, it can be used with other languages such R, Julia, Scala, etc.
- When installing a package using Conda, it installs its dependencies with it. Also, Conda is able to figure out the platform you're using without the need to specify the platform when installing packages.
- When installing a package, Conda:
  - Checks the platform.
  - Checks the Python version.
  - Install the latest version of the package that is compatible with Python.
  - If it has dependencies, installs the latest versions of the dependencies that are also compatible with each other.
- Under semantic versioning, software is labeled with a three-part version identifier of the form `MAJOR.MINOR.PATCH`; the label components are non-negative integers separated by periods. Assuming all software starts at version 0.0.0, the `MAJOR` version number is increased when significant new functionality is introduced (often with corresponding API changes). Increases in the `MINOR` version number generally reflect improvements (e.g., new features) that avoid backward-incompatible API changes. For instance, adding an optional argument to a function API (in a way that allows old code to run unchanged) is a change worthy of increasing the `MINOR` version number. An increment to the `PATCH` version number is approriate mostly for bug fixes that preserve the same `MAJOR` and MINOR revision numbers. Software patches do not typically introduce new features or change APIs at all (except sometimes to address security issues).
- We can specify `MAJOR`, `MAJOR.MINOR`, or `MAJOR.MINOR.PATCH` when installing any package.
- We can use logical operators to install versions of a package. Examples:
  - `conda install 'python=3.6|3.7'`.
  - `conda install 'python=3.6|3.7*'` .
  - `conda install 'python>=3.6, <=3.7'`.

<h2 style="font-family: Georgia; font-size:1.5em;color:purple; font-style:bold">
Common Commands</h2>

- To update a package, `conda update pckg`.
- To uninstall a package, `conda remove pckg`.
- To search what available versions of a specific package is available, use `conda search pckg`.
- `conda list` will list all installed packages.
- `conda list -n env-name` will list all packages in the environment env-name.
- `conda list pckg` will give information about pckg.
- When installing a pckg without including a channel, it defaults to the main channel that is maintained by Anaconda Inc.
- There other channels where people can upload their packages to and we can reach to those channels when looking for installation such fastai. We use `conda install -c fastai fastai`. Here the channel is fastai and the pckg is also fastai.
- `conda search -c conda-forge -c fastai --override-channels --platform osx-64 fastai` means:
  - Search for fastai in two channels: conda-forge, fastai.
  - override-channels means do not go to default main channel.
  - platform specify which platform.
- Sometimes we don't know the channel of the pckg, we can use `anaconda search pckg` that will return all the channels that the pckg is at and their versions.
- conda-forge is almost as good as the main channels which is led by the community. It has a lot more packages than the main channel.
- There is no system that rates channels, so be carefel when installing packages from any channel.
- We can list all packages in a channel such as `conda search -c conda-forge --override-channels` that will list all packages for the conda-forge channel.

<h2 style="font-family: Georgia; font-size:1.5em;color:purple; font-style:bold">
Environments</h2>

- Environments are a good practice of documenting data science/software development work.
- Environments are nothing more than a directory that contains all the packages so that when trying to import them, it imports them from this directory only. we can use `conda env list` to see all the available environments on our machine.
- To get the packages from a specific environment by name, use `conda list -n env-name`. Otherwise, we get the packages from the current environment.
- To activate an environment, use `conda activate env-name`. To deactivate, `conda deactivate`.
- Environments usually don't take a lot of space.
- We can remove environments using `conda env remove -n env-name`.
- To create an environment, use `conda create -n env-name`. We can also add additional package names to install after creation such as `conda create -n env-name python=3.6* numpy>=1.1`.
- To export an environment, use `conda env export -n env-name`. This will return the output to the terminal. We can also export to a file. For that use `conda env export -n env-name -f env-name.yml`. The '.yml' extension is strongly enouraged. Doing this will assure that all the packages used can be installed by others exactly.
- We can create also an environment from .yml file using `conda env create -f env-name.yml`. Note also that if we only use `conda env create`, it will look for a file that has .yml extension and has the same name as env-name in the current local directory. Moreover, we can create the .yml file with doing the export ourselves and only specify what is important in our environments.