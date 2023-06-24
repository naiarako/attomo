# attomo

A Python package for tissue attenuation imaging with Pulse-Echo Ultrasound Attenuation Tomography technique. The method and implementation details are carefully described in 

> [tba].

Please, cite this article if you use codes in <i>attomo</i>.


### Installation

(1) If required, set up python by installing the latest <a href="https://docs.conda.io/en/latest/miniconda.html">Miniconda</a> distribution.

(2) All required packages can be installed by creating a new enviroment for <i>attomo</i> as

<code> conda env create -f environment_cute_cpu(gpu).yml </code>

(3) Activate the environment:

<code> conda activate attomo </code>

(4) Set up <i>attomo</i> module (developer mode) by running:

<code> pip install -e . </code>

in this directory.

(5) You are now ready to run the self-explanatory jupyter notebooks! Just type

<code> jupyter-notebook </code>

and select the notebooks (e.g., <a href="Attenuation_tomography.ipynb">Attenuation_tomography.ipynb</a>).


### Data:

This public version of the code is written to use the ultrasound signals computed from the <a href="http://www.k-wave.org/">k-Wave</a> open-source wave propagation simulator. The Matlab script used to generate the necessary data can be found in <a href="data/Script_kWave_simulation.m">./data/Script_kWave_simulation.m </a>.

