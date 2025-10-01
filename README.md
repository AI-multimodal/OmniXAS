<div align="center">

<h1>Omnixas</h1>

</div>

OmniXAS is a library designed to predict X-ray Absorption Spectra from material structures using deep learning as outlined in our [manuscript](https://www.arxiv.org/abs/2409.19552) . It provides a modnlar and extensible framework for data processing, model training and evaluating deep learning models for XAS prediction. Inference using the trained models is also integrated into [Lightshow](https://github.com/AI-multimodal/Lightshow).


- [Installation](#installation)
  - [Development](#development)
  - [Docker](#docker)
- [Usage](#usage)
  - [Featurization](#featurization)
  - [Material Splitting](#material-splitting)
  - [ML Split](#ml-split)
  - [Scaling (Optional)](#scaling-optional)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Plotting](#plotting)
- [Abstract](#abstract)
- [Funding acknowledgement](#funding-acknowledgement)
- [Citation](#citation)
- [Contact](#contact)

## Installation

```bash
pip install -v "git+https://github.com/AI-multimodal/OmniXAS.git"
```

### Development

```bash
git clone https://github.com/AI-multimodal/OmniXAS.git
cd OmniXAS
conda create --name omnixas python=3.11.0
conda activate omnixas
pip install -v .
```
Environment also available at [install.sh](install/install.sh).


### Docker

```bash
docker build -t omnixas:bnl install
docker run --gpus [all|'"device=0,1,2"'] -v /path/OmniXAS:/workspace -it --user $(id -u):$(id -g) omnixas:bnl bash
```


## Usage

After installing the library, you can use it as follows:

```python
import omnixas
# Example usage code here
```

### Featurization

```python
from omnixas.data.xas import ElementSpectrum, IntensityValues, Material, EnergyGrid
from omnixas.core.periodic_table import Element, SpectrumType
from omnixas.featurizer.m3gnet_featurizer import M3GNetSiteFeaturizer

spectrum_1 = ElementSpectrum(
    element=Element.Cu,
    type=SpectrumType.FEFF,
    index=0,
    material= Material( id = ..., structure = ...),
    intensities=IntensityValues(...)
    energies=EnergyGrid(...),
)

# featurize material structure corresponding to the spectrum_1
feature_1 = M3GNetSiteFeaturizer().featurize(spectrum_1.material.structure, spectrum_1.index)
...
features = [feature_1, feature_2, ...]
spectra = [spectrum_1.intensities, spectrum_2.intensities, ...]
```

### Material Splitting

```python

from omnixas.data.material_split import MaterialSplitter
idSite = [(id, site) for id in spectra.keys() for site in spectra[id].keys()]
split_idSite = MaterialSplitter.split(
    idSite=idSite,
    target_fractions=target_fractions,
    seed=seed
)
train_idSite, val_idSite, test_idSite = split_idSite
```

### ML Split

```python
from omnixas.data.ml_data import MLData, MLSplits
train_data = MLData(
    X=[features[id][site] for id, site in train_idSite],
    y=[spectra[id][site] for id, site in train_idSite],
)
val_data = ...
test_data = ...
split = MLSplits(train=train_data, val=valdata, test=testdata)
```

### Scaling (Optional)

- `Warning`: Do not use scaler that makes the spectrum values negative if the models produce only positive values (e.g. `XASBlock` with `Softplus` activation !!)

```python
from omnixas.data.scaler import ScaledMlSplit, UncenteredRobustScaler
split = ScaledMlSplit.from_splits(
    splits= MLSplits,
    x_scaler= UncenteredRobustScaler,
    y_scaler= UncenteredRobustScaler,
)
```

### Training

```python

from omnixas.model.xasblock_regressor import XASBlockRegressor
model = XASBlockRegressor(
    directory=f"checkpoints/{element}",
    max_epochs=100,
    early_stopping_patience=25,  # stops if val_loss does not improve for 25 epochs
    overwrite_save_dir=True,  # delete save_dir else adds new files to it
    input_dim=64, # feature dimension
    output_dim=200, # spectra dimension
    hidden_dims=[200,200], # widths of hidden layer of MLP
    initial_lr=1e-2,  # initial learning rate, will be optimized by lr finder later
    batch_size=128,
)
model.fit(split) # full split object needs to be passed coz it contains val data used in logging
```

### Evaluation

```python
from omnixas.model.metrics import ModelMetrics
predictions = model.predict(split.val.X)
targets = split.val.y
metrics = ModelMetrics(predictions=predictions, targets=targets) 
print(f"MSE: {metrics.mse}")
print(f"Median of mse of spectra: {metrics.median_of_mse_per_spectra}")
```

### Plotting

```python
deciles = metrics.deciles
fig, axs = plt.subplots(9, 1, figsize=(6, 20))
for i, (d, ax) in enumerate(zip(deciles, axs)):
    ax.plot(d[0], label="target")
    ax.plot(d[1], label="prediction")
    ax.fill_between( range(len(d[0])), d[0], d[1], alpha=0.5, interpolate=True)
    ax.legend()
    ax.set_title(f"Decile {i+1}")
fig.tight_layout()
fig.show()
```

## Abstract

[**A Universal Deep Learning Framework for Materials X-ray Absorption Spectra**](https://www.arxiv.org/abs/2409.19552)

*Shubha R. Kharel, Fanchen Meng, Xiaohui Qu, Matthew R. Carbone, Deyu Lu*

X-ray absorption spectroscopy (XAS) is a powerful characterization technique for probing the local chemical environment of absorbing atoms. However, analyzing XAS data presents significant challenges, often requiring extensive, computationally intensive simulations, as well as significant domain expertise. These limitations hinder the development of fast, robust XAS analysis pipelines that are essential in high-throughput studies and for autonomous experimentation. We address these challenges with OmniXAS, a framework that contains a suite of transfer learning approaches for XAS prediction, each contributing to improved accuracy and efficiency, as demonstrated on K-edge spectra database covering eight 3d transition metals (Ti-Cu). The OmniXAS framework is built upon three distinct strategies. First, we use M3GNet to derive latent representations of the local chemical environment of absorption sites as input for XAS prediction, achieving up to order-of-magnitude improvements over conventional featurization techniques. Second, we employ a hierarchical transfer learning strategy, training a universal multi-task model across elements before fine-tuning for element-specific predictions. Models based on this cascaded approach after element-wise fine-tuning outperform element-specific models by up to 69%. Third, we implement cross-fidelity transfer learning, adapting a universal model to predict spectra generated by simulation of a different fidelity with a higher computational cost. This approach improves prediction accuracy by up to 11% over models trained on the target fidelity alone. Our approach boosts the throughput of XAS modeling by orders of magnitude versus first-principles simulations and is extendable to XAS prediction for a broader range of elements. This transfer learning framework is generalizable to enhance deep-learning models that target other properties in materials research.

## Funding acknowledgement

This research is based upon work supported by the U.S. Department of Energy, Office of Science, Office Basic Energy Sciences, under Award Number FWP PS-030. This research used resources of the Center for Functional Nanomaterials (CFN), which is a U.S. Department of Energy Office of Science User Facility, at Brookhaven National Laboratory under Contract No. DE-SC0012704.

The Software resulted from work developed under a U.S. Government Contract No. DE-SC0012704 and are subject to the following terms: the U.S. Government is granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable worldwide license in this computer software and data to reproduce, prepare derivative works, and perform publicly and display publicly.

THE SOFTWARE IS SUPPLIED "AS IS" WITHOUT WARRANTY OF ANY KIND. THE UNITED STATES, THE UNITED STATES DEPARTMENT OF ENERGY, AND THEIR EMPLOYEES: (1) DISCLAIM ANY WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR NON-INFRINGEMENT, (2) DO NOT ASSUME ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF THE SOFTWARE, (3) DO NOT REPRESENT THAT USE OF THE SOFTWARE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS, (4) DO NOT WARRANT THAT THE SOFTWARE WILL FUNCTION UNINTERRUPTED, THAT IT IS ERROR-FREE OR THAT ANY ERRORS WILL BE CORRECTED. IN NO EVENT SHALL THE UNITED STATES, THE UNITED STATES DEPARTMENT OF ENERGY, OR THEIR EMPLOYEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, CONSEQUENTIAL, SPECIAL OR PUNITIVE DAMAGES OF ANY KIND OR NATURE RESULTING FROM EXERCISE OF THIS LICENSE AGREEMENT OR THE USE OF THE SOFTWARE.

## Citation

If you use our code, please consider citing our manuscript: [A Universal Deep Learning Framework for Materials X-ray Absorption Spectra](https://www.arxiv.org/abs/2409.19552)

## Contact

For any questions or feedback, please contact the maintainer: [Shubha Raj Kharel](mailto:shubha.raj.kharel@gmail.com).
