# DiTau_ML_mass - Estimations of di-tau mass using Machine Learning

[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-0.22.1-blue.svg)](https://scikit-learn.org/)
[![Keras](https://img.shields.io/badge/Keras-2.3.1-red.svg)](https://keras.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.1.0-orange.svg)](https://www.tensorflow.org/)
[![ROOT](https://img.shields.io/badge/ROOT-9.0.0-blue.svg)](https://root.cern/)

This repository aims at giving you the ability to use models from [DL_for_HTT_mass](https://github.com/lucastorterotot/DL_for_HTT_mass) in real analysis.

When using this package, please acknowledge it with [the provided citations](https://github.com/lucastorterotot/DiTau_ML_mass/blob/main/DiTau_ML_mass.bib). Thanks!

# Output variable

The output from the models is a mass prediction, in GeV, for the particle decaying into two tau leptons.
We call this outpout `ml_mass` and you can write it with LaTeX using
```
\newcommand{\mlmass}{\ensuremath{m_{\mathrm{ML}}}}
```

# Available models

Models are stored in `models/`.
In each subdirectory, there are 3 files:

- `*.json` contains the model hyperparameters;
- `*.h5` contains the neurons parameters;
- `inputs_for_models_in_this_dir.py` is the ordered list of input variables needed.

More information is available on the inputs in the next section.
Currently available models are listed below.

## DNN1

The `DNN1` model in located in `models/DNN1/`.
It is a
normal
deep
feedforward
fully-connected
neural network
with:

- 3 hidden layers;
- 1000 neurons per hidden layer;
- the softplus ($x \to \ln(1+e^x)$) activation function for the hidden neurons;
- the linear activation function for the single-neuron output layer;
- optimized by Adam;
- with our custom loss defined as
```
def custom_loss(y_pred, y_true):

    factor = 1.0
    
    if y_pred - y_true >= 1000 - y_pred:
        factor = 0.0 # remove event from the loss
    elif y_true - y_pred >= y_pred - 50:
        factor = 0.1 # do not remove completely to keep training convergence

    loss = abs((y_true - y_pred)/(y_true**.5) * factor)

    return mean(loss)
```
- neurons weights initialized by the Glorot Uniform method;
- the input layer taking all the 27 variables listed in the next section.

## DNN2

The `DNN2` model is similar to `DNN1`, except:

- ReLU activation function for neurons in hidden layers (instead of softplus);
- neurons weights initialized by the Glorot Normal method (instead of Glorot Uniform).

This `DNN2` model exhibits slightly better performances than `DNN1` on the di-tau mass predictions.

# Input variables definitions

## Inputs list

The following variables can be used by the provided models. Before using a model, please check that you have them in your ntuple!

- `tau1_pt_reco`: transverse momentum of the first tau visible decay products. The first tau is defined below;
- `tau1_eta_reco`: pseudo-rapidity of the first tau visible decay products;
- `tau1_phi_reco`: azimuthal angle of the first tau visible decay products;
- `tau2_pt_reco`: transverse momentum of the second tau visible decay products. The second tau is defined below;
- `tau2_eta_reco`: pseudo-rapidity of the second tau visible decay products;
- `tau2_phi_reco`: azimuthal angle of the second tau visible decay products;
- `jet1_pt_reco`: transverse momentum of the leading jet (higher pT), jet selection is defined below;
- `jet1_eta_reco`: pseudo-rapidity of the leading jet;
- `jet1_phi_reco`: azimuthal angle of the leading jet;
- `jet2_pt_reco`: transverse momentum of the sub-leading jet (second higher pT);
- `jet2_eta_reco`: pseudo-rapidity of the sub-leading jet;
- `jet2_phi_reco`: azimuthal angle of the sub-leading jet;
- `remaining_jets_pt_reco`: transverse momentum of the Additionnal Hadronic Activity (AHA), defined below;
- `remaining_jets_eta_reco`: pseudo-rapidity of the AHA;
- `remaining_jets_phi_reco`: azimuthal angle of the AHA;
- `remaining_jets_N_reco`: number of jets in the AHA;
- `PuppiMET_pt_reco`: transverse missing momentum obtained from PUPPI (PuppiMET);
- `PuppiMET_phi_reco`: azimuthal angle of the missing transverse momentum;
- `MET_covXX_reco`: first diagonal element of the MET covariance matrix;
- `MET_covXY_reco`: non-diagonal element of the MET covariance matrix;
- `MET_covYY_reco`: second diagonal element of the MET covariance matrix;
- `PuppimT1_reco`: `tau1` transverse mass `mT(tau1, PuppiMET)`;
- `PuppimT2_reco`: `tau2` transverse mass `mT(tau2, PuppiMET)`;
- `PuppimTtt_reco`: `tau1` vs `tau2` transverse mass `mT(tau1, tau2)` (not to be misunderstood as `mT(tau1+tau2, MET)`!);
- `PuppimTtot_reco`: total transverse mass (with PuppiMET);
- `PU_npvsGood_reco`: number of Pile-Up vertices, stored as `npvsGood` in NanoAODs;
- `N_neutrinos_reco`: expected amont of neutrinos from the tau leptons decays given the identified channel (2 in TauTau, 3 in MuTau or EleTau, 4 in MuMuor EleMu or EleEle).

## Detailed definitions

The model should be able to predict masses even if your analysis uses other selection cuts.

- `tau1` and `tau2` are the visible decay products of the tau leptons. In asymmetric channels (MuTau, EleTau, EleMu), `tau1` is the first part of the channel name (i.e. the muon, the electron, the electron respectively). For symmetric channels (TauTau, MuMu, EleEle), `tau1` is the physics object of higher pT.
- Jets selection is: pT > 30 GeV and |eta| < 4.7;
- b-jets selection is: b-tagged jet and pT > 20 GeV and |eta| < 2.5;
- The Additionnal Hadronic Activity (AHA) is defined as the momentum vectorial sum of the remaining jets:
```
AHA_px, AHA_py, AHA_pz = 0, 0, 0
NjetsAHA = 0
for jet in jets_sorted_by_decreasing_pT[1:]: # ignore the two leading jets
    AHA_px, AHA_py, AHA_pz += jet_px, jet_py, jet_pz
    NjetsAHA += 1
```
and then pT, eta and phi for AHA are obtained from its px, py and pz.
- Transverse masses `mT(tau1, PuppiMET)`, `mT(tau2, PuppiMET)` and `mT(tau1, tau2)` are defined as
```
mT(A, B)**2 = 2 * pT(A) * pT(B) * ( 1 - cos(phi(A) - phi(B)) )
```
- The total transverse mass `mTtot` is defined as
```
PuppimTtot**2 = mT(tau1, PuppiMET)**2 + mT(tau2, PuppiMET)**2 + mT(tau1, tau2)**2
```


# Using the code

## Installation

## How to use the provided code

## Implementing it in your own analysis

# Questions or issues
