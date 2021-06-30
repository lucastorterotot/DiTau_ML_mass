#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
from keras.models import model_from_json
import array

N_neutrinos_in_channel = {
    "tt" : 2,
    "mt" : 3,
    "et" : 3,
    "mm" : 4,
    "em" : 4,
    "ee" : 4,
}

class DNN_model_from_json(object):
    
    def __init__(self, json_file):
        """ Load a DNN using the path to the json file. """
        # load json and create model
        NN_weights_path_and_file = json_file.split('/')
        NN_weights_path_and_file[-1] = "NN_weights-{}".format(NN_weights_path_and_file[-1].replace('.json', '.h5'))
        NN_weights_file = "/".join(NN_weights_path_and_file)

        json_file_ = open(json_file, 'r')
        loaded_model_json = json_file_.read()
        json_file_.close()
        loaded_model = model_from_json(loaded_model_json)
    
        # load weights into new model
        loaded_model.load_weights(NN_weights_file)
        print("Loaded DNN model from disk:")
        print("\t{}".format(json_file))

        self.model = loaded_model

        # load list of inputs for the model
        sys.path.insert(0, json_file.rstrip(json_file.split('/')[-1]))
        import inputs_for_models_in_this_dir
        reload(inputs_for_models_in_this_dir) # avoid being stuck with previous versions
        this_model_inputs = inputs_for_models_in_this_dir.inputs
        self.inputs = this_model_inputs
            
    def predict(self, evt, channel):
        """ Return the predicted value for an event in channel (tt, mt, et, mm, em, ee). """
        # Get the inputs from tree
        df = {}
        for input in self.inputs:
            if input not in ["mt_tt", "N_neutrinos_reco"]:
                df[input] = evt.GetLeaf(input).GetValue(0)
        # get mt_tt as defined for training
        df["mt_tt"] = (2*df["pt_1"]*df["pt_2"]*(1-np.cos(df["phi_1"]-df["phi_2"])))**.5
        # derive N neutrinos
        df["N_neutrinos_reco"] = N_neutrinos_in_channel[channel]
        # Set -10 to 0 for variable in ["jpt_r", "jeta_r", "jphi_r", "Njet_r"] as defined in training
        for variable in ["jpt_r", "jeta_r", "jphi_r", "Njet_r"]:
            if variable in self.inputs:
                if df[input] == -10:
                    df[input] = 0
        return self.model.predict(np.array([[df[input] for input in self.inputs]]))
