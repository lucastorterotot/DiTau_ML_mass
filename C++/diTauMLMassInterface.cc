/*###
--- C++ interface to DiTau_ML_mass
--- https://github.com/lucastorterotot/DiTau_ML_mass
--- Davide Zuolo (University and INFN Milano - Bicocca)
--- March 2021
###*/

#include "../interface/diTauMLMassInterface.h"

namespace ditauMLMass {

diTauMLMass::diTauMLMass(const std::string & model)
{
       nn_desc.graph.reset(tensorflow::loadMetaGraph(model));
       nn_desc.session = tensorflow::createSession(nn_desc.graph.get(), model);
       nn_desc.input_layer = "serving_default_dense_1_input:0";
       nn_desc.output_layer = "StatefulPartitionedCall:0";
}

float diTauMLMass::GetScore(const double tau1_pt_reco, const double tau1_eta_reco, const double tau1_phi_reco, const double tau2_pt_reco, const double tau2_eta_reco, const double tau2_phi_reco,
			                 const double jet1_pt_reco, const double jet1_eta_reco, const double jet1_phi_reco, const double jet2_pt_reco, const double jet2_eta_reco, const double jet2_phi_reco,
			                 const double remaining_jets_pt_reco, const double remaining_jets_eta_reco, const double remaining_jets_phi_reco, const int remaining_jets_N_reco, 
			                 const double MET_pt_reco, const double MET_phi_reco, const double MET_covXX_reco, const double MET_covXY_reco, const double MET_covYY_reco,
			                 const double mT1_reco, const double mT2_reco, const double mTtt_reco, const double mTtot_reco, const int PU_npvsGood_reco, const int N_neutrinos_reco
			                )
{
    tensorflow::Tensor x(tensorflow::DT_FLOAT, tensorflow::TensorShape{1, diTauMLMass::n_variables});

    x.flat<float>().setZero();

    x.tensor<float, 2>()(0, InputVars::vars::tau1_pt_reco)            = tau1_pt_reco;
    x.tensor<float, 2>()(0, InputVars::vars::tau1_eta_reco)           = tau1_eta_reco;
    x.tensor<float, 2>()(0, InputVars::vars::tau1_phi_reco)           = tau1_phi_reco;
    x.tensor<float, 2>()(0, InputVars::vars::tau2_pt_reco)            = tau2_pt_reco;
    x.tensor<float, 2>()(0, InputVars::vars::tau2_eta_reco)           = tau2_eta_reco;
    x.tensor<float, 2>()(0, InputVars::vars::tau2_phi_reco)           = tau2_phi_reco;
    x.tensor<float, 2>()(0, InputVars::vars::jet1_pt_reco)            = jet1_pt_reco;
    x.tensor<float, 2>()(0, InputVars::vars::jet1_eta_reco)           = jet1_eta_reco;
    x.tensor<float, 2>()(0, InputVars::vars::jet1_phi_reco)           = jet1_phi_reco;
    x.tensor<float, 2>()(0, InputVars::vars::jet2_pt_reco)            = jet2_pt_reco;
    x.tensor<float, 2>()(0, InputVars::vars::jet2_eta_reco)           = jet2_eta_reco;
    x.tensor<float, 2>()(0, InputVars::vars::jet2_phi_reco)           = jet2_phi_reco;
    x.tensor<float, 2>()(0, InputVars::vars::remaining_jets_pt_reco)  = remaining_jets_pt_reco;
    x.tensor<float, 2>()(0, InputVars::vars::remaining_jets_eta_reco) = remaining_jets_eta_reco;
    x.tensor<float, 2>()(0, InputVars::vars::remaining_jets_phi_reco) = remaining_jets_phi_reco;
    x.tensor<float, 2>()(0, InputVars::vars::remaining_jets_N_reco)   = remaining_jets_N_reco;
    x.tensor<float, 2>()(0, InputVars::vars::MET_pt_reco)             = MET_pt_reco;
    x.tensor<float, 2>()(0, InputVars::vars::MET_phi_reco)            = MET_phi_reco;
    x.tensor<float, 2>()(0, InputVars::vars::MET_covXX_reco)          = MET_covXX_reco;
    x.tensor<float, 2>()(0, InputVars::vars::MET_covXY_reco)          = MET_covXY_reco;
    x.tensor<float, 2>()(0, InputVars::vars::MET_covYY_reco)          = MET_covYY_reco;
    x.tensor<float, 2>()(0, InputVars::vars::mT1_reco)                = mT1_reco;
    x.tensor<float, 2>()(0, InputVars::vars::mT2_reco)                = mT2_reco;
    x.tensor<float, 2>()(0, InputVars::vars::mTtt_reco)               = mTtt_reco;
    x.tensor<float, 2>()(0, InputVars::vars::mTtot_reco)              = mTtot_reco;
    x.tensor<float, 2>()(0, InputVars::vars::PU_npvsGood_reco)        = PU_npvsGood_reco;
    x.tensor<float, 2>()(0, InputVars::vars::N_neutrinos_reco)        = N_neutrinos_reco;

    std::vector<tensorflow::Tensor> pred_vec;

    tensorflow::run(nn_desc.session, { { nn_desc.input_layer, x } },{ nn_desc.output_layer }, &pred_vec);

    return pred_vec.at(0).matrix<float>()(0);
}

diTauMLMass::~diTauMLMass()
{
    tensorflow::closeSession(nn_desc.session);
}

}// namespace ditauMLMass
