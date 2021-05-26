/*###
--- C++ interface to DiTau_ML_mass
--- https://github.com/lucastorterotot/DiTau_ML_mass
--- Davide Zuolo (University and INFN Milano - Bicocca)
--- March 2021
###*/

#include <vector>
#include <string>
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

namespace ditauMLMass{

namespace InputVars{
    enum vars {tau1_pt_reco = 0, tau1_eta_reco = 1, tau1_phi_reco = 2, tau2_pt_reco = 3, tau2_eta_reco = 4, tau2_phi_reco = 5,
               jet1_pt_reco = 6, jet1_eta_reco = 7, jet1_phi_reco = 8, jet2_pt_reco = 9, jet2_eta_reco = 10, jet2_phi_reco = 11,
               remaining_jets_pt_reco = 12, remaining_jets_eta_reco = 13, remaining_jets_phi_reco = 14, remaining_jets_N_reco = 15,
               MET_pt_reco = 16, MET_phi_reco = 17, MET_covXX_reco = 18, MET_covXY_reco = 19, MET_covYY_reco = 20,
               mT1_reco = 21, mT2_reco = 22, mTtt_reco = 23, mTtot_reco = 24, PU_npvsGood_reco = 25, N_neutrinos_reco = 26
    };
}

class diTauMLMass {
public:
    static constexpr size_t n_variables = 27;

    diTauMLMass(const std::string& model);
    ~diTauMLMass();

    struct NNDescriptor {
        std::unique_ptr<tensorflow::MetaGraphDef> graph;
        tensorflow::Session* session;
        std::string input_layer;
        std::string output_layer;
    };

    float GetScore(const double tau1_pt_reco, const double tau1_eta_reco, const double tau1_phi_reco, const double tau2_pt_reco, const double tau2_eta_reco, const double tau2_phi_reco,
                   const double jet1_pt_reco, const double jet1_eta_reco, const double jet1_phi_reco, const double jet2_pt_reco, const double jet2_eta_reco, const double jet2_phi_reco,
                   const double remaining_jets_pt_reco, const double remaining_jets_eta_reco, const double remaining_jets_phi_reco, const int remaining_jets_N_reco, 
                   const double MET_pt_reco, const double MET_phi_reco, const double MET_covXX_reco, const double MET_covXY_reco, const double MET_covYY_reco,
                   const double mT1_reco, const double mT2_reco, const double mTtt_reco, const double mTtot_reco, const int PU_npvsGood_reco, const int N_neutrinos_reco
                   );

private:
    NNDescriptor nn_desc;
};
}// namespace ditauMLMass
