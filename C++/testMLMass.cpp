/*###
--- C++ interface to DiTau_ML_mass
--- https://github.com/lucastorterotot/DiTau_ML_mass
--- Davide Zuolo (University and INFN Milano - Bicocca)
--- March 2021
###*/

#include "../interface/diTauMLMassInterface.h"

int main ()
{

ditauMLMass::diTauMLMass test("${CMSSW_BASE}/src/DiTau_ML_mass/models/DNN1/");
std::cout << test.GetScore (213.75, 1.834961, 0.67041, 132.75, 0.425781, -2.181641, 70.625, 1.378906, 1.009766, 66.75, 2.707031, -3.060547, 55.1875, 0.429688, 1.071289, 1, 227.125, -2.140625, 828.0, 458.0, 816.0, 434.75, 7.121094, 333.5, 548.0, 18, 2) << std::endl;
return 0;

// Expected NN output for these inputs: 699.542

}

