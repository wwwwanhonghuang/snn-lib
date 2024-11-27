#ifndef FUNCTION_HPP
#define FUNCTION_HPP
#include <vector>
typedef std::vector<double> (*NeuronDynamicsModel)(double input, double* state, double t, double* parameters, double dt);
typedef std::vector<double> (*SynapseDynamicsModel)(double input, double* state, double t, double* parameters, double dt);
typedef double (*SynapseKernel)(int t, double tau);
#endif