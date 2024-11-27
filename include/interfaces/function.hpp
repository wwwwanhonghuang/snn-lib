#ifndef FUNCTION_HPP
#define FUNCTION_HPP
#include <functional>
// typedef std::vector<double> (*NeuronDynamicsModel)(double input, double* state, double t, double* parameters, double dt);
// typedef std::vector<double> (*SynapseDynamicsModel)(double input, double* state, double t, double* parameters, double dt);
using NeuronDynamicsModel = std::function<std::vector<double>(double input, double* state, double t, double* parameters, double dt)>;
using SynapseDynamicsModel = std::function<std::vector<double>(double input, double* state, double t, double* parameters, double dt)>;

typedef double (*SynapseKernel)(int t, double tau);
#endif