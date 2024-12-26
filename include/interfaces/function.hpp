#ifndef FUNCTION_HPP
#define FUNCTION_HPP
#include <functional>
using NeuronDynamicsModel = std::function<std::vector<double>(int neuron_id, double input, double* state, int t, double* parameters, double dt)>;
using SynapseDynamicsModel = std::function<std::vector<double>(double input, double* state, int t, double* parameters, double dt)>;

typedef double (*SynapseKernel)(int t, double tau);
#endif