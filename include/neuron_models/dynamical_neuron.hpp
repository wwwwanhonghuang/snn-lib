#ifndef DYNAMICAL_NEURON_HPP
#define DYNAMICAL_NEURON_HPP
#include "neuron_models/neuron.hpp"
#include "neuron_models/neuron_meta_structure.hpp"
#include <memory>

namespace snnlib{
    class SNNNeuronMetaStructure;
    class DynamicalNeuron : public AbstractSNNNeuron, public std::enable_shared_from_this<DynamicalNeuron> {
        public:
        DynamicalNeuron(int n_neurons, int n_states, 
            std::shared_ptr<SNNNeuronMetaStructure> meta_neuron_structure);
        std::shared_ptr<DynamicalNeuron> shared_this;
        std::shared_ptr<SNNNeuronMetaStructure> meta_neuron_structure;

        void initialize();
        double get_state(const std::string& state_name, double* x);
        double output_V(int neuron_id, double* x, double* output_P, int t, double dt);
        std::vector<double> neuron_dynamics(std::shared_ptr<snnlib::DynamicalNeuron> self, int neuron_id, double I, double* x, double t, double* P, double dt);
    };
}

#endif