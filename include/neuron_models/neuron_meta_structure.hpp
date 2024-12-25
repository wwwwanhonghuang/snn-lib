#ifndef NEURON_META_STRUCTURE_HPP
#define NEURON_META_STRUCTURE_HPP
#include <memory>
#include <functional>
#include <unordered_map>
#include "neuron_models/dynamical_neuron.hpp"

namespace snnlib{
    class DynamicalNeuron;
    class SNNNeuronMetaStructure : public std::enable_shared_from_this<SNNNeuronMetaStructure>{
        public:
        std::shared_ptr<SNNNeuronMetaStructure> build_output_V_callback(std::function<double(std::shared_ptr<snnlib::DynamicalNeuron> self, int, double*, double*, int, double)> callback);
        std::shared_ptr<SNNNeuronMetaStructure> build_neuron_dynamics(std::function<std::vector<double>(
            std::shared_ptr<DynamicalNeuron>, int, double, double*, double, double*, double)> dynamics);
        std::shared_ptr<SNNNeuronMetaStructure> build_state(const std::string& state_name, int state_id);

        std::function<double(std::shared_ptr<DynamicalNeuron>, int, double*, double*, int, double)> output_V_callback;
        std::function<std::vector<double>(std::shared_ptr<DynamicalNeuron>, int, double, double*, double, double*, double)> neuron_dynamics;
        std::unordered_map<std::string, double> state_variables;
    };
}

#endif