#include "neuron_models/neuron_meta_structure.hpp"

namespace snnlib{
    std::shared_ptr<SNNNeuronMetaStructure>  SNNNeuronMetaStructure::build_output_V_callback(std::function<double(std::shared_ptr<snnlib::DynamicalNeuron> self, int, double*, double*, int, double)> callback){
        this->output_V_callback = callback;
        return shared_from_this();
    }
    std::shared_ptr<SNNNeuronMetaStructure> SNNNeuronMetaStructure::build_neuron_dynamics(std::function<std::vector<double>(
        std::shared_ptr<snnlib::DynamicalNeuron>, int, double, double*, double, double*, double)> dynamics){   
        this->neuron_dynamics = dynamics;
        return shared_from_this();
    }
    std::shared_ptr<SNNNeuronMetaStructure> SNNNeuronMetaStructure::build_state(const std::string& state_name, int state_id){
        state_variables[state_name] = state_id;
        return shared_from_this();
    }
}
