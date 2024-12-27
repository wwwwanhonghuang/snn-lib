#include "network/network_builder.hpp"

namespace snnlib
{
    std::shared_ptr<snnlib::AbstractNeuronInitializer> 
        NeuronConfiguration::get_predefined_initializer(const std::string& initializer_name){
        if (initializer_name == "rest_potential_initializer") {
            return neuron_rest_potential_initializer;
        } else if (initializer_name == "reset_potential_initializer") {
            return neuron_reset_potential_initializer;
        }
        else {
            std::cerr << "Error: unrecognized neuron initializer " << initializer_name << std::endl;
            assert(false);
            return nullptr;
        }
    }

    std::shared_ptr<snnlib::NeuronConfiguration> NeuronConfiguration::add_initializer(const std::string& initializer_name) {
        return add_initializer(get_predefined_initializer(initializer_name));
    }

    std::shared_ptr<snnlib::NeuronConfiguration> NeuronConfiguration::add_initializer(std::shared_ptr<snnlib::AbstractNeuronInitializer> 
        initializer) {
        if(initializer)
            _initializers.push_back(initializer);
        return shared_from_this();
    }

    void NeuronConfiguration::apply_initializer() {
        for(auto& initializer : _initializers){
            initializer->initialize(_neuron);
        }
    }

    NeuronConfiguration::NeuronConfiguration(std::shared_ptr<snnlib::AbstractSNNNeuron> neuron){
        this->_neuron = neuron;
    }
} // namespace snnlib
