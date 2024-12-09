#ifndef NEURON_CONFIGURATION_HPP
#define NEURON_CONFIGURATION_HPP
#include <memory>
#include <cassert>
#include "neuron_models/initializer.hpp"
namespace snnlib{
class NeuronConfiguration {
        public:
            std::shared_ptr<snnlib::AbstractNeuronInitializer> 
                get_predefined_initializer(const std::string& initializer_name){
                if (initializer_name == "rest_potential_initializer") {
                    return neuron_rest_potential_initializer;
                } else {
                    std::cerr << "Error: unrecognized neuron initializer " << initializer_name << std::endl;
                    assert(false);
                }
            }
            // Add an initializer and return the current instance for chaining
            std::shared_ptr<snnlib::NeuronConfiguration> add_initializer(const std::string& initializer_name) {
                return add_initializer(get_predefined_initializer(initializer_name));
            }

            std::shared_ptr<snnlib::NeuronConfiguration> add_initializer(std::shared_ptr<snnlib::AbstractNeuronInitializer> 
                initializer) {
                if(initializer)
                    _initializers.push_back(initializer);
                return std::make_shared<snnlib::NeuronConfiguration>(this);
            }

            // Apply configurations to finalize
            void apply_initializer() {
                for(auto& initializer : _initializers){
                    initializer->initialize(_neuron);
                }
            }
            NeuronConfiguration(std::shared_ptr<snnlib::AbstractSNNNeuron> neuron){
                this->_neuron = neuron;
            }
        private:
            std::shared_ptr<snnlib::AbstractNeuronInitializer> neuron_rest_potential_initializer = 
            std::make_shared<snnlib::RestPotentialInitializer>();
        
            std::vector<std::shared_ptr<snnlib::AbstractNeuronInitializer>> _initializers;
            std::shared_ptr<snnlib::AbstractSNNNeuron> _neuron;
};
}
#endif