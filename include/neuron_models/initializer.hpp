#ifndef NEURON_INITIALIZER_HPP
#define NEURON_INITIALIZER_HPP
#include <memory>
#include "neuron_models/neuron.hpp"
namespace snnlib
{
    struct AbstractNeuronInitializer
    {
        virtual void initialize(std::shared_ptr<snnlib::AbstractSNNNeuron> neuron) = 0;

    };

    struct RestPotentialInitializer : snnlib::AbstractNeuronInitializer
    {
        void initialize(std::shared_ptr<snnlib::AbstractSNNNeuron> neuron){
            neuron->setMembranePotential(-65.0);
        }
    };
    
}
#endif