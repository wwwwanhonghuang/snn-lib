#ifndef NEURON_RECORDER_HPP
#define NEURON_RECORDER_HPP
#include <fstream>
#include <iostream>
#include <memory>
#include "neuron_models/neuron.hpp"
namespace snnlib
{
    struct NeuronRecorder
    {
    public:
        void record_membrane_potential_to_file(const std::string& filepath, std::shared_ptr<snnlib::AbstractSNNNeuron> neuron){
            std::ofstream output_stream(filepath);
            if(!output_stream){
                std::cout << "Error: cannot open file: " << filepath << std::endl;
                return;
            }
            _record_membrane_potential(output_stream, neuron);
        }
        void _record_membrane_potential(std::ostream& output_stream, std::shared_ptr<snnlib::AbstractSNNNeuron> neuron);
    };
}
#endif