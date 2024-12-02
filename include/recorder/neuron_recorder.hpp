#ifndef NEURON_RECORDER_HPP
#define NEURON_RECORDER_HPP
#include <fstream>
#include <iostream>
#include "neuron_models/neuron.hpp"
struct NeuronRecorder
{
    public:
        void record_membrane_potential_to_file(const std::string& filepath, snnlib::AbstractSNNNeuron& neuron){
            std::ofstream& output_stream(filepath);
            if(!output_stream){
                std::cout << "Error: cannot open file: " << filepath << std::endl;
                return;
            }
            _record_membrane_potential(output_stream, neuron);
        }
        void _record_membrane_potential(const std::ostream& output_stream, snnlib::AbstractSNNNeuron& neuron);
};
#endif