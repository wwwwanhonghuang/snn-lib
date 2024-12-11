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
        void record_membrane_potential_to_file(const std::string& filepath, std::shared_ptr<snnlib::AbstractSNNNeuron> neuron, int t){
            if(file_streams.find(filepath) == file_streams.end()){
                std::unique_ptr<std::ofstream> output_stream_ptr = std::make_unique<std::ofstream>(filepath);

                if(!output_stream_ptr->is_open()){
                    std::cout << "Error: cannot open file: " << filepath << std::endl;
                    return;
                }
                file_streams[filepath] = std::move(output_stream_ptr);
                std::ofstream& output_stream = *file_streams[filepath];
                output_stream << "t\\id, ";
                for(int i = 0; i < neuron->n_neurons; i++){
                    output_stream << i;
                    if(i != neuron->n_neurons - 1)
                        output_stream << ", ";
                }
                output_stream << std::endl;
            }
            
            std::ofstream& output_stream = *file_streams[filepath];
            output_stream << t << ", ";
            _record_membrane_potential(output_stream, neuron);
        }
        void _record_membrane_potential(std::ostream& output_stream, std::shared_ptr<snnlib::AbstractSNNNeuron> neuron);
    private:
        std::unordered_map<std::string, std::unique_ptr<std::ofstream>> file_streams;
    };

}
#endif