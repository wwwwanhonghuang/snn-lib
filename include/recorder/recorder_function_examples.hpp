#ifndef RECORDER_FUNCTION_EXAMPLES_HPP
#define RECORDER_FUNCTION_EXAMPLES_HPP
#include "recorder/recorder.hpp"
#include "recorder/connection_recorder.hpp"
#include "recorder/neuron_recorder.hpp"

namespace snnlib
{
    struct recorder_function_examples
    {
            

        static snnlib::ConnectionRecordCallback generate_weight_recorder(){
            snnlib::ConnectionRecordCallback weight_recorder = 
                [](const std::string& connection_name, std::shared_ptr<snnlib::AbstractSNNConnection> connection, int t, double dt) -> void{
                    if(t == 0)  snnlib::WeightRecorder::record_connection_weights_to_file(std::string("data/logs/") + connection_name + std::string(".weights"), connection);
                };
            return weight_recorder;
        }

        static snnlib::NeuroRecordCallback 
        generate_membrane_potential_recorder(std::shared_ptr<snnlib::NeuronRecorder> neuron_recorder){
            snnlib::NeuroRecordCallback membrane_potential_recorder = 
                [neuron_recorder](const std::string& neuron_name, std::shared_ptr<snnlib::AbstractSNNNeuron> neuron, int t, double dt) -> void{
                    neuron_recorder->record_membrane_potential_to_file(
                        std::string("data/logs/")  + neuron_name
                        + std::string(".v"), neuron, t
                    );
            };
            return membrane_potential_recorder;
        }

        

        static snnlib::ConnectionRecordCallback generate_response_recorder(std::shared_ptr<snnlib::ConnectionRecorder> connection_recorder){
            snnlib::ConnectionRecordCallback response_recorder = 
            [connection_recorder](const std::string& connection_name, std::shared_ptr<snnlib::AbstractSNNConnection> connection, int t, double dt) -> void{
                connection_recorder->record_synapse_response_to_file(std::string("data/logs/") + connection_name + std::string(".r"), connection, t);
            };
            return response_recorder;
        }

    };

} // namespace snnlib

#endif