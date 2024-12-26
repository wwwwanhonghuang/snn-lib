#include "recorder/connection_recorder.hpp"

#include <fstream>
#include <cassert>
#include <iostream>
namespace snnlib{
    void ConnectionRecorder::_record_synapse_response(std::ostream& output_stream, 
            std::shared_ptr<snnlib::AbstractSNNConnection> connection){
        int n_presynapse_neuronsn = connection->synapses->presynapse_neurons->n_neurons;
        int n_postsynapse_neuronsn = connection->synapses->postsynapse_neurons->n_neurons;

        std::vector<double> response = connection->synapses->output_I();
        for(int i = 0; i < n_presynapse_neuronsn; i++){
            for(int j = 0; j < n_postsynapse_neuronsn; j++){
                int index = connection->synapses->n_states_per_synapse * i + 
                    connection->synapses->OFFSET_STATE_I;
                double response = connection->synapses->x[index];
                    output_stream << response << " ";
                if(i * n_postsynapse_neuronsn + j != n_presynapse_neuronsn * n_postsynapse_neuronsn - 1){
                    output_stream << ", ";
                }
            }
        }
        output_stream << std::endl;
    }

    void WeightRecorder::record_connection_weights_to_file(const std::string& path, std::shared_ptr<snnlib::AbstractSNNConnection> connection) {
        std::ofstream output_stream(path);
        if(!output_stream){
            std::cout << "Error: cannot open file " << path << std::endl;
            return; 
        }
        _record_connection_weights(output_stream, connection);
    }

    void WeightRecorder::_record_connection_weights(std::ostream& output_stream, std::shared_ptr<snnlib::AbstractSNNConnection> connection) {
        int n_presynapse_neuronsn = connection->synapses->presynapse_neurons->n_neurons;
        int n_postsynapse_neuronsn = connection->synapses->postsynapse_neurons->n_neurons;
        assert((int)connection->weights.size() == n_presynapse_neuronsn * n_postsynapse_neuronsn);

        for(int i = 0; i < n_presynapse_neuronsn; i++){
            for(int j = 0; j < n_postsynapse_neuronsn; j++){
                int index = i * n_postsynapse_neuronsn + j;
                double weight_value = connection->weights[index] * connection->connected[index];
                output_stream << weight_value << " ";
                if(i * n_postsynapse_neuronsn + j != n_presynapse_neuronsn * n_postsynapse_neuronsn - 1){
                    output_stream << ", ";
                }
            }
            output_stream << std::endl;
        }
    }
}