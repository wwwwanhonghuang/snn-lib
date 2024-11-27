#include "recorder/weight_recorder.hpp"

#include <fstream>
#include <cassert>
#include <iostream>
namespace snnlib{
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
        assert(connection->weights.size() == n_presynapse_neuronsn * n_postsynapse_neuronsn);

        for(int i = 0; i < n_presynapse_neuronsn; i++){
            for(int j = 0; j < n_postsynapse_neuronsn; j++){
                int index = i * n_postsynapse_neuronsn + j;
                double weight_value = connection->weights[index] * connection->connected[index];
                output_stream << weight_value << " ";
            }
            output_stream << std::endl;
        }
    }
}