#ifndef WEIGHT_RECORDER_HPP
#define WEIGHT_RECORDER_HPP
#include <string>
#include <memory>
#include <fstream>
#include "connections/connection.hpp"

namespace snnlib{

    struct ConnectionRecorder{
        public:
        void record_synapse_response_to_file(const std::string& filepath, std::shared_ptr<snnlib::AbstractSNNConnection> connection, int t){
            
            if(file_streams.find(filepath) == file_streams.end()){
                    std::unique_ptr<std::ofstream> output_stream_ptr = std::make_unique<std::ofstream>(filepath);

                    if(!output_stream_ptr->is_open()){
                        std::cout << "Error: cannot open file: " << filepath << std::endl;
                        return;
                    }
                    file_streams[filepath] = std::move(output_stream_ptr);
                    std::ofstream& output_stream = *file_streams[filepath];
                    output_stream << "t\\id, ";
                    std::shared_ptr<snnlib::AbstractSNNSynapse> synapse = connection->synapses;
                    for(int i = 0; i < synapse->n_presynapse_neurons(); i++){
                        for(int j = 0; j < synapse->n_postsynapse_neurons(); j++){
                        output_stream << i << "-" << j;
                        if(i * synapse->n_postsynapse_neurons() + j != synapse->n_presynapse_neurons() * synapse->n_postsynapse_neurons() - 1)
                            output_stream << ", ";
                        }
                    }
                    output_stream << std::endl;
                }
                std::ofstream& output_stream = *file_streams[filepath];
                output_stream << t << ", ";
                _record_synapse_response(output_stream, connection);
            
        }
        void _record_synapse_response(std::ostream& output_stream, std::shared_ptr<snnlib::AbstractSNNConnection> connection);
    private:
        std::unordered_map<std::string, std::unique_ptr<std::ofstream>> file_streams;
    };

    struct WeightRecorder
    {
        static void record_connection_weights_to_file(const std::string& path, std::shared_ptr<snnlib::AbstractSNNConnection> connection);
        private:
        // Internal helper method to handle writing the connection weights to a stream
        static void _record_connection_weights(std::ostream& output_stream, std::shared_ptr<snnlib::AbstractSNNConnection> connection);
    };
}

#endif