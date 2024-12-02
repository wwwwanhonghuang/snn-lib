#ifndef WEIGHT_RECORDER_HPP
#define WEIGHT_RECORDER_HPP
#include <string>
#include <memory>
#include "connections/connection.hpp"
namespace snnlib{
    struct WeightRecorder
    {
        void record_connection_weights_to_file(const std::string& path, std::shared_ptr<snnlib::AbstractSNNConnection> connection);
        private:
        // Internal helper method to handle writing the connection weights to a stream
        void _record_connection_weights(std::ostream& output_stream, std::shared_ptr<snnlib::AbstractSNNConnection> connection);
    };
}

#endif