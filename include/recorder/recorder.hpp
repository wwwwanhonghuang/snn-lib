#ifndef RECORDER_HPP
#define RECORDER_HPP
#include <memory>
#include <functional>
#include <unordered_map>
#include <vector>
#include "neuron_models/neuron.hpp"
#include "connections/connection.hpp"
namespace snnlib
{
        
    using NeuroRecordCallback = std::function<void(const std::string& neuron_name, std::shared_ptr<snnlib::AbstractSNNNeuron> neuron, int t, int dt)>;
    using ConnectionRecordCallback = std::function<void(const std::string& connection_name, std::shared_ptr<snnlib::AbstractSNNConnection> connection, int t, int dt)>;


    struct RecorderFacade
    {
        std::unordered_map<std::string, std::vector<NeuroRecordCallback>> neuron_recorder_callback;
        std::unordered_map<std::string, std::vector<ConnectionRecordCallback>> connection_recorder_callback;

        void add_neuron_record_item(const std::string& neuron, NeuroRecordCallback callback);
        void add_connection_record_item(const std::string& connection, ConnectionRecordCallback callback);
        void process_neuron_recorder(const std::string& neuron_name, std::shared_ptr<snnlib::AbstractSNNNeuron> neuron, int t, int dt);
        void process_connection_recorder(const std::string& connection_name, std::shared_ptr<AbstractSNNConnection> connection, int t, int dt);
    };
} // namespace snnlib


#endif