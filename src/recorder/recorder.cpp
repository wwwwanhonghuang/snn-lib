#include "recorder/recorder.hpp"
#include <iostream>

namespace snnlib {

    
    void RecorderFacade::add_neuron_record_item(const std::string& neuron_name, NeuroRecordCallback callback) {
        neuron_recorder_callback[neuron_name].push_back(callback);
    }

    void RecorderFacade::add_connection_record_item(const std::string& connection_name, ConnectionRecordCallback callback) {
        connection_recorder_callback[connection_name].push_back(callback);
    }

    void RecorderFacade::process_neuron_recorder(const std::string& neuron_name, std::shared_ptr<AbstractSNNNeuron> neuron, int t, int dt) {
        auto it = neuron_recorder_callback.find(neuron_name);
        if (it != neuron_recorder_callback.end()) {
            for (const auto& callback : it->second) {
                callback(neuron_name, neuron, t, dt);
            }
        }
    }

    void RecorderFacade::process_connection_recorder(std::string connection_name, std::shared_ptr<AbstractSNNConnection> connection, int t, int dt) {
        auto it = connection_recorder_callback.find(connection_name);
        if (it != connection_recorder_callback.end()) {
            for (const auto& callback : it->second) {
                callback(connection_name, connection, t, dt);
            }
        }
    }
} 
