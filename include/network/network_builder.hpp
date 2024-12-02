
#ifndef NETWORK_BUILDER_HPP
#define NETWORK_BUILDER_HPP
#include <vector>
#include <memory>
#include <unordered_map>
#include "network/network.hpp"
#include "neuron_models/neuron.hpp"
#include "connections/connection.hpp"
#include "neuron_models/initializer.hpp"
#include "network/initializer/initializer.hpp"
#include "network/initializer/normal_weight_initializer.hpp"

namespace snnlib{
    struct NetworkBuilder
    {        
        NetworkBuilder();
        std::shared_ptr<snnlib::SNNNetwork> build_network();
        int add_neuron(std::string neuron_name, std::shared_ptr<snnlib::AbstractSNNNeuron> neuron);
        int add_neuron_with_initializer(std::string neuron_name, std::shared_ptr<snnlib::AbstractSNNNeuron> neuron, std::shared_ptr<snnlib::AbstractNeuronInitializer> initinitializer);
        int add_neuron_with_initializer(std::string neuron_name, std::shared_ptr<snnlib::AbstractSNNNeuron> neuron, std::string initinitializer_name);

        int record_synapse(std::string synapse_name, std::shared_ptr<snnlib::AbstractSNNSynapse> synapse);
        int add_connection(std::string connection_name, std::shared_ptr<snnlib::AbstractSNNConnection> connection);
        int add_connection_with_initializer(std::string connection_name, std::shared_ptr<snnlib::AbstractSNNConnection> connection, std::shared_ptr<snnlib::AbstractSNNConnectionInitializer> initializer);
        int add_connection_with_initializer(std::string connection_name, std::shared_ptr<snnlib::AbstractSNNConnection> connection, std::string initializer_name);

        std::shared_ptr<snnlib::AbstractSNNNeuron> get_neuron(std::string neuron_name);
        std::shared_ptr<snnlib::AbstractSNNConnection> get_connection(std::string connection_name);
        std::shared_ptr<snnlib::AbstractSNNSynapse> get_synapse(std::string synapse_name);
        private:
            std::shared_ptr<snnlib::SNNNetwork> _network;
            std::unordered_map<std::string, std::shared_ptr<snnlib::AbstractSNNSynapse>> synapse_record;
    };
}
#endif