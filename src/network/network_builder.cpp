#include "network/network_builder.hpp"

namespace snnlib{
    NetworkBuilder::NetworkBuilder(){
        _network = new snnlib::SNNNetwork();
    }
    SNNNetwork* NetworkBuilder::build_network(){
        _network->initialize();
        return _network;
    }
    int NetworkBuilder::add_neuron(std::string neuron_name, std::shared_ptr<snnlib::AbstractSNNNeuron> neuron){
        int size = _network->neurons.size();
        _network->neurons[neuron_name] = neuron;
        return size;
    }
    int NetworkBuilder::record_synapse(std::string synapse_name, std::shared_ptr<snnlib::AbstractSNNSynapse> synapse){
        int size = synapse_record.size();
        synapse_record[synapse_name] = synapse;
        return size;
    }

    int NetworkBuilder::add_connection(std::string connection_name, std::shared_ptr<snnlib::AbstractSNNConnection> connection){
        int size = _network->connections.size();
        _network->connections[connection_name] = connection;
        return size;
    }

    std::shared_ptr<snnlib::AbstractSNNNeuron> NetworkBuilder::get_neuron(std::string neuron_name){
        return _network->neurons[neuron_name];
    }
    
    std::shared_ptr<snnlib::AbstractSNNConnection> NetworkBuilder::get_connection(std::string connection_name){
        return _network->connections[connection_name];
    }

    std::shared_ptr<snnlib::AbstractSNNSynapse> NetworkBuilder::get_synapse(std::string synapse_name){
        return synapse_record[synapse_name];
    }
}