#include "network/network_builder.hpp"

namespace snnlib{
    static std::shared_ptr<snnlib::AbstractNeuronInitializer> neuron_rest_potential_initializer
        = std::make_shared<snnlib::RestPotentialInitializer>();

    static std::shared_ptr<snnlib::AbstractSNNConnectionInitializer> connection_normal_weight_intializer
        = std::make_shared<snnlib::NormalWeightInitializer>();

    NetworkBuilder::NetworkBuilder(){
        _network = std::make_shared<snnlib::SNNNetwork>();
    }
    std::shared_ptr<SNNNetwork> NetworkBuilder::build_network(){
        _network->initialize();
        return _network;
    }
    int NetworkBuilder::add_neuron(std::string neuron_name, std::shared_ptr<snnlib::AbstractSNNNeuron> neuron){
        int size = _network->neurons.size();
        _network->neurons[neuron_name] = neuron;

        return size;
    }

    int NetworkBuilder::add_neuron_with_initializer(std::string neuron_name, std::shared_ptr<snnlib::AbstractSNNNeuron> neuron, std::shared_ptr<snnlib::AbstractNeuronInitializer> initinitializer){
        int size = add_neuron(neuron_name, neuron);
        _network->neuron_initializers[neuron_name] = initinitializer;
        return size;
    }
    
    int NetworkBuilder::add_neuron_with_initializer(std::string neuron_name, std::shared_ptr<snnlib::AbstractSNNNeuron> neuron, std::string initinitializer_name){
        if(initinitializer_name == "rest_potential_initializer"){
            return add_neuron_with_initializer(neuron_name, neuron, neuron_rest_potential_initializer);
        }
        std::cout << "Error: unrecognized neuron initializer " << initinitializer_name << std::endl;
        assert(false);
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

    int NetworkBuilder::add_connection_with_initializer(std::string connection_name, std::shared_ptr<snnlib::AbstractSNNConnection> connection, std::shared_ptr<snnlib::AbstractSNNConnectionInitializer> initializer){
        int size = add_connection(connection_name, connection);
        _network->connection_initializers[connection_name] = initializer;
        return size;
    }
    int NetworkBuilder::add_connection_with_initializer(std::string connection_name, std::shared_ptr<snnlib::AbstractSNNConnection> connection, std::string initializer_name){
        if(initializer_name == "connection_normal_initializer"){
            return add_connection_with_initializer(connection_name, connection, connection_normal_weight_intializer);
        }
        std::cout << "Error: unrecognized connection initializer " << initializer_name << std::endl;
        assert(false);
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