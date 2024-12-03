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
   

    int NetworkBuilder::add_connection(
        const std::string& connection_name,
        std::shared_ptr<snnlib::AbstractSNNConnection> connection,
        std::shared_ptr<snnlib::AbstractSNNConnectionInitializer> initializer = nullptr,
        const std::string& initializer_name = "") {
        
        int size = _network->connections.size();
        _network->connections[connection_name] = connection;

        if (initializer) {
            // If an initializer object is provided, use it
            _network->connection_initializers[connection_name] = initializer;
        } else if (!initializer_name.empty()) {
            // If an initializer name is provided, map it to the corresponding object
            if (initializer_name == "connection_normal_initializer") {
                _network->connection_initializers[connection_name] = connection_normal_weight_intializer;
            } else {
                std::cerr << "Error: unrecognized connection initializer " << initializer_name << std::endl;
                assert(false);
            }
        }

        return size;
    }

    int NetworkBuilder::record_synapse(std::string synapse_name, std::shared_ptr<snnlib::AbstractSNNSynapse> synapse){
        int size = synapse_record.size();
        synapse_record[synapse_name] = synapse;
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