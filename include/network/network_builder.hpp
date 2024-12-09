
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
#include "network/initializer/weight_initializers.hpp"

namespace snnlib{
    struct NetworkBuilder
    {        
        std::shared_ptr<snnlib::AbstractSNNConnectionInitializer> connection_normal_weight_intializer =
            std::make_shared<snnlib::NormalWeightInitializer>();
        
        std::shared_ptr<snnlib::AbstractNeuronInitializer> neuron_rest_potential_initializer = 
            std::make_shared<snnlib::RestPotentialInitializer>();
        
        template <typename SynapseType, typename... Args>
        std::shared_ptr<snnlib::AbstractSNNSynapse> build_synapse(
            const std::string& synapse_name,
            const std::string& source_neuron,
            const std::string& target_neuron,
            Args&&... args) {
            
            auto synapse = std::make_shared<SynapseType>(
                get_neuron(source_neuron), 
                get_neuron(target_neuron), 
                std::forward<Args>(args)...
            );
            record_synapse(synapse_name, synapse);
            return synapse;
        }
        
        template <typename ConnectionType>
        std::shared_ptr<snnlib::AbstractSNNConnection> connection(const std::string& synapse_name) {
            auto synapse = get_synapse(synapse_name); // Assuming `get_synapse` is a member function
            return std::make_shared<ConnectionType>(synapse);
        }


        template <typename NeuronType, typename... Args>
        int build_neuron(const std::string& neuron_name,
            std::shared_ptr<snnlib::AbstractNeuronInitializer> custom_initializer = nullptr,
            const std::string& initializer_name = "", Args&&... args) {

            // Add the neuron to the network
            int size = _network->neurons.size();
            auto neuron = std::make_shared<NeuronType>(std::forward<Args>(args)...);

            _network->neurons[neuron_name] = neuron;

            // Determine the initializer
            std::shared_ptr<snnlib::AbstractNeuronInitializer> initializer;
            if (custom_initializer) {
                initializer = custom_initializer;  // Use custom initializer
            } else if (!initializer_name.empty()) {
                if (initializer_name == "rest_potential_initializer") {
                    initializer = neuron_rest_potential_initializer;
                } else {
                    std::cerr << "Error: unrecognized neuron initializer " << initializer_name << std::endl;
                    assert(false);
                }
            }

            // Add the neuron initializer if available
            if (initializer) {
                _network->neuron_initializers[neuron_name] = initializer;
            }

            return size;
        }

        template <typename ConnectionType, typename... Args>
        void build_connection(
            const std::string& connection_name,
            const std::string& synapse_name,
            std::shared_ptr<snnlib::AbstractSNNConnectionInitializer> custom_initializer = nullptr,
            const std::string& initializer_name = "",
            Args&&... args
            ) {
            
            auto synapse = synapse_record[synapse_name];
            // Build the connection using the synapse
            auto connection = std::make_shared<ConnectionType>(synapse, std::forward(args)...);

            // Determine the initializer
            std::shared_ptr<snnlib::AbstractSNNConnectionInitializer> initializer;
            if (custom_initializer) {
                initializer = custom_initializer;
                add_connection(connection_name, connection, initializer, "");

            } else if (!initializer_name.empty()) {
                if (initializer_name == "connection_normal_initializer") {
                    initializer = connection_normal_weight_intializer;
                } else {
                    std::cerr << "Error: unrecognized connection initializer " << initializer_name << std::endl;
                    assert(false);
                }
                add_connection(connection_name, connection, nullptr, initializer_name);

            }

        }
        NetworkBuilder();
        std::shared_ptr<snnlib::SNNNetwork> build_network();
        
        int record_synapse(std::string synapse_name, std::shared_ptr<snnlib::AbstractSNNSynapse> synapse);
        int add_connection(
            const std::string& connection_name,
            std::shared_ptr<snnlib::AbstractSNNConnection> connection,
            std::shared_ptr<snnlib::AbstractSNNConnectionInitializer> initializer,
            const std::string& initializer_name);

        std::shared_ptr<snnlib::AbstractSNNNeuron> get_neuron(std::string neuron_name);
        std::shared_ptr<snnlib::AbstractSNNConnection> get_connection(std::string connection_name);
        std::shared_ptr<snnlib::AbstractSNNSynapse> get_synapse(std::string synapse_name);
        private:
            std::shared_ptr<snnlib::SNNNetwork> _network;
            std::unordered_map<std::string, std::shared_ptr<snnlib::AbstractSNNSynapse>> synapse_record;
    };
}
#endif