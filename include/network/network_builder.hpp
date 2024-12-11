
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
    class SynapseConfiguration;
    class NeuronConfiguration;
    class ConnectionConfiguration;

    struct NetworkBuilder
    {
        std::unordered_map<std::string, std::shared_ptr<snnlib::NeuronConfiguration>>
            neuron_configuration_map;

        std::unordered_map<std::string, std::shared_ptr<snnlib::ConnectionConfiguration>>
            connection_configuration_map;

        template <typename SynapseType, typename... Args>
        std::shared_ptr<snnlib::SynapseConfiguration> build_synapse(
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
            return std::make_shared<snnlib::SynapseConfiguration>(synapse);
        }
        
        template <typename ConnectionType>
        std::shared_ptr<snnlib::AbstractSNNConnection> connection(const std::string& synapse_name) {
            auto synapse = get_synapse(synapse_name); // Assuming `get_synapse` is a member function
            return std::make_shared<ConnectionType>(synapse);
        }

        template <typename NeuronType, typename... Args>
        std::shared_ptr<snnlib::NeuronConfiguration> build_neuron(const std::string& neuron_name, Args&&... args) {
            int size = _network->neurons.size();
            auto neuron = std::make_shared<NeuronType>(std::forward<Args>(args)...);

            _network->neurons[neuron_name] = neuron;           


            if(neuron_configuration_map.find(neuron_name) != neuron_configuration_map.end()){
                std::cerr << "Network Build Error: A neuron with name " 
                    << neuron_name << " has already been recorded." << std::endl;
                return nullptr;
            }

            std::shared_ptr<snnlib::NeuronConfiguration> configuration = std::make_shared<snnlib::NeuronConfiguration>(neuron);
            neuron_configuration_map[neuron_name] = configuration;
            return configuration;
        }


        std::shared_ptr<snnlib::ConnectionConfiguration> 
            register_connection(const std::string& connection_name, 
            std::shared_ptr<snnlib::AbstractSNNConnection> connection);
            
        template <typename ConnectionType, typename... Args>
        std::shared_ptr<snnlib::ConnectionConfiguration> build_connection(
            const std::string& connection_name,
            const std::string& synapse_name,
            Args&&... args
            ) {
            
            auto synapse = synapse_record[synapse_name];
            auto connection = std::make_shared<ConnectionType>(synapse, std::forward(args)...);

            _network->connections[connection_name] = connection;           
            
            if(connection_configuration_map.find(connection_name) != connection_configuration_map.end()){
                std::cerr << "Network Build Error: A connection with name " 
                    << connection_name << " has already been recorded." << std::endl;
                return nullptr;
            }

            std::shared_ptr<snnlib::ConnectionConfiguration> configuration = 
                std::make_shared<snnlib::ConnectionConfiguration>(connection);
            connection_configuration_map[connection_name] = configuration;
            return configuration;
        }
        NetworkBuilder();
        std::shared_ptr<snnlib::SNNNetwork> build_network();
        
        int record_synapse(std::string synapse_name, std::shared_ptr<snnlib::AbstractSNNSynapse> synapse);
        
        std::shared_ptr<snnlib::AbstractSNNNeuron> get_neuron(std::string neuron_name);
        std::shared_ptr<snnlib::AbstractSNNConnection> get_connection(std::string connection_name);
        std::shared_ptr<snnlib::AbstractSNNSynapse> get_synapse(std::string synapse_name);
        private:
            std::shared_ptr<snnlib::SNNNetwork> _network;
            std::unordered_map<std::string, std::shared_ptr<snnlib::AbstractSNNSynapse>> synapse_record;
    };

    class SynapseConfiguration :  public std::enable_shared_from_this<SynapseConfiguration>{
        public:
            SynapseConfiguration(std::shared_ptr<snnlib::AbstractSNNSynapse> synapse);

            template<typename ConnectionType>
            std::shared_ptr<snnlib::ConnectionConfiguration> 
                build_connection(const std::string& connection_name, 
                std::shared_ptr<snnlib::NetworkBuilder> network_builder){
                    std::shared_ptr<ConnectionType> connection = 
                        std::make_shared<ConnectionType>(this->_synapse);
                    network_builder->register_connection(connection_name, connection);
                    return std::make_shared<snnlib::ConnectionConfiguration>(connection);
            }
        private:
            std::shared_ptr<snnlib::AbstractSNNSynapse> _synapse;
    };

    
    class NeuronConfiguration : public std::enable_shared_from_this<NeuronConfiguration> {
        public:
            std::shared_ptr<snnlib::AbstractNeuronInitializer> 
                get_predefined_initializer(const std::string& initializer_name);

            // Add an initializer and return the current instance for chaining
            std::shared_ptr<snnlib::NeuronConfiguration> add_initializer(const std::string& initializer_name);
            std::shared_ptr<snnlib::NeuronConfiguration> add_initializer(std::shared_ptr<snnlib::AbstractNeuronInitializer> 
                initializer);

            // Apply configurations to finalize
            void apply_initializer();

            NeuronConfiguration(std::shared_ptr<snnlib::AbstractSNNNeuron> neuron);
        private:
            std::shared_ptr<snnlib::AbstractNeuronInitializer> neuron_rest_potential_initializer = 
            std::make_shared<snnlib::RestPotentialInitializer>();
        
            std::vector<std::shared_ptr<snnlib::AbstractNeuronInitializer>> _initializers;
            std::shared_ptr<snnlib::AbstractSNNNeuron> _neuron;
    };


    class ConnectionConfiguration :  public std::enable_shared_from_this<ConnectionConfiguration>{
        public:
            std::shared_ptr<snnlib::AbstractSNNConnectionInitializer> 
                get_predefined_initializer(const std::string& initializer_name);

            std::shared_ptr<snnlib::ConnectionConfiguration> add_initializer(const std::string& initializer_name);

            std::shared_ptr<snnlib::ConnectionConfiguration> add_initializer(std::shared_ptr<snnlib::AbstractSNNConnectionInitializer> 
                initializer);

            void apply_initializer();

            ConnectionConfiguration(std::shared_ptr<snnlib::AbstractSNNConnection> connection);
        private:
            std::shared_ptr<snnlib::AbstractSNNConnectionInitializer> connection_normal_weight_intializer =
            std::make_shared<snnlib::NormalWeightInitializer>();
          
        
            std::vector<std::shared_ptr<snnlib::AbstractSNNConnectionInitializer>> _initializers;
            std::shared_ptr<snnlib::AbstractSNNConnection> _connection;
    };
}
#endif