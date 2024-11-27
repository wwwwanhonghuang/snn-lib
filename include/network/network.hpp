#ifndef NETWORK_HPP
#define NETWORK_HPP
#include <unordered_map>
#include <string>
#include "neuron_models/neuron.hpp"
#include "connections/connection.hpp"

namespace snnlib{
    struct SNNNetwork{
        // name -> neuron mapping
        std::unordered_map<std::string, snnlib::AbstractSNNNeuron*> neurons;
        std::unordered_map<std::string, snnlib::AbstractSNNConnection*> connections;
        std::unordered_map<snnlib::AbstractSNNNeuron*, std::string> neuron_name_map;
        std::unordered_map<snnlib::AbstractSNNConnection*, std::string> connection_name_map;
        
        // name -> id mapping
        std::unordered_map<std::string, int> neuron_id_map;
        std::unordered_map<std::string, int> connection_id_map;
        std::vector<std::vector<snnlib::AbstractSNNConnection*>> connection_matrix;

        bool is_neuron_connected(const std::string& presynapse_neuron_name, const std::string& postsynapse_neuron_name) {
            auto presynapse_it = neuron_id_map.find(presynapse_neuron_name);
            auto postsynapse_it = neuron_id_map.find(postsynapse_neuron_name);
            
            if (presynapse_it == neuron_id_map.end() || postsynapse_it == neuron_id_map.end()) {
                throw std::invalid_argument("Neuron names not found in the network.");
            }

            int presynapse_neuron_id = presynapse_it->second;
            int postsynapse_neuron_id = postsynapse_it->second;

            return connection_matrix[presynapse_neuron_id][postsynapse_neuron_id];
        }
        void initialization(){
            neuron_id_map.clear();
            connection_id_map.clear();
            connection_matrix.clear();

             // Initialize neuron mappings
            for (auto& neuron_record_item : neurons) {
                neuron_id_map[neuron_record_item.first] = neuron_id_map.size();
                neuron_name_map[neuron_record_item.second] = neuron_record_item.first;
            }

            // Initialize connection mappings
            for (auto& connection_record_item : connections) {
                connection_id_map[connection_record_item.first] = connection_id_map.size();
                connection_name_map[connection_record_item.second] = connection_record_item.first;
            }

            size_t num_neurons = neuron_id_map.size();
            connection_matrix.assign(num_neurons, std::vector<snnlib::AbstractSNNConnection*>(num_neurons, nullptr));

            // Populate the connection matrix
            for (auto& connection_record_item : connections) {
                // Retrieve presynaptic and postsynaptic neuron names
                std::string presynapse_neuron_name = neuron_name_map[connection_record_item.second->synapses->presynapse_neurons];
                std::string postsynapse_neuron_name = neuron_name_map[connection_record_item.second->synapses->postsynapse_neurons];

                // Retrieve neuron IDs from their names
                int presynapse_neuron_id = neuron_id_map[presynapse_neuron_name];
                int postsynapse_neuron_id = neuron_id_map[postsynapse_neuron_name];

                // Set the connection in the matrix (assuming directed connection)
                connection_matrix[presynapse_neuron_id][postsynapse_neuron_id] = connection_record_item.second;
            }
            
        }

        void global_update(){
            for(auto& neuron_record_item: neurons){
                neuron_record_item.second->update_states_from_buffer();
            }
            for(auto& connection_record_item: connections){
                connection_record_item.second->update_states_from_buffer();
            }
        }

        void evolve_states(){
            for(auto& neuron_record_item: neurons){
                // calculate I
                int neuron_id = neuron_id_map[neuron_record_item.first];
                std::vector<double> input_current(neuron_record_item.second->n_neurons, 0);
                for(int i = 0; i < neuron_id_map.size(); i++){
                    if(!connection_matrix[i][neuron_id]) continue;
                    auto current_connection = connection_matrix[i][neuron_id];
                    std::vector<double> synpase_out = current_connection->synapses->I();
                    int n_presynpase_neurons = current_connection->synapses->n_presynapse_neurons();
                    int n_postsynpase_neurons = current_connection->synapses->n_postsynapse_neurons();

                    for(int syn_i = 0; syn_i < n_presynpase_neurons; syn_i++){
                        for(int syn_j = 0; syn_j < n_postsynpase_neurons; syn_j++){
                            int index = syn_i * n_postsynpase_neurons + syn_j;
                            double weight = current_connection->weights[index];
                            synpase_out[index] += weight * synpase_out[index];
                        }
                    }
                }
                neuron_record_item.second->forward_states_to_buffer(input_current, 0, &neuron_record_item.second->P[0], 0);
            }
            
            for(auto& connection_record_item: connections){
                connection_record_item.second->forward_states_to_buffer(0, std::vector<double>(), 0, 0, 0);
            }
        }
    };
}
#endif