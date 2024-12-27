#include "network/network.hpp"
#include "recorder/neuron_recorder.hpp"
#include "recorder/recorder.hpp"
#include <cassert>
#include <omp.h>
namespace snnlib
{
    bool SNNNetwork::is_neuron_connected(const std::string& presynapse_neuron_name, 
            const std::string& postsynapse_neuron_name) {
        
        // Check if the neuron names exist in the map
        auto presynapse_it = neuron_id_map.find(presynapse_neuron_name);
        auto postsynapse_it = neuron_id_map.find(postsynapse_neuron_name);

        if (presynapse_it == neuron_id_map.end()) {
            throw std::invalid_argument("Presynapse neuron '" + presynapse_neuron_name + "' not found in the network.");
        }

        if (postsynapse_it == neuron_id_map.end()) {
            throw std::invalid_argument("Postsynapse neuron '" + postsynapse_neuron_name + "' not found in the network.");
        }

        // Retrieve neuron IDs
        int presynapse_neuron_id = presynapse_it->second;
        int postsynapse_neuron_id = postsynapse_it->second;

        // Bounds check for connection matrix
        if (presynapse_neuron_id < 0 || presynapse_neuron_id >= (int)connection_matrix.size() ||
            postsynapse_neuron_id < 0 || postsynapse_neuron_id >= (int)connection_matrix[presynapse_neuron_id].size()) {
            throw std::out_of_range("Neuron ID is out of range for the connection matrix.");
        }

        return connection_matrix[presynapse_neuron_id][postsynapse_neuron_id].size() > 0;
    }
    
    void SNNNetwork::initialize(){
        std::cout << "Build connection matrix" << std::endl;
        neuron_id_map.clear();
        connection_id_map.clear();
        connection_matrix.clear();

        for (auto& neuron_record_item : neurons) {
            neuron_id_map[neuron_record_item.first] = neuron_id_map.size();
            neuron_name_map[neuron_record_item.second] = neuron_record_item.first;
        }

        for (auto& connection_record_item : connections) {
            connection_id_map[connection_record_item.first] = connection_id_map.size();
            connection_name_map[connection_record_item.second] = connection_record_item.first;
        }

        size_t num_neurons = neuron_id_map.size();
        connection_matrix.assign(num_neurons, 
            std::vector<std::vector<std::shared_ptr<snnlib::AbstractSNNConnection>>>(
                num_neurons, 
                std::vector<std::shared_ptr<snnlib::AbstractSNNConnection>>()));

        // Populate the connection matrix
        for (auto& connection_record_item : connections) {
            // Retrieve presynaptic and postsynaptic neuron names
            std::shared_ptr<snnlib::AbstractSNNSynapse> synapses = connection_record_item.second->synapses;
            std::string presynapse_neuron_name = neuron_name_map[synapses->presynapse_neurons];
            std::string postsynapse_neuron_name = neuron_name_map[synapses->postsynapse_neurons];

            // Retrieve neuron IDs from their names
            int presynapse_neuron_id = neuron_id_map[presynapse_neuron_name];
            int postsynapse_neuron_id = neuron_id_map[postsynapse_neuron_name];

            connection_matrix[presynapse_neuron_id][postsynapse_neuron_id].emplace_back(connection_record_item.second);
        }
        
        std::cout << "initialize network components" << std::endl;
        for(auto& neuron_record_item : neurons){
            neuron_record_item.second->initialize();
        }
        for(auto& connection_record_item : connections){
            connection_record_item.second->synapses->initialize();
            connection_record_item.second->initialize();
        }
    }

    void SNNNetwork::global_update(){
        for(auto& neuron_record_item: neurons){
            neuron_record_item.second->update_states_from_buffer();
        }
        for(auto& connection_record_item: connections){
            connection_record_item.second->update_states_from_buffer();
        }
    }

    void SNNNetwork::evolve_states(int t, double dt, std::shared_ptr<snnlib::RecorderFacade> recorder_facade){
        std::vector<decltype(neurons)::value_type> neuron_vector(neurons.begin(), neurons.end());

        #pragma omp parallel for
        for(size_t neuron_idx = 0; neuron_idx < neuron_vector.size(); ++neuron_idx){
            auto& neuron_record_item = neuron_vector[neuron_idx];
            int neuron_id = neuron_id_map[neuron_record_item.first];
            std::vector<double> input_current(neuron_record_item.second->n_neurons, 0);
            
            if(recorder_facade){
                recorder_facade->process_neuron_recorder(neuron_record_item.first, neuron_record_item.second, t, dt);
            }

            for(auto& iter : neuron_id_map){
                int i = iter.second;
                if(connection_matrix[i][neuron_id].empty()) continue;
                
                std::vector<std::shared_ptr<snnlib::AbstractSNNConnection>>& 
                    connections = connection_matrix[i][neuron_id];
                
                for(auto current_connection : connections){
                    std::vector<double> synpase_out = current_connection->synapses->output_I();
                    
                    assert((int)current_connection->synapses->presynapse_neurons->n_neurons == (int)synpase_out.size() / neuron_record_item.second->n_neurons);

                    int n_presynpase_neurons = current_connection->synapses->n_presynapse_neurons();
                    int n_postsynpase_neurons = current_connection->synapses->n_postsynapse_neurons();


                    #pragma omp parallel for
                    for(int presyn_idx = 0; presyn_idx < n_presynpase_neurons; presyn_idx++){
                        for(int postsyn_idx = 0; postsyn_idx < n_postsynpase_neurons; postsyn_idx++){
                            int index = presyn_idx * n_postsynpase_neurons + postsyn_idx;
                            input_current[postsyn_idx] += synpase_out[index];
                        }   
                    }
                }
                
            }

            neuron_record_item.second->forward_states_to_buffer(input_current, t, &neuron_record_item.second->P[0], dt);
        }

        #pragma omp parallel
        for(auto& connection_record_item: connections){
            if(recorder_facade){
                recorder_facade->process_connection_recorder(connection_record_item.first, connection_record_item.second, t, dt);
            }
            auto synapse = connection_record_item.second->synapses;
            int n_synapse = synapse->n_presynapse_neurons() * synapse->n_postsynapse_neurons();
            std::vector<double> S(n_synapse, 0.0);
            
            #pragma omp parallel for collapse(2)
            for(int i = 0; i < synapse->n_presynapse_neurons(); i++){
                for (int j = 0; j < synapse->n_postsynapse_neurons(); j++)
                {
                    double presynapse_neuron_output_potential = synapse->presynapse_neurons->output_V(i, &synapse->presynapse_neurons->x_buffer[i * synapse->presynapse_neurons->get_n_states()], NULL, t, dt);
                    S[i * synapse->n_postsynapse_neurons() + j] = presynapse_neuron_output_potential;
                    // if(presynapse_neuron_output_potential > 0){
                    //     std::cout << synapse->presynapse_neurons->n_neurons << " neurons fired::" << presynapse_neuron_output_potential << std::endl;
                    // }
                }
            }

            connection_record_item.second->forward_states_to_buffer(S, t, 
                &connection_record_item.second->synapses->P[0], dt);
        }



    }
}
