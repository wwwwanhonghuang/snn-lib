#include "network/network.hpp"
#include "recorder/neuron_recorder.hpp"
namespace snnlib
{
    bool SNNNetwork::is_neuron_connected(const std::string& presynapse_neuron_name, const std::string& postsynapse_neuron_name) {
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
        if (presynapse_neuron_id < 0 || presynapse_neuron_id >= connection_matrix.size() ||
            postsynapse_neuron_id < 0 || postsynapse_neuron_id >= connection_matrix[presynapse_neuron_id].size()) {
            throw std::out_of_range("Neuron ID is out of range for the connection matrix.");
        }

        // Check if the connection exists
        return static_cast<bool>(connection_matrix[presynapse_neuron_id][postsynapse_neuron_id]);
    }

    
    void SNNNetwork::initialize(){
        std::cout << "build connection matrix" << std::endl;
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
        connection_matrix.assign(num_neurons, std::vector<std::shared_ptr<snnlib::AbstractSNNConnection>>(num_neurons, nullptr));

        // Populate the connection matrix
        for (auto& connection_record_item : connections) {
            // Retrieve presynaptic and postsynaptic neuron names
            std::shared_ptr<snnlib::AbstractSNNSynapse> synapses = connection_record_item.second->synapses;
            std::string presynapse_neuron_name = neuron_name_map[synapses->presynapse_neurons];
            std::string postsynapse_neuron_name = neuron_name_map[synapses->postsynapse_neurons];

            // Retrieve neuron IDs from their names
            int presynapse_neuron_id = neuron_id_map[presynapse_neuron_name];
            int postsynapse_neuron_id = neuron_id_map[postsynapse_neuron_name];

            // Set the connection in the matrix (assuming directed connection)
            connection_matrix[presynapse_neuron_id][postsynapse_neuron_id] = connection_record_item.second;
        }
        
        // initialize all components
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

    void SNNNetwork::evolve_states(int t, double dt){
        std::cout << " begin evolve states" << std::endl;

        for(auto& neuron_record_item: neurons){
            std::cout << "   >> neuron: " << neuron_record_item.first << 
                " size = " << neuron_record_item.second->n_neurons << std::endl;
            std::cout << "     >> calculate current from synapses." << std::endl;
            int neuron_id = neuron_id_map[neuron_record_item.first];
            std::vector<double> input_current(neuron_record_item.second->n_neurons, 0);
            
            snnlib::NeuronRecorder neuron_recorder;

            neuron_recorder.record_membrane_potential_to_file(
                std::string("data/logs/") + std::string("t_") + std::to_string(t) + std::string("_")
                + neuron_record_item.first + ".v", neuron_record_item.second);

            // calculate current input to this neuron. 
            for(int i = 0; i < neuron_id_map.size(); i++){
                if(!connection_matrix[i][neuron_id]) continue;
                std::shared_ptr<snnlib::AbstractSNNConnection> current_connection = connection_matrix[i][neuron_id];
                std::vector<double> synpase_out = current_connection->synapses->output_I();
                std::cout << "       >> found connection " << connection_name_map[current_connection] << 
                    " connect to neuron " << neuron_record_item.first << std::endl;
                std::cout << "       >> synpase output size = " << synpase_out.size() 
                    << " expect presynapse neuron count = " << synpase_out.size() / neuron_record_item.second->n_neurons
                    << " real presynapse neuron count = " << current_connection->synapses->presynapse_neurons->n_neurons
                    << std::endl;
                assert(current_connection->synapses->presynapse_neurons->n_neurons == synpase_out.size() / neuron_record_item.second->n_neurons);

                int n_presynpase_neurons = current_connection->synapses->n_presynapse_neurons();
                int n_postsynpase_neurons = current_connection->synapses->n_postsynapse_neurons();
                assert(current_connection->weights.size() == n_presynpase_neurons * n_postsynpase_neurons);

                for(int syn_i = 0; syn_i < n_presynpase_neurons; syn_i++){
                    for(int syn_j = 0; syn_j < n_postsynpase_neurons; syn_j++){
                        int index = syn_i * n_postsynpase_neurons + syn_j;
                        input_current[syn_j] += synpase_out[index];
                    }
                }
            }

            snnlib::SimulationStateRecorder recorder;
            recorder.record_input_current_to_file(std::string("data/logs/t_") + std::to_string(t) + std::string("_") + 
            neuron_record_item.first + std::string(".input_current"), input_current);
            std::cout << "   >> begin forward_states_to_buffer: " << std::endl;
            
            neuron_record_item.second->forward_states_to_buffer(input_current, t, &neuron_record_item.second->P[0], dt);
        }

        for(auto& connection_record_item: connections){
            auto synapse = connection_record_item.second->synapses;
            int n_synapse = synapse->n_presynapse_neurons() * synapse->n_postsynapse_neurons();
            std::vector<double> S(n_synapse, 0.0);
            for(int i = 0; i < synapse->n_presynapse_neurons(); i++){
                for (int j = 0; j < synapse->n_postsynapse_neurons(); j++)
                {
                    S[i * synapse->n_postsynapse_neurons() + j] = 
                        synapse->presynapse_neurons->output_V(&synapse->presynapse_neurons->x[i * synapse->presynapse_neurons->get_n_states()], NULL, t, dt);
                }
            }
            connection_record_item.second->forward_states_to_buffer(S, t, &connection_record_item.second->synapses->P[0], dt);
            
        }
    }
}
