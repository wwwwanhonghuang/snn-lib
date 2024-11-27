#ifndef SYNAPSE_HPP
#define SYNAPSE_HPP
#include <vector>
#include "interfaces/function.hpp"
#include "neuron_models/neuron.hpp"

namespace snnlib{
    struct AbstractSNNSynapse
    {
        public:
        SynapseDynamicsModel synapse_dynamics;
        snnlib::AbstractSNNNeuron* presynapse_neurons;
        snnlib::AbstractSNNNeuron* postsynapse_neurons;
        int n_presynapse_neurons(){
            if(presynapse_neurons) return presynapse_neurons->n_neurons;
            return 0; 
        };
        int n_postsynapse_neurons(){
            if(postsynapse_neurons) return postsynapse_neurons->n_neurons;
            return 0;
        };
        int n_states_per_synapse;
        std::vector<double> states;
        std::vector<double> states_buffer;


        virtual std::vector<double> I() = 0;
        AbstractSNNSynapse(snnlib::AbstractSNNNeuron* presynapse_neurons, 
            snnlib::AbstractSNNNeuron* postsynpase_neurons, int n_states_per_synapse)
        : presynapse_neurons(presynapse_neurons), postsynapse_neurons(postsynpase_neurons) {
            this->n_states_per_synapse = n_states_per_synapse;
            states.assign(n_states_per_synapse * presynapse_neurons->n_neurons * postsynpase_neurons->n_neurons, 0);
            states_buffer.assign(n_states_per_synapse * presynapse_neurons->n_neurons * postsynpase_neurons->n_neurons, 0);
        }
        
        
        virtual ~AbstractSNNSynapse() {}
        void forward_states_to_buffer(double weight, const std::vector<double>& S, double t, double* P, double dt){
            for(int i = 0; i < n_presynapse_neurons(); i++){
                for(int j = 0; j < n_postsynapse_neurons(); j++){
                    int index = i * n_postsynapse_neurons() + j;
                    std::vector<double> dx = synapse_dynamics(S[index],
                        &states[index * n_states_per_synapse], t, P, dt
                    );
                    for(int k = 0; k < n_states_per_synapse; k++)
                        states_buffer[index * n_states_per_synapse + k] = dx[k];
                }
            }
        }
        void _evolve_state(int weight, const std::vector<double>& S, double t, double* P, double dt){
            for(int i = 0; i < n_presynapse_neurons(); i++){
                for(int j = 0; j < n_postsynapse_neurons(); j++){
                    int index = i * n_postsynapse_neurons() + j;
                    std::vector<double> dx = synapse_dynamics(S[index],
                        &states[index * n_states_per_synapse], t, P, dt
                    );
                    for(int k = 0; k < n_states_per_synapse; k++)
                        states[index * n_states_per_synapse + k] = dx[k];
                }
            }
        }
        void update_states_from_buffer(){
            for(int i = 0; i < n_presynapse_neurons() * n_postsynapse_neurons() * n_states_per_synapse; i++){
                states[i] = states_buffer[i];
            }
        }
    };

    struct AbstractCurrentBaseSynpase: public AbstractSNNSynapse
    {
        private:
            std::vector<double> x;
            
        public:
            AbstractCurrentBaseSynpase(snnlib::AbstractSNNNeuron* presynapse_neurons, snnlib::AbstractSNNNeuron* postsynpase_neurons, int n_states_per_synapse):
                AbstractSNNSynapse(presynapse_neurons, postsynpase_neurons, n_states_per_synapse){}
            double* current(){
                return &x[0];
            }
            virtual std::vector<double> I() = 0;
    };
    
    struct ConvolutionCurrentBasedSynapse: public AbstractCurrentBaseSynpase
    {
        SynapseKernel kernel;
        ConvolutionCurrentBasedSynapse(snnlib::AbstractSNNNeuron* presynapse_neurons, snnlib::AbstractSNNNeuron* postsynpase_neurons):
            AbstractCurrentBaseSynpase(presynapse_neurons, postsynpase_neurons, 2){

            }
        std::vector<double> I(){
            return std::vector<double>(states.begin(), states.begin() + n_presynapse_neurons() * n_postsynapse_neurons());
        }
    };
    
    struct AbstractConductanceBaseSynpase: AbstractSNNSynapse
    {
        private:
            std::vector<double> x;
            
        public:
            double* current(){
                return &x[0];
            }
            virtual std::vector<double> I() = 0;
    };
}
#endif