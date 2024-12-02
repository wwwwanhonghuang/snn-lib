#ifndef SYNAPSE_HPP
#define SYNAPSE_HPP
#include <vector>
#include <memory>
#include "interfaces/function.hpp"
#include "neuron_models/neuron.hpp"
#include "macros.def"
namespace snnlib{
    struct AbstractSNNSynapse
    {
        public:
        SynapseDynamicsModel synapse_dynamics;
        std::shared_ptr<snnlib::AbstractSNNNeuron> presynapse_neurons;
        std::shared_ptr<snnlib::AbstractSNNNeuron> postsynapse_neurons;
        
        int n_states_per_synapse;
        std::vector<double> x;
        std::vector<double> x_buffer;
        std::vector<double> P;
        DEF_DYN_SYSTEM_STATE(0, I);

        int n_presynapse_neurons();
        int n_postsynapse_neurons();
        virtual void initialize(){
                
        }
        virtual std::vector<double> output_I() = 0;
        AbstractSNNSynapse(std::shared_ptr<snnlib::AbstractSNNNeuron> presynapse_neurons, 
            std::shared_ptr<snnlib::AbstractSNNNeuron> postsynpase_neurons, int n_states_per_synapse)
        : presynapse_neurons(presynapse_neurons), postsynapse_neurons(postsynpase_neurons) {
            this->n_states_per_synapse = n_states_per_synapse;
            x.assign(n_states_per_synapse * presynapse_neurons->n_neurons * postsynpase_neurons->n_neurons, 0);
            x_buffer.assign(n_states_per_synapse * presynapse_neurons->n_neurons * postsynpase_neurons->n_neurons, 0);
        }
        
        
        virtual ~AbstractSNNSynapse() = default;
        void forward_states_to_buffer(
            const std::vector<double>& weights, // synaptic weights
            const std::vector<double>& S,      // presynaptic spike train
            double t,                          // current time
            double* P,                         // parameters
            double dt                          // time step
        );

        void _evolve_state(int weight, const std::vector<double>& S, double t, double* P, double dt);
        void update_states_from_buffer();
    };

    static SynapseDynamicsModel create_single_exponential_dynamics(int state_current_index, int param_tau_index) {
        return [state_current_index, param_tau_index](double input, double* x, double t, double* P, double dt) -> std::vector<double> {
            double I = x[state_current_index];  // Current state (I)
            double tau = P[param_tau_index]; // Fetch tau using captured index
            double dot_I = (-I / tau) + input;
            return {dot_I};
        };
    }

    static SynapseDynamicsModel create_double_exponential_dynamics(int state_current_index, int state_aux_index, int param_tau_rise_index, int param_tau_decay_index) {
        return [state_current_index, state_aux_index, param_tau_rise_index, param_tau_decay_index](double input, double* x, double t, double* P, double dt) -> std::vector<double> {
            double I = x[state_current_index];
            double aux = x[state_aux_index];
            double tau_rise = P[param_tau_rise_index];
            double tau_decay = P[param_tau_decay_index];
            double dot_aux = (-aux / tau_rise) + input;
            double dot_I = -I / tau_decay + aux;
            return {dot_I, dot_aux};
        };
    }

    struct CurrentBasedKernalSynapse: public AbstractSNNSynapse{
        bool kernel_param_tau;
        DEF_DYN_SYSTEM_STATE(1, aux);

        DEF_DYN_SYSTEM_PARAM(0, kernel_param_tau);
        DEF_DYN_SYSTEM_PARAM(1, kernel_param_tau_2);

        CurrentBasedKernalSynapse(std::shared_ptr<snnlib::AbstractSNNNeuron> presynapse_neurons,
                            std::shared_ptr<snnlib::AbstractSNNNeuron> postsynpase_neurons,
                            std::string synapse_dynamics_model_name,
                            double kernel_param_tau, double kernel_param_tau_2, double g_syn, double E_syn):
            AbstractSNNSynapse(presynapse_neurons, postsynpase_neurons, 2)
        {
            if(synapse_dynamics_model_name == "single_exponential"){
                this->synapse_dynamics = create_single_exponential_dynamics(0, 0);
            }else if(synapse_dynamics_model_name == "double_exponential"){
                this->synapse_dynamics = create_double_exponential_dynamics(0, 1, 0, 1);
            }
            P.assign({kernel_param_tau, kernel_param_tau_2, g_syn, E_syn});

        }
        
        std::vector<double> output_I(){
            int pre_neurons = n_presynapse_neurons();
            int post_neurons = n_postsynapse_neurons();
            std::vector<double> I;

            for(int i = 0; i < pre_neurons * post_neurons; i++){
                I.push_back(x[i * n_states_per_synapse + 0]);
            }
            return I;
        }
    };

    struct ConductanceBasedKernalSynapse: public AbstractSNNSynapse{
        bool kernel_param_tau;
        DEF_DYN_SYSTEM_STATE(1, aux);
        

        DEF_DYN_SYSTEM_PARAM(0, kernel_param_tau);
        DEF_DYN_SYSTEM_PARAM(1, kernel_param_tau_2);
        DEF_DYN_SYSTEM_PARAM(2, g_syn);
        DEF_DYN_SYSTEM_PARAM(3, E_syn);

        ConductanceBasedKernalSynapse(std::shared_ptr<snnlib::AbstractSNNNeuron> presynapse_neurons,
                            std::shared_ptr<snnlib::AbstractSNNNeuron> postsynpase_neurons,
                            std::string synapse_dynamics_model_name,
                            double kernel_param_tau, double kernel_param_tau_2, double g_syn, double E_syn):
            AbstractSNNSynapse(presynapse_neurons, postsynpase_neurons, 2)
        {
            if(synapse_dynamics_model_name == "single_exponential"){
                this->synapse_dynamics = create_single_exponential_dynamics(0, 0);
            }else if(synapse_dynamics_model_name == "double_exponential"){
                this->synapse_dynamics = create_double_exponential_dynamics(0, 1, 0, 1);
            }
            P.assign({kernel_param_tau, kernel_param_tau_2, g_syn, E_syn});
        }
        
        std::vector<double> output_I(){
            std::vector<double> synpase_strength = 
                std::vector<double>(n_presynapse_neurons() * n_postsynapse_neurons(), 0);
            for(int i = 0; i < n_presynapse_neurons(); i++){
                for (int j = 0; j < n_postsynapse_neurons(); j++)
                {
                    int index = i * n_postsynapse_neurons() + j;
                    synpase_strength[index] = x[index * n_states_per_synapse + 0];
                }
            }
            std::vector<double> I(synpase_strength.size(), 0);
         
            for(int i = 0; i < synpase_strength.size(); i++){
                I[i] = synpase_strength[i] * param_g_syn(this->P.data()) 
                                           * (postsynapse_neurons->state_V(this->x.data()) - param_E_syn(this->P.data()));
            }
            return I;
        }
    };

}
#endif