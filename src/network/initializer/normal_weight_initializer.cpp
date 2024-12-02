#include "network/initializer/normal_weight_initializer.hpp"

namespace snnlib
{
    void NormalWeightInitializer::initialize(std::shared_ptr<snnlib::AbstractSNNConnection> connection)
    {
        int n_synapses = connection->weights.size();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0.0, 1.0);

        for(int i = 0; i < n_synapses; i++){
            if(!connection->connected[i]) continue;
            connection->weights[i] = d(gen);
        }
    }        
    
}


