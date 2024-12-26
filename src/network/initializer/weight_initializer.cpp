#include "network/initializer/weight_initializers.hpp"
#include <string>
namespace snnlib
{
    NormalWeightInitializer::NormalWeightInitializer(double scale){
        this->_scale = scale;
    }
    void NormalWeightInitializer::initialize(std::shared_ptr<snnlib::AbstractSNNConnection> connection)
    {
        int n_synapses = connection->weights.size();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0.0, 1.0);

        for(int i = 0; i < n_synapses; i++){
            if(!connection->connected[i]) continue;
            connection->weights[i] = d(gen) * _scale;
            // std::cout << "set weight = " << connection->weights[i] << std::endl;
        }
    };
    
    void IdenticalWeightInitializer::initialize(std::shared_ptr<snnlib::AbstractSNNConnection> connection){
        int n_synapses = connection->weights.size();
        
        for(int i = 0; i < n_synapses; i++){
            if(!connection->connected[i]) continue;
            connection->weights[i] = this->_weight;
        }
    }

    void SpecificWeightInitializer::initialize(std::shared_ptr<snnlib::AbstractSNNConnection> connection){
        int n_synapses = connection->weights.size();
        if(this->_weights.size() != connection->weights.size()){
            throw std::invalid_argument(std::string("Connection weight initialization with SpecificWeightInitializer \
            should receive a connection, which has the same amount of synapses as the amount of elements of the weights provided. \
            size of weights provided = ") + std::to_string(this->_weights.size()) + 
            std::string(" != connection's size of weights = ") +
            std::to_string(connection->weights.size()));
        }
        std::copy(this->_weights.begin(), this->_weights.end(), connection->weights.begin());
    }
    
}


