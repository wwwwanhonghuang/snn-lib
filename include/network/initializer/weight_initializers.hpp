#ifndef NORMAL_WRIGHT_INITIALIZER_HPP
#define NORMAL_WRIGHT_INITIALIZER_HPP

#include <random>
#include <vector>
#include "connections/connection.hpp"
#include "network/initializer/initializer.hpp"

namespace snnlib{
    struct NormalWeightInitializer: AbstractSNNConnectionInitializer
    {
        private:
            double _scale = 1.0;
        public:
            virtual void initialize(std::shared_ptr<snnlib::AbstractSNNConnection> connection);
            NormalWeightInitializer(){}
            NormalWeightInitializer(double scale);
    };

    struct IdenticalWeightInitializer: AbstractSNNConnectionInitializer{
        private:
            double _weight;
        public:
            IdenticalWeightInitializer(double weight){
                this->_weight = weight;
            }
            virtual void initialize(std::shared_ptr<snnlib::AbstractSNNConnection> connection);
    };

    struct SpecificWeightInitializer: AbstractSNNConnectionInitializer{
        private:
            std::vector<double> _weights;
        public:
            SpecificWeightInitializer(std::vector<double> weights): _weights(weights)
            {
                
            }
            virtual void initialize(std::shared_ptr<snnlib::AbstractSNNConnection> connection);
    };

    
}

#endif