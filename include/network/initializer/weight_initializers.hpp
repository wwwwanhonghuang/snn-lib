#ifndef NORMAL_WRIGHT_INITIALIZER_HPP
#define NORMAL_WRIGHT_INITIALIZER_HPP

#include <random>
#include "connections/connection.hpp"
#include "network/initializer/initializer.hpp"

namespace snnlib{
    struct NormalWeightInitializer: AbstractSNNConnectionInitializer
    {
        public:
            virtual void initialize(std::shared_ptr<snnlib::AbstractSNNConnection> connection);
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
}

#endif