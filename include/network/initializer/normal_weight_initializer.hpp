#ifndef NORMAL_WRIGHT_INITIALIZER_HPP
#define NORMAL_WRIGHT_INITIALIZER_HPP

#include <random>
#include "connections/connection.hpp"
#include "network/initializer/initializer.hpp"

namespace snnlib{
    struct NormalWeightInitializer: AbstractSNNConnectionWeightInitializer
    {
        public:
            virtual void initialize(std::shared_ptr<snnlib::AbstractSNNConnection> connection);
    };
}

#endif