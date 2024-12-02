#ifndef INITIALIZER_HPP
#define INITIALIZER_HPP
#include "connections/connection.hpp"

namespace snnlib{
    struct AbstractSNNConnection;
    struct AbstractSNNConnectionInitializer
    {
        virtual void initialize(std::shared_ptr<snnlib::AbstractSNNConnection> connection) = 0;
    };
}

#endif