#ifndef ENFORCED_MODEL_HEADER
#define ENFORCED_MODEL_HEADER

#include <iostream>
#include <vector>
#include <cmath>
#include "../utils/utils.hpp"

namespace Models {
    class Neuron {
        private:
            std::vector<double> weights_in;
            double bias;
        
        private:
            double Sigmoid(double x) const;

            double Dot(const std::vector<double>& x, const std::vector<double>& y) const;
        
        public:
            Neuron(int num_weights_in);

            double WeightedSum(const std::vector<double>& inputs) const;

            Neuron& operator=(const Neuron& other);
    };

    class Model {
        private:
            int input_dim;
            std::vector<std::vector<Neuron>> hidden;
            std::vector<Neuron> outputs;

        public:
            Model(int input_dim, const std::vector<int>& hidden_dims, int output_dim);

            Model(const Model& other);

            std::vector<double> Forward(const std::vector<double>& input) const;

            Model& operator=(const Model& other);
    };
}

#endif