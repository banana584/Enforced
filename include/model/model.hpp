#ifndef ENFORCED_MODEL_HEADER
#define ENFORCED_MODEL_HEADER

#include <iostream>
#include <vector>
#include <cmath>
#include "../utils/utils.hpp"

namespace Reinforcement {
    class Agent;
}

/**
 * @namespace Models
 * @brief Contains classes relating to neural networks
 */
namespace Models {
    // Forward declaration
    class Model;

    /**
     * @class Neuron
     * @brief A singular neuron used in a neural network
     */
    class Neuron {
        private:
            // The weights for the incoming data
            std::vector<double> weights_in;
            // The bias applied in the weighted sum
            double bias;
        
        private:
            /**
             * @brief Sigmoid activation function
             * 
             * @see https://en.wikipedia.org/wiki/Sigmoid_function
             * 
             * @param x The input to the function
             * 
             * @return The output of the function
             */
            double Sigmoid(double x) const;

            /**
             * @brief Utility dot product function
             * 
             * @see https://en.wikipedia.org/wiki/Dot_product
             * 
             * @param x The first vector in the dot
             * @param y The second vector in the dot
             * 
             * @return The resulting number of the dot product
             */
            double Dot(const std::vector<double>& x, const std::vector<double>& y) const;
        
        public:
            /**
             * @brief Constructor to initialize weights and bias randomly
             * 
             * @param num_weights_in The number of weights inputting to this neuron
             */
            Neuron(int num_weights_in);

            /**
             * @brief Creates a prediction from inputs, weights and bias
             * 
             * This takes in inputs, multiplies by weights using the dot product and adds a bias to the final result
             * 
             * @param inputs The inputs used in the dot product with weights
             * 
             * @return The weighted sum - Dot product + bias
             */
            double WeightedSum(const std::vector<double>& inputs) const;

            /**
             * @brief Neuron assignment operator - sets this to other
             * 
             * @param other The neuron to copy values into this
             * 
             * @return The neuron with the copied values from other
             */
            Neuron& operator=(const Neuron& other);

        friend Model;
        friend Reinforcement::Agent;
    };

    class Model {
        protected:
            // The dim of the inputs of the network
            int input_dim;
            // The hidden layers of the network
            std::vector<std::vector<Neuron>> hidden;
            // The output layer of the network
            std::vector<Neuron> outputs;

        public:
            /**
             * @brief Constructor for model
             * 
             * @param input_dim The size of inputs given to the model
             * @param hidden_dims A vector of the size for each layer - each element is a dim for each layer
             * @param output_dim The number of outputs the model can give
             */
            Model(int input_dim, const std::vector<int>& hidden_dims, int output_dim);

            Model();

            /**
             * @brief Copy constructor for model
             * 
             * @param other The other model to copy values into this from
             */
            Model(const Model& other);

            /**
             * @brief Pass inputs through model to make a prediction
             * 
             * @param input The inputs given to the model to pass through
             * 
             * @return The outputs after the input was passed through the model
             */
            std::vector<double> Forward(const std::vector<double>& input) const;

            /**
             * @brief Assignment operator between models
             * 
             * @param other The other model to copy values into this from
             * 
             * @return This after the values were copied into it
             */
            Model& operator=(const Model& other);
    };
}

#endif