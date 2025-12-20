#ifndef ENFORCED_REINFORCEMENT_HEADER
#define ENFORCED_REINFORCEMENT_HEADER

#include <functional>
#include <algorithm>
#include "../utils/utils.hpp"
#include "../model/model.hpp"

/**
 * @namespace Reinforcement
 * @brief Adds reinforcement learning to neural networks
 */
namespace Reinforcement {
    /**
     * @class Agent
     * @brief Extention to model class used in reinforcement learning
     * 
     * Adds a score function pointer, a modify function and comparison operators
     */
    class Agent : public Models::Model {
        protected:
            // Function pointer which should be used to score a neural network
            std::function<double(Models::Model)> score;
        
        public:
            /**
             * @brief Constructor for agent
             * 
             * @param score_function A function pointer to a function that scores neural networks
             * @param input_dim The input size of the neural network
             * @param hidden_dims The size of each hidden layer - each element in vector is a hidden layer dim
             * @param output_dim The output size of the neural network
             */
            Agent(const std::function<double(Models::Model)>& score_function, int input_dim, const std::vector<int>& hidden_dims, int output_dim);

            /**
             * @brief Modifies one weight or bias of one neuron
             * 
             * @param t The time value of how far through the training process we are through
             * @param randomness A temperature value - the higher it is the more extreme the changes are likely to be
             */
            void Modify(double t, double randomness);

            /**
             * @brief Scores the current neural network
             */
            double Score();

            /**
             * @brief Less than operator used for comparisons
             * 
             * @param other The other agent to compare this to
             * 
             * @return A boolean value whether this is less than other or not
             */
            bool operator<(const Agent& other) const;

            /**
             * @brief Greater than operator used for comparisons
             * 
             * @param other The other agent to compare this to
             * 
             * @return A boolean value whether this is greater than other or not
             */
            bool operator>(const Agent& other) const;

            /**
             * @brief Equal to operator used for comparisons
             * 
             * @param other The other agent to compare this to
             * 
             * @return A boolean value whether this is equal to other or not
             */
            bool operator==(const Agent& other) const;
    };

    /**
     * @class AgentBatch
     * @brief Contains many agents at once to perform reinforcement learning
     */
    class AgentBatch {
        protected:
            // All the agents to learn
            std::vector<Agent> agents;
            // The amount of agents learning
            int max_agents;
            // How far through the training process we are
            double t;
            // How random we want the learning to be
            double randomness;

        public:
            /**
             * @brief Constructor for agent batch
             * 
             * @param num_agents The amount of agents in the batch
             * @param randomness How random the training will be - higher for more random
             * @param score_function A function to score a neural network on how good it is
             * @param input_dim The inputs each network can take
             * @param hidden_dims The size of each hidden layer - each element in the vector is a hidden layer dim
             * @param output_dim The outputs each network can give
             */
            AgentBatch(int num_agents, double randomness, const std::function<double(Models::Model)>& score_function, int input_dim, const std::vector<int>& hidden_dims, int output_dim);

            /**
             * @brief Orders each network by score
             */
            void Order();

            /**
             * @brief Deletes the worst half of the batch
             */
            void Cut();

            /**
             * @brief Refills the agents after a cut and modifies the new copies
             */
            void Refill(double t);

            /**
             * @brief Orders, Cuts and Refills agents in a loop of max_epochs
             * 
             * @param max_epochs how many times the Order, Cut and Refill cycle is performed
             */
            void Learn(int max_epochs);
    };
}

#endif