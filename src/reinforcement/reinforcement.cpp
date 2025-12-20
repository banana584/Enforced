#include "../../include/reinforcement/reinforcement.hpp"

namespace Reinforcement {
    Agent::Agent(const std::function<double(Models::Model)>& score_function, int input_dim, const std::vector<int>& hidden_dims, int output_dim) {
        this->score = score_function;
        
        this->input_dim = input_dim;

        int last_dim = input_dim;
        for (size_t i = 0; i < hidden_dims.size(); i++) {
            std::vector<Models::Neuron> layer;
            for (int j = 0; j < hidden_dims[i]; j++) {
                layer.emplace_back(Models::Neuron(last_dim));
            }
            last_dim = hidden_dims[i];
            this->hidden.emplace_back(layer);
        }

        for (int i = 0; i < output_dim; i++) {
            this->outputs.emplace_back(Models::Neuron(last_dim));
        }
    }

    void Agent::Modify(double t, double randomness) {
        double new_randomness = randomness / std::max(t, 0.1);

        size_t hidden_max_size = 0;
        for (size_t i = 0; i < hidden.size(); i++) {
            hidden_max_size += hidden[i].size();
        }
        size_t max_size = hidden_max_size + outputs.size();

        size_t index = GetRandomNumber<int>(0, (int)max_size);
        bool outputs = false;

        if (index > hidden_max_size) {
            index -= hidden_max_size;
            outputs = true;
        }

        Models::Neuron* neuron = nullptr;

        if (outputs) {
            neuron = &this->outputs[index];
        } else {
            int hidden_traversed = 0;
            for (size_t i = 0; i < hidden.size(); i++) {
                if (index > i && index < hidden_traversed + (hidden[i].size())) {
                    neuron = &hidden[i][index];
                }
            }
        }

        max_size = neuron->weights_in.size() + 1;

        index = GetRandomNumber<int>(0, max_size);

        double* modify;
        if (index > neuron->weights_in.size()) {
            index -= neuron->weights_in.size();
            modify = &neuron->bias;
        } else {
            modify = &neuron->weights_in[index];
        }

        *modify += GetRandomNumber<double>(-2 * new_randomness, 2 * new_randomness);
    }

    double Agent::Score() {
        return score(*this);
    }

    bool Agent::operator<(const Agent& other) const {
        return this->score(*this) < other.score(other);
    }

    bool Agent::operator>(const Agent& other) const {
        return this->score(*this) > other.score(other);
    }

    bool Agent::operator==(const Agent& other) const {
        return this->score(*this) == other.score(other);
    }



    AgentBatch::AgentBatch(int num_agents, double randomness, const std::function<double(Models::Model)>& score_function, int input_dim, const std::vector<int>& hidden_dims, int output_dim) {
        this->max_agents = num_agents;
        this->t = 0.0;
        this->randomness = randomness;
        for (int i = 0; i < num_agents; i++) {
            this->agents.emplace_back(score_function, input_dim, hidden_dims, output_dim);
        }
    }

    void AgentBatch::Order() {
        std::sort(agents.begin(), agents.end());
    }

    void AgentBatch::Cut() {
        agents = std::vector<Agent>(agents.begin() + (agents.size() / 2), agents.end());
    }

    void AgentBatch::Refill(double t) {
        for (size_t i = 0; i < agents.size(); i++) {
            agents.emplace_back(agents[i]);
            agents.end()->Modify(t, randomness);
        }
    }

    void AgentBatch::Learn(int max_epochs) {
        for (int epoch = 0; epoch < max_epochs; epoch++) {
            Order();
            Cut();
            Refill(t);
            t = epoch / max_epochs;
        }
    }
}