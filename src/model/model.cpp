#include "../../include/model/model.hpp"

namespace Models {
    double Neuron::Sigmoid(double x) const {
        return 1 / (1 + exp(-x));
    }

    double Neuron::Dot(const std::vector<double>& x, const std::vector<double>& y) const {
        double dot = 0.0;

        for (size_t i = 0; i < x.size(); i++) {
            dot += x[i] * y[i];
        }

        return dot;
    }

    Neuron::Neuron(int num_weights_in) {
        for (int i = 0; i < num_weights_in; i++) {
            weights_in.emplace_back(GetRandomNumber<double>(0.1, 2));
        }

        bias = GetRandomNumber<double>(0, 2);
    }
    
    double Neuron::WeightedSum(const std::vector<double>& inputs) const {
        return Sigmoid(Dot(inputs, weights_in) + bias);
    }

    Neuron& Neuron::operator=(const Neuron& other) {
        if (this == &other) {
            return *this;
        }

        this->weights_in = other.weights_in;
        this->bias = other.bias;

        return *this;
    }



    Model::Model(int input_dim, const std::vector<int>& hidden_dims, int output_dim) {
        this->input_dim = input_dim;

        int last_dim = input_dim;
        for (size_t i = 0; i < hidden_dims.size(); i++) {
            std::vector<Neuron> layer;
            for (int j = 0; j < hidden_dims[i]; j++) {
                layer.emplace_back(Neuron(last_dim));
            }
            last_dim = hidden_dims[i];
            this->hidden.emplace_back(layer);
        }

        for (int i = 0; i < output_dim; i++) {
            this->outputs.emplace_back(Neuron(last_dim));
        }
    }

    Model::Model() {
        
    }

    Model::Model(const Model& other) {
        this->input_dim = other.input_dim;
        this->hidden = other.hidden;
        this->outputs = other.outputs;
    }

    std::vector<double> Model::Forward(const std::vector<double>& input) const {
        std::vector<double> layer_1;
        for (size_t i = 0; i < hidden[0].size(); i++) {
            double prediction = hidden[0][i].WeightedSum(input);
            layer_1.emplace_back(prediction);
        }

        std::vector<double> hidden_layer = input;
        for (size_t i = 0; i < hidden.size(); i++) {
            std::vector<double> layer;
            for (size_t j = 0; j < hidden[i].size(); j++) {
                double prediction = hidden[i][j].WeightedSum(hidden_layer);
                layer.emplace_back(prediction);
            }
            hidden_layer = layer;
        }

        std::vector<double> layer_2;
        for (size_t i = 0; i < outputs.size(); i++) {
            double prediction = outputs[i].WeightedSum(hidden_layer);
            layer_2.emplace_back(prediction);
        }

        return layer_2;
    }

    Model& Model::operator=(const Model& other) {
        if (this == &other) {
            return *this;
        }

        this->input_dim = other.input_dim;
        this->hidden = other.hidden;
        this->outputs = other.outputs;

        return *this;
    }
}