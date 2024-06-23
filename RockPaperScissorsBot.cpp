﻿#include<cstdio>
#include<cstring>
#include<iostream>
#include<random>
#include<array>
#include<vector>
#include<cmath>
using std::array, std::vector;
enum Move {
    ROCK, PAPER, SCISSORS
};
enum Result {
    LOST, WON, DRAW
};
static constexpr size_t input_size{ 10 };
static constexpr size_t layer_size{ 100 };
static constexpr size_t layer_count{ 100 };
static constexpr float learning_rate{ 0.5f };
vector<float> empty_deltas{};

inline void clear_deltas() {
    for (int i = 0; i < empty_deltas.size(); ++i) {
        empty_deltas[i]=0.0f;
    }
}

static constexpr array<float, layer_size> empty_layer{ []()consteval {
    array<float,layer_size> ret{};
    for (int i = 0; i < layer_size; ++i) {
        ret[i] = 0.0f;
    }
    return ret;
}() };

inline float sigmoid(float x) {
    return 1 / (1 + std::exp(-x));
}
/*
the weights between layer l − 1 and l, where (w_i,j)^l is the weight between the j-th node in layer l − 1 and the i-th node in layer l
*/
struct Network {
    array<array<bool, 3>, input_size> input_nodes{};
    array<float, 3 * input_size * layer_size> input_layer_weights{};
    array<array<float, layer_size * layer_size>, layer_count - 1> layer_weights{};
    array<float, 3 * layer_size> output_layer_weights{};
    array<float, 3> output_nodes{};
    array<array<float, layer_size>, layer_count> weighted_inputs{};
    array<array<float, layer_size>, layer_count> weighted_activations{};
    size_t counter;
    Network() : counter{ 0 } {
        for (size_t i = 0; i < output_layer_weights.size(); ++i) {
            output_layer_weights[i] = float(-0.5 + 0.01 * (std::rand() % 100));
            //output_layer_weights[i] = 1.0f;
        }
        for (int i = 0; i < input_layer_weights.size(); ++i) {
            input_layer_weights[i] = 1.0f;
        }
        for (int i = 0; i < layer_weights.size(); ++i) {
            size_t counter{ 0 };
            for (int j = 0; j < layer_size * layer_size; ++j) {
                //layer_weights[i][j] = 1.0f;
                layer_weights[i][j] = float(-0.5 + 0.01 * (std::rand() % 100));
                /*
                layer_weights[i][j] = 0.5f;
                if ((counter++) % 2 == 0) {
                    layer_weights[i][j] = 1.0f;
                }
                else {
                    layer_weights[i][j] = -0.5f;
                }
                */
            }
        }
    }
    void train(const Result res, const Move should_have_played) {
        const array<float, 3> target{ 0.5f * (should_have_played == PAPER) + 1.0f * (should_have_played == ROCK),
            0.5f * (should_have_played == SCISSORS) + 1.0f * (should_have_played == PAPER),
            0.5f * (should_have_played == ROCK) + 1.0f * (should_have_played == SCISSORS) };
        //std::cout <<"Target: " << target[0] << ", " << target[1] << ", " << target[2] << "\n";
        array<array<float, layer_size* layer_size>, layer_count - 1> new_weights{};
        array<float, 3 * layer_size> new_output_weights{};
        //array<float, 3 * layer_size * input_size> new_input_weights{};
        for (size_t i = 0; i < layer_size; ++i) {
            const size_t offset{ i * 3 };
            for (size_t j = 0; j < 3; ++j) {
                new_output_weights[offset + j] = output_layer_weights[offset + j] - learning_rate * weighted_activations.back()[i] * (output_nodes[j] - target[j]) * output_nodes[j] * (1 - output_nodes[j]);
                //std::cout << weighted_activations.back()[i] << "\n"<< output_nodes[j]<<"\n";
                //std::cout << output_layer_weights[offset + j] << "-> " << new_output_weights[offset + j] << "\n";
            }
        }
        /*
        for (int j = 0; j < layer_weights.size(); ++j) {
            for (int i = 0; i < layer_weights[j].size(); ++i) {
                new_weights[layer_weights.size() - 1][j * layer_size + i] = output_layer_weights[3ULL * j + i] - learning_rate * weighted_activations[layer_weights.size() - 1][j] * delta(layer_weights.size() - 1, j, target);
            }
        }
        */
        clear_deltas();
        for (int l = (int)layer_weights.size()-1; l > -1; --l) {
            for (int i = 0; i < layer_size; ++i) {
                for (int j = 0; j < layer_size; ++j) {
                    const float d{ delta(l, j, target, empty_deltas) };
                    //std::cout << d << " (delta)\n";
                    new_weights[l][j * layer_size + i] = layer_weights[l][layer_size * j + i] - learning_rate * weighted_activations[l+1][i] * d;
                    //std::cout << weighted_activations.back()[i] << "\n"<< output_nodes[j]<<"\n";
                    //std::cout << layer_weights[l][layer_size * j + i] << " -> " << new_weights[l][j * layer_size + i] << "\n";
                }
            }
        }
        //for (int i = 0; i < layer_size; ++i) {
        //    for (int j = 0; j < 3; ++j) {
        //        std::cout << new_output_weights[3 * i + j] << ",";
        //    }
        //    std::cout << "__";
        //}
        layer_weights = new_weights;
        output_layer_weights = new_output_weights;
        //input_layer_weights = new_input_weights;
    }
    inline float delta(const size_t layer, const size_t j, const array<float, 3>& target, vector<float>& known_deltas) const {
        if (known_deltas[layer*layer_size + j] != 0.0f) {
            return known_deltas[layer*layer_size + j];
        }
        const float factor{ weighted_activations[layer][j] * (1 - weighted_activations[layer][j]) };
        if (layer == layer_weights.size() - 1) {
            float acc{ 0 };
            for (int i = 0; i < output_nodes.size(); ++i) {
                acc += output_layer_weights[3 * j + i] * output_nodes[i] * (output_nodes[i] - target[i]) * output_nodes[i] * (1 - output_nodes[i]);
            }
            //std::cout << acc * factor << " (delta)\n";
            const float ret{ acc * factor };
            known_deltas[layer*layer_size + j] = ret;
            return ret;
        }
        float acc{ 0 };
        for (int i = 0; i < layer_size; ++i) {
            acc += layer_weights[layer][j * layer_size + i] * delta(layer + 1, i, target, known_deltas);
        }
        const float ret{ acc * factor };
        known_deltas[layer*layer_size + j] = ret;
        return factor * ret;
    }
    void run() {
        array<array<float, layer_size>, 2> values{ {empty_layer,empty_layer} };
        bool side{ false };
        for (int i = 0; i < input_nodes.size(); ++i) {
            for (int j = 0; j < input_nodes[i].size(); ++j) {
                const float val{ 1.0f * (input_nodes[i][j]) };
                const size_t weight_offset{ 3UL * i + j};
                for (int k = 0; k < layer_size; ++k) {
                    values[(side)][k] += input_layer_weights[weight_offset + k] * val;
                }
            }
        }
        for (int i = 0; i < layer_size; ++i) {
            const float num{ sigmoid(values[(side)][i]) };
            values[(side)][i] = num;
            weighted_activations[0][i] = num;
        }
        side = !side;
        for (size_t i = 0; i < layer_weights.size(); ++i) {
            const array<float, layer_size * layer_size>& layer{ layer_weights[i] };
            const size_t value_index{ (side) };
            const size_t source_value_index{ (!side) };
            array<float, layer_size>& target = values[(side)];
            target = empty_layer;
            array<float, layer_size>& source = values[(!side)];
            array<float, layer_size> outputs{ empty_layer };
            for (int j = 0; j < layer_size; ++j) {
                const size_t offset{ layer_size * j };
                for (int k = 0; k < layer_size; ++k) {
                    target[k] += source[k] * layer[offset + k];
                }
            }
            for (int j = 0; j < layer_size; ++j) {
                const float num{ sigmoid(target[j]) };
                outputs[j] = num;
                target[j] = num;
            }
            weighted_activations[i+1] = outputs;
            side = !side;
        }
        output_nodes = { 0,0,0 };
        for (size_t i = 0; i < layer_size; ++i) {
            const size_t offset{ i * 3 };
            const float val{ values[(side)][i] };
            for (size_t j = 0; j < 3; ++j) {
                output_nodes[j] += val * output_layer_weights[offset + j];
            }
        }
        for (int i = 0; i < output_nodes.size(); ++i) {
            output_nodes[i] = sigmoid(output_nodes[i]);
        }
        //std::cout << "\nOutput: " << output_nodes[0] << "," << output_nodes[1] << "," << output_nodes[2] << "\n";
    }
    Move next(const array<array<bool, 3>, input_size>& inputs) {
        input_nodes = inputs;
        run();
        float value{ std::numeric_limits<float>::min() };
        int ret{ 0 };
        for (int i = 0; i < output_nodes.size(); ++i) {
            if (output_nodes[i] > value) {
                value = output_nodes[i];
                ret = i;
            }
        }
        //play randomly for the first 200 moves
        //(the model is still being trained though)
        if (counter++ < 200) return (Move)(std::rand() % 3);
        return (Move)ret;
    }
};
struct Engine {
    Network net{};
    vector<Move> moves{};
    vector<Move> opponent_moves{};
    array<array<bool, 3>, input_size> last_moves{};
    Move next() {
        const Move ret{ net.next(last_moves) };
        moves.push_back(ret);
        return ret;
    }
    void parse_result(const Result res) {
        static constexpr array<Move, 9> to_opponents_move{ {PAPER,SCISSORS,ROCK,SCISSORS,ROCK,PAPER,ROCK,PAPER,SCISSORS} };
        opponent_moves.push_back(to_opponents_move[3 * res + moves.back()]);
        for (size_t i = last_moves.size() - 1; i > 0; --i) {
            last_moves[i] = last_moves[i - 1];
        }
        array<bool, 3> arr{ false,false,false };
        arr[opponent_moves.back()] = true;
        last_moves[0] = arr;
        /*
        for (int i = 0; i < last_moves.size(); ++i) {
            for (int j = 0; j < last_moves[0].size(); ++j) {
                std::cout << ((last_moves[i][j]) ? 1.0f : 0.0f) << ",";
            }
            std::cout << "__";
        }
        std::cout << std::endl;
        */
        Move should_have_played;
        if (res == WON) {
            should_have_played = moves.back();
        }
        else {
            static constexpr array<Move, 3> to_win{ PAPER,SCISSORS,ROCK };
            should_have_played = to_win[opponent_moves.back()];
        }
        net.train(res, should_have_played);
    }
};
int main() {
    for (int i = 0; i < layer_size * layer_count; ++i) {
        empty_deltas.push_back(0.0f);
    }
    static constexpr size_t buff_size{ 20 };
    char input[buff_size]{};
    fflush(stdin);
    fflush(stdout);
    Engine bot{};
    while (true) {
        memset(input, 0, sizeof(input));
        fflush(stdout);
        if (!fgets(input, buff_size, stdin)) {
            continue;
        }
        if (input[0] == '\n') {
            continue;
        }
        if (strncmp(input, "isready", 7) == 0) {
            std::cout << "readyok" << std::endl;
        }
        else if (strncmp(input, "next", 4) == 0) {
            std::cout << bot.next() << std::endl;
        }
        else if (strncmp(input, "0", 1) == 0) {
            bot.parse_result(LOST);
        }
        else if (strncmp(input, "1", 1) == 0) {
            bot.parse_result(WON);
        }
        else if (strncmp(input, "2", 1) == 0) {
            bot.parse_result(DRAW);
        }
        else if (strncmp(input, "quit", 4) == 0) {
            break;
        }
    }
    return 0;
}