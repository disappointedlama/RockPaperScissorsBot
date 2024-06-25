#include<cstdio>
#include<cstring>
#include<iostream>
#include<random>
#include<array>
#include<vector>
#include<cmath>
#include<cstring>
#include<memory>
using std::array, std::vector;
enum Move {
    ROCK, PAPER, SCISSORS
};
enum Result {
    LOST, WON, DRAW
};
static constexpr size_t input_size{ 5 };
static constexpr size_t random_play{ 20 };
static constexpr size_t training_start{ 20 };
static constexpr size_t hist_max{(1ULL<<(2*input_size+2))-1};
struct History{
    unsigned long long hist{0};
    inline void push(const Move move){
        hist = (hist>>2) | (size_t)move << (2*input_size);
    }
    unsigned long long& get(){return hist;}
};
struct Engine {
    History hist{};
    vector<Move> moves{};
    vector<Move> opponent_moves{};
    array<array<bool, 3>, input_size> last_moves{};
    std::unique_ptr<array<array<size_t,4>,hist_max>> occurences{std::make_unique<array<array<size_t,4>,hist_max>>()};
    Move next() {
        Move ret;
        if(moves.size()<random_play){
            ret = (Move)(std::rand()%3);
        }
        else{
            size_t index{0};
            size_t val{0};
            for(size_t i=0;i<3;++i){
                if(occurences->at(hist.get())[i]>val){
                    val = occurences->at(hist.get())[i];
                    index = i;
                }
            }
            static constexpr array<Move,3> to_winning{PAPER,SCISSORS,ROCK};
            ret=to_winning[index];
        }
        moves.push_back(ret);
        return ret;
    }
    void parse_result(const Result res) {
        static constexpr array<Move, 9> to_opponents_move{ {PAPER,SCISSORS,ROCK,SCISSORS,ROCK,PAPER,ROCK,PAPER,SCISSORS} };
        opponent_moves.push_back(to_opponents_move[3 * res + moves.back()]);
        if(opponent_moves.size()<training_start) return;
        ++(occurences->at(hist.get())[opponent_moves.back()]);
        ++(occurences->at(hist.get())[3]);
        if(occurences->at(hist.get())[3]%4==0){
            size_t& occ{occurences->at(hist.get())[(opponent_moves.back()+1)%3]};
            if(occ){
                --occ;
            }
            occ = occurences->at(hist.get())[(opponent_moves.back()+2)%3];
            if(occ){
                --occ;
            }
        }
        hist.push(opponent_moves.back());
        hist.push(moves.back());
    }
};
int main() {
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