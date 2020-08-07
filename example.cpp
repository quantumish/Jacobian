//
//  example.cpp
//  Jacobian
//
//  Created by David Freifeld
//

#include <indicators/cursor_control.hpp>
#include <indicators/progress_bar.hpp>
#include <indicators/block_progress_bar.hpp>
using namespace indicators;

#include "./src/bpnn.hpp"
#include "./src/utils.hpp"
#include "unistd.h"
#include <ctime>

double bench(int batch_sz)
{
    auto start = std::chrono::high_resolution_clock::now();
    Network net ("./data_banknote_authentication.txt", batch_sz, 0.0155, 0.03, 2, 0, 0.9);
    net.add_layer(4, "linear");
    net.add_layer(5, "relu");
    net.add_layer(2, "linear");
    net.initialize();
    for (int i = 0; i < 1000; i++) {
        net.train();
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / pow(10,9);
}

int main()
{
    std::cout << bench(16) << "\n";
    // show_console_cursor(false);
    // BlockProgressBar bar{
    //   option::BarWidth{80},
    //   option::Start{"["},
    //   option::End{"]"},
    //   option::ForegroundColor{Color::white}  ,
    //   option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}
    // };
    // Network net ("./data_banknote_authentication.txt", 16, 0.0155, 0.03, 2, 0, 0.9);
    // net.add_layer(4, "linear");
    // net.add_layer(5, "relu");
    // net.add_layer(2, "linear");
    // net.initialize();
    // bar.set_option(option::PostfixText{"Starting train"});
    // for (int i = 0; i < 5000; i++) {
    //   char msg[32];
    //   sprintf(msg, "Starting epoch %i", i);
    //   std::string str(msg);
    //   bar.set_option(option::PostfixText{str});
    //   net.train();
    //   bar.set_progress((float)i/50 * 100);
    // }
    // bar.set_progress(100); // Ensure we are done.
    // std::cout << "\nFinal cost: " << net.get_cost() << " Final validation cost:" << net.get_val_cost() << "\n";
    // show_console_cursor(true);
}
