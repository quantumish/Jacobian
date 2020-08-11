//
//  checks.cpp
//  Jacobian
//
//  Created by David Freifeld
//

#include "./src/bpnn.hpp"
 
#define ZERO_THRESHOLD pow(10, -5)

Network default_net()
{
    Network net ("./data_banknote_authentication.txt", 16, 0.0155, 0.03, 2, 0, 0.9);
    net.add_layer(4, "linear");
    net.add_layer(5, "lecun_tanh");
    net.add_layer(2, "linear");
    net.initialize();
    return net;
}

Network explicit_copy(Network src)
{
    Network dst ("./data_banknote_authentication.txt", 16, 0.0155, 0.03, 2, 0, 0.9);
    dst = src;
    for (int i = 0; i < src.layers.size(); i++) {
        dst.layers[i] = src.layers[i];
    }
    return dst;
}

void regularization_check(int& sanity_passed, int& total_checks)
{
    Network net = default_net();
    std::cout << "\u001b[4m\u001b[1mSANITY CHECKS:\u001b[0m\n";
    // Check if regularization strength increases loss (as it should).
    std::cout << "Regularization sanity check...";

    net.list_net();
    Network copy1 = explicit_copy(net);
    Network copy2 = explicit_copy(net);
    copy1.next_batch();
    copy1.feedforward();
  
    copy2.next_batch();
    copy2.feedforward();
    net.list_net();
    if (copy1.cost() > copy2.cost()) {
        std::cout << " \u001b[32mPassed!\n\u001b[37m";
        sanity_passed++;
    }
    else std::cout << " \u001b[31mFailed.\n\u001b[37m";
    total_checks++;
}

void zero_check(int& sanity_passed, int& total_checks)
{
    Network net = default_net();
    std::cout << "Zero-cost sanity check...";
    net.next_batch();
    float finalcost;
    for (int i = 0; i < 100000; i++) {
      net.feedforward();
      net.backpropagate();
      finalcost = net.cost();
      if (finalcost <= ZERO_THRESHOLD) {
        break;
      }
    }
    if (finalcost <= ZERO_THRESHOLD) {
      std::cout << " \u001b[32mPassed!\n\u001b[37m";
      sanity_passed++;
    }
    else std::cout << " \u001b[31mFailed.\n\u001b[37m";
    total_checks++;
}

void floating_point_check(int& sanity_passed, int& total_checks)
{
    Network net = default_net();
    std::cout << "Gradient floating-point sanity check...";
    net.next_batch();
    net.feedforward();
    std::vector<Eigen::MatrixXf> gradients;
    std::vector<Eigen::MatrixXf> deltas;
    Eigen::MatrixXf error = ((*net.layers[net.length-1].contents) - (*net.labels));
    gradients.push_back(error.cwiseProduct(*net.layers[net.length-1].dZ));
    deltas.push_back((*net.layers[net.length-2].contents).transpose() * gradients[0]);
    int counter = 1;
    for (int i = net.length-2; i >= 1; i--) {
      gradients.push_back((gradients[counter-1] * net.layers[i].weights->transpose()).cwiseProduct(*net.layers[i].dZ));
      deltas.push_back(net.layers[i-1].contents->transpose() * gradients[counter]);
      counter++;
    }
    auto check_gradients = [](std::vector<Eigen::MatrixXf> vec) -> bool {
      for (Eigen::MatrixXf i : vec) {
        for (int j = 0; j < i.rows(); j++) {
          for (int k = 0; k < i.cols(); k++) {
            if (i(j,k) == -0 || i(j,k) == INFINITY || i(j,k) == NAN || i(j,k) == -INFINITY) {
              return true;
            }
          }
        }
      }
      return false;
    };
    if (check_gradients(gradients) == false && check_gradients(deltas) == false) {
      std::cout << " \u001b[32mPassed!\n\u001b[37m";
      sanity_passed++;
    }
    else std::cout << " \u001b[31mFailed.\n\u001b[37m";
    total_checks++;
}

void expected_loss_check(int& sanity_passed, int& total_checks)
{
    Network net = default_net();
    std::cout << "Expected loss sanity check...";
    net.next_batch();
    net.feedforward();
    if (net.cost() <= 1) {
      std::cout << " \u001b[32mPassed!\n\u001b[37m";
      sanity_passed++;
    }
    else std::cout << " \u001b[31mFailed.\n\u001b[37m";
    total_checks++;
}

void update_check(int& sanity_passed, int& total_checks)
{
    // std::cout << "Layer updates sanity check...";
    // Network copy6 ("./data_banknote_authentication.txt", 16, 0.05, 0.03, 0, 0.9);
    // Network copy7 ("./data_banknote_authentication.txt", 16, 0.05, 0.03, 0, 0.9);
    // //copy2.list_net();
    // //copy1.list_net();
    // int passed;
    // for (int i = 0; i < copy1.layers.size()-1; i++) {
    //   if (*copy1.layers[i].weights == *copy2.layers[i].weights) {
    //     //      std::cout << *copy2.layers[i].weights <<"uninitweight\n\n";
    //     //      std::cout << *copy1.layers[i].weights << " "<<i<<"weight\n\n";
    //     passed = -1;
    //   }
    // }
    // for (int i = 1; i < copy1.layers.size(); i++) {
    //   if (*copy1.layers[i].bias == *copy2.layers[i].bias) {
    //     //      std::cout << *copy2.layers[i].bias <<"uninitbias\n\n";
    //     //  std::cout << *copy1.layers[i].bias <<" " << i << "bias\n\n";
    //     passed = -1;
    //   }
    // }
    // if (passed == 1) {
    //   std::cout << " \u001b[32mPassed!\n\u001b[37m";
    //   sanity_passed++;
    // }
    // else std::cout << " \u001b[31mFailed.\n\u001b[37m";
}

void sanity_checks()
{
    int sanity_passed = 0;
    int total_checks = 0;
    zero_check(sanity_passed, total_checks);
    std::cout << "\u001b[1m\nPassed " << sanity_passed << "/" << total_checks <<" sanity checks.\u001b[0m\n";
    if ((float)sanity_passed/total_checks < 0.5) {
        std::cout << "Majority of sanity checks failed. Exiting." << "\n";
        exit(1);
    }
}

void grad_checks()
{
}
