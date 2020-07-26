//
//  checks.cpp
//  Jacobian
//
//  Created by David Freifeld
//  Copyright © 2020 David Freifeld. All rights reserved.
//

void checks(Network net)
{
  Network original = net;
  int sanity_passed = 0;
  std::cout << "\u001b[4m\u001b[1mSANITY CHECKS:\u001b[0m\n";
  // Check if regularization strength increases loss (as it should).
  std::cout << "Regularization sanity check...";

  //  list_net();
  
  Network copy1 ("./data_banknote_authentication.txt", 16, 0.05, 0.03, 0, 0.9);
  Network copy2 ("./data_banknote_authentication.txt", 16, 0.05, 0.03, 0, 0.9);
  copy1.lambda += 1;
  copy1.next_batch();
  copy1.feedforward();
  copy2.next_batch();
  copy2.feedforward();
  if (copy1.cost() > copy2.cost()) {
    std::cout << " \u001b[32mPassed!\n\u001b[37m";
    sanity_passed++;
  }
  else std::cout << " \u001b[31mFailed.\n\u001b[37m";

  //  net.list_net();
  
  // Check if zero cost is achievable on a batch
  std::cout << "Zero-cost sanity check...";
  Network copy3 ("./data_banknote_authentication.txt", 16, 0.05, 0.03, 0, 0.9);
  copy3.lambda = 0;
  copy3.next_batch();
  float finalcost;
  for (int i = 0; i < 10000; i++) {
    copy3.feedforward();
    copy3.backpropagate();
    finalcost = copy3.cost();
    if (finalcost <= ZERO_THRESHOLD) {
      break;
    }
  }
  if (finalcost <= ZERO_THRESHOLD) {
    std::cout << " \u001b[32mPassed!\n\u001b[37m";
    sanity_passed++;
  }
  else std::cout << " \u001b[31mFailed.\n\u001b[37m";

  //  list_net();
  
  std::cout << "Gradient floating-point sanity check...";
  Network copy4 ("./data_banknote_authentication.txt", 16, 0.05, 0.03, 0, 0.9);
  copy4.next_batch();
  copy4.feedforward();
  std::vector<Eigen::MatrixXf> gradients;
  std::vector<Eigen::MatrixXf> deltas;
  Eigen::MatrixXf error = ((*copy4.layers[copy4.length-1].contents) - (*copy1.labels));
  gradients.push_back(error.cwiseProduct(*copy4.layers[copy4.length-1].dZ));
  deltas.push_back((*copy4.layers[copy4.length-2].contents).transpose() * gradients[0]);
  int counter = 1;
  for (int i = copy4.length-2; i >= 1; i--) {
    gradients.push_back((gradients[counter-1] * copy4.layers[i].weights->transpose()).cwiseProduct(*copy4.layers[i].dZ));
    deltas.push_back(copy4.layers[i-1].contents->transpose() * gradients[counter]);
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

  //  list_net();
  
  std::cout << "Expected loss sanity check...";
  
  Network copy5 ("./data_banknote_authentication.txt", 16, 0.05, 0.03, 0, 0.9);
  copy5.next_batch();
  copy5.feedforward();
  if (copy5.cost() <= 1) {
    std::cout << " \u001b[32mPassed!\n\u001b[37m";
    sanity_passed++;
  }
  else std::cout << " \u001b[31mFailed.\n\u001b[37m";

  //  list_net();
  
  std::cout << "Layer updates sanity check...";
  Network copy6 ("./data_banknote_authentication.txt", 16, 0.05, 0.03, 0, 0.9);
  Network copy7 ("./data_banknote_authentication.txt", 16, 0.05, 0.03, 0, 0.9);
  //copy2.list_net();
  //copy1.list_net();
  int passed;
  for (int i = 0; i < copy1.layers.size()-1; i++) {
    if (*copy1.layers[i].weights == *copy2.layers[i].weights) {
      //      std::cout << *copy2.layers[i].weights <<"uninitweight\n\n";
      //      std::cout << *copy1.layers[i].weights << " "<<i<<"weight\n\n";
      passed = -1;
    }
  }
  for (int i = 1; i < copy1.layers.size(); i++) {
    if (*copy1.layers[i].bias == *copy2.layers[i].bias) {
      //      std::cout << *copy2.layers[i].bias <<"uninitbias\n\n";
      //  std::cout << *copy1.layers[i].bias <<" " << i << "bias\n\n";
      passed = -1;
    }
  }
  if (passed == 1) {
    std::cout << " \u001b[32mPassed!\n\u001b[37m";
    sanity_passed++;
  }
  else std::cout << " \u001b[31mFailed.\n\u001b[37m";

  // std::cout << "Side effects sanity check...";
  
  // if (net == original) {
  //   std::cout << " \u001b[32mPassed!\n\u001b[37m";
  //   sanity_passed++;
  // }
  // else std::cout << " \u001b[31mFailed.\n\u001b[37m";

  std::cout << "\u001b[1m\nPassed " << sanity_passed << "/6" <<" sanity checks.\u001b[0m\n\n\n";
  
  //  net.list_net();
  
  // float epsilon = 0.0001;
  // Network copy = *this;
  // std::vector<Eigen::MatrixXf> approx_gradients;
  // for (int i = 0; i < copy.layers.size()-1; i++) {
  //   Eigen::MatrixXf current_approx = *copy.layers[i].weights;
  //   for (int j = 0; i < copy.layers[i].weights->rows(); i++) {
  //     for (int k = 0; i < copy.layers[i].weights->cols(); i++) {
  //       Network sim1 = copy;
  //       (*sim1.layers[i].contents)(j,k) += epsilon;
  //       sim1.feedforward();
  //       Network sim2 = copy;
  //       (*sim2.layers[i].contents)(j,k) -= epsilon;
  //       sim2.feedforward();
  //       current_approx(j,k) = (sim1.cost() - sim2.cost())/(2*epsilon);
  //     }
  //   }
  //   approx_gradients.push_back(current_approx);
  // }
  // for (Eigen::MatrixXf i : approx_gradients) {
  //   std::cout << i << "\n\n";
  // }
  // std::vector<Eigen::MatrixXf> gradients;
  // std::vector<Eigen::MatrixXf> deltas;
  // Eigen::MatrixXf error = ((*layers[length-1].contents) - (*labels));
  // gradients.push_back(error.cwiseProduct(*layers[length-1].dZ));
  // deltas.push_back((*layers[length-2].contents).transpose() * gradients[0]);
  // int counter = 1;
  // for (int i = length-2; i >= 1; i--) {
  //   gradients.push_back((gradients[counter-1] * layers[i].weights->transpose()).cwiseProduct(*layers[i].dZ));
  //   deltas.push_back(layers[i-1].contents->transpose() * gradients[counter]);
  //   counter++;
  // }
  //printf("Beginning train on %i instances for %i epochs...\n", instances, total_epochs);

}
