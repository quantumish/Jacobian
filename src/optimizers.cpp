//
//  optimizers.cpp
//  Jacobian
//
//  Created by David Freifeld
//

void Network::init_optimizer(char* name, ...)
{
  va_list args;
  va_start(args, name);
  if (strcmp(name, "momentum") == 0) {
    float beta = va_arg(args, double);
    va_end(args);
    update = [this, beta](std::vector<Eigen::MatrixXf> deltas, int i) {
      *layers[length-2-i].weights -= (beta * *layers[length-2-i].m) + (learning_rate * deltas[i]);
      *layers[length-2-i].m = (learning_rate * deltas[i]);
    };
  }
  else if (strcmp(name, "demon") == 0) {
    float beta_init = va_arg(args, double);
    float max_ep = va_arg(args, int);
    float beta = beta_init;
    int prev_epoch = -1;
    va_end(args);
    update = [this, max_ep, prev_epoch, beta_init, beta](std::vector<Eigen::MatrixXf> deltas, int i) mutable {
      if (epochs > prev_epoch) {
        beta = beta_init * (1-(epochs/max_ep)) / ((beta_init * (1-(epochs/max_ep))) + (1-beta_init));
        prev_epoch = epochs;
      }
      *layers[length-2-i].weights -= (beta * *layers[length-2-i].m) + (learning_rate * deltas[i]);
      *layers[length-2-i].m = (learning_rate * deltas[i]);
    };
  }
  else if (strcmp(name, "adam") == 0) {
    float beta1 = va_arg(args, double);
    float beta2 = va_arg(args, double);
    float epsilon = va_arg(args, double);
    va_end(args);
    // TODO: Add bias correction (requires figuring out measuring t)
    // TODO: cwiseProduct here is sketchy, look into me
    update = [this, beta1, beta2, epsilon](std::vector<Eigen::MatrixXf> deltas, int i) {
      *layers[length-2-i].m = (beta1 * *layers[length-2-i].m) + ((1-beta1)*deltas[i]);
      *layers[length-2-i].v = (beta2 * *layers[length-2-i].v) + (1-beta2)*(deltas[i].cwiseProduct(deltas[i]));
      *layers[length-2-i].weights -= learning_rate * ((layers[length-2-i].v->cwiseSqrt()).array()+epsilon).pow(-1).cwiseProduct(layers[length-2-i].m->array()).matrix();
    };
  }
  else if (strcmp(name, "adamax") == 0) {
    float beta1 = va_arg(args, double);
    float beta2 = va_arg(args, double);
    float epsilon = va_arg(args, double);
    va_end(args);
    // TODO: Add bias correction for m (requires figuring out measuring t)
    update = [this, beta1, beta2, epsilon](std::vector<Eigen::MatrixXf> deltas, int i) {
      *layers[length-2-i].m = (beta1 * *layers[length-2-i].m) + ((1-beta1)*deltas[i]);
      // FIXME: Use of .sum() here is incredibly questionable. Do this correctly.
      if ((beta2 * *layers[length-2-i].v).sum() > deltas[i].array().abs().sum()) *layers[length-2-i].v = (beta2 * *layers[length-2-i].v);
      else *layers[length-2-i].v = deltas[i].array().abs().matrix();
      *layers[length-2-i].weights -= learning_rate * (layers[length-2-i].v->array().pow(-1).cwiseProduct(layers[length-2-i].m->array())).matrix();
    };
  }
}
