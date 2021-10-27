/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include <array>

#include "utils/base.h"
#include "utils/utils.h"

using std::array;

// 每个线程一个，各自积累梯度。
// 积累了一个batch的梯度后，提交到全局参数做一次update
template <int dim1, int dim2>
class LocalMLP {
 public:
  LocalMLP(size_t input_size) : hidden1(input_size) {}
  ~LocalMLP() {}

  void initParam() {}
  void forward(const vector<real_t>& input,
               array<real_t, dim2>& out_data) const {
    for (int row = 0; row < dim_in; row++) {
      out_data[row] = 0.0;
      for (int col = 0; col < dim_in; col++) {
        out_data[row] += in_data[col] * mat[row][col];
      }
    }
  }

  struct Network {
    vector<array<real_t, dim1>> hidden1;
    real_t hidden2[dim1][dim2];
  };

 private:
  Network param;
  Network x;

  // adam
  Network avg_grad;     // 1st momentum (the mean) of the gradient
  Network avg_squared;  // 2nd raw momentum (the uncentered variance) of the
                        // gradient
  real_t beta1power_t;
  real_t beta2power_t;
};


template <int dim1, int dim2>
class GlobalMLP {
 public:
  GlobalMLP(size_t input_size) : hidden1(input_size) {}
  ~GlobalMLP() {}

  void initParam() {}

  void update(Network & grad) {}
  
  void fetch(Network & p) {}

  struct Network {
    vector<array<real_t, dim1>> hidden1;
    real_t hidden2[dim1][dim2];
  };

 private:
  Network param;

  // adam
  Network avg_grad;     // 1st momentum (the mean) of the gradient
  Network avg_squared;  // 2nd raw momentum (the uncentered variance) of the
                        // gradient
  real_t beta1power_t;
  real_t beta2power_t;
};
