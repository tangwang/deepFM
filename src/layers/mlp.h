/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include <array>

#include "utils/base.h"
#include "utils/utils.h"

using std::array;

struct AdamContext {
  real_t avg_grad;
  real_t avg_squared;
};

struct PropagationContext {
  real_t forward_value;
  real_t backward_value;
};

template <int dim1, int dim2, typename NodeType>
class MLPParam<dim1, dim2> {
 public:
  vector<array<NodeType, dim1>> hidden1;
  array<array<NodeType, dim2>, dim1> hidden2;
  array<NodeType, dim2> output;
};

// 每个线程一个，各自积累梯度。
// 积累了一个batch的梯度后，提交到全局参数做一次update
template <int dim1, int dim2>
class MLPLocal {
 public:
  MLPLocal(size_t input_size) : hidden1(input_size) {}
  ~MLPLocal() {}
  
  void forward(const vector<real_t>& input,
               array<real_t, dim2>& out_data) {
    for (int row = 0; row < dim_in; row++) {
      out_data[row] = 0.0;
      for (int col = 0; col < dim_in; col++) {
        out_data[row] += in_data[col] * mat[row][col];
      }
    }
  }

  void push(MLPServer & param_server) {
    // 先做batch_reduce
    // 提交到参数param server
    param_server.push(propagation_context);

    // 更新本地参数
    param_server.pull(param);
  }

 private:
  MLPParam<dim1, dim2, real_t> param;
  MLPParam<dim1, dim2, PropagationContext> propagation_context;
};

template <int dim1, int dim2>
class MLPServer {
 public:
  MLPServer(size_t input_size) : hidden1(input_size) {}
  ~MLPServer() {}

  void initParam() {}

  void sync(MLPParam<dim1, dim2, PropagationContext> & grad, MLPParam<dim1, dim2, real_t> & to) {
    push(grad);
    pull(to);
  }

  void push(MLPParam<dim1, dim2, PropagationContext> & grad) {

    const real_t beta1 = train_opt.adam.beta1;
    const real_t beta2 = train_opt.adam.beta2;
    const real_t weight_decay_w = train_opt.adam.weight_decay_w;
    const real_t weight_decay_V = train_opt.adam.weight_decay_V;

    // correction learning rate
    this->beta1power_t *= beta1;
    this->beta2power_t *= beta2;
    real_t bias_correction1 = (1 - this->beta1power_t);
    real_t bias_correction2 = (1 - this->beta2power_t);
    real_t corection_lr = bias_correction
                              ? (train_opt.adam.lr *
                                 std::sqrt(bias_correction2) / bias_correction1)
                              : lr;

    // i_hidden1
    size_t input_size = param.hidden1.size();
    for (size_t i = 0; i < input_size; i++) {
      array<real_t, dim1> &i_hidden1 = param.hidden1[i];
      for (size_t j = 0; j < dim1; j++) {
        real_t & hidden1_i_j = i_hidden1[j];

      }
    }

    // i_hidden2


    // update w
    real_t &w = this->fm_param.w;
    real_t &wm = this->avg_grad.w;
    real_t &wv = this->avg_squared.w;

    wm = beta1 * wm + (1 - beta1) * grad.w;
    real_t avg_squared = beta2 * wv + (1 - beta2) * grad.w * grad.w;
    wv = amsgrad ? std::max(wv, avg_squared) : avg_squared;

    w -= corection_lr * (wm / (std::sqrt(wv) + eps) + weight_decay_w * w);

    // update V
    for (int f = 0; f < DIM; ++f) {
      real_t &vf = this->fm_param.V[f];
      real_t &vmf = this->avg_grad.V[f];
      real_t &vvf = this->avg_squared.V[f];

      real_t vgf = grad.V[f];

      vmf = beta1 * vmf + (1 - beta1) * vgf;
      vvf = beta2 * vvf + (1 - beta2) * vgf * vgf;
      vf -= corection_lr * (vmf / (std::sqrt(vvf) + eps) + weight_decay_V * vf);
    }
  }


  }

  void pull(MLPParam<dim1, dim2, real_t> & to) {
    to = param;
  }

 private:
  MLPParam<dim1, dim2, real_t> param;
  MLPParam<dim1, dim2, AdamContext> adam_context;
  real_t beta1power_t;  // 1st momentum (the mean) of the gradient
  real_t beta2power_t;  // 2nd raw momentum (the uncentered variance) of the

  static constexpr real_t eps = 1e-8;
  static constexpr real_t tolerance = 1e-5;
  static constexpr bool resetPolicy = true;
  static constexpr bool exactObjective = false;
  static constexpr bool bias_correction = true;

  static constexpr bool amsgrad = false; // 保留历史最大的v_t，记为v_{max}，每次计算都是用最大的v_{max}，否则是用当前v_t

};
