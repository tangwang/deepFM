/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "solver/parammeter_container.h"

class AdamParamUnit {
 public:
  FMParamUnit fm_param;
  FMParamUnit avg_grad; // 1st momentum (the mean) of the gradient
  FMParamUnit avg_squared; // 2nd raw momentum (the uncentered variance) of the gradient
  real_t beta1power_t;
  real_t beta2power_t;

  AdamParamUnit() {
    fm_param.w = 0.0;
    avg_grad.w = 0.0;
    avg_squared.w = 0.0;
    beta1power_t = 1.0;
    beta2power_t = 1.0;
    for (int f = 0; f < DIM; ++f) {
      fm_param.V[f] = utils::gaussian(0.0, train_opt.init_stdev);
      avg_grad.V[f] = 0.0;
      avg_squared.V[f] = 0.0;
    }
  }

  void update(const FMParamUnit &grad) {

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

    // update w
    real_t &w = this->fm_param.w;
    real_t &wm = this->avg_grad.w;
    real_t &wv = this->avg_squared.w;

    wm = beta1 * wm + (1 - beta1) * grad.w;
    real_t avg_squared = beta2 * wv + (1 - beta2) * grad.w * grad.w;
    wv = amsgrad ? std::max(wv, avg_squared) : avg_squared;

    DEBUG_OUT << "adam_solver: grad:" << grad
              << " corection_lr:" << corection_lr << " count "
              << param_node.count << " wm:" << wm << " wv:" << wv << " update:"
              << corection_lr *
                     (wm / (std::sqrt(wv) + eps) + weight_decay_w * w)
              << endl
              << "fm_param:" << this->fm_param.w << ","
              << this->fm_param.V[0] << ","
              << this->fm_param.V[1] << endl
              << "avg_grad:" << this->avg_grad.w << ","
              << this->avg_grad.V[0] << ","
              << this->avg_grad.V[1] << endl
              << "avg_squared:" << this->avg_squared.w << ","
              << this->avg_squared.V[0] << ","
              << this->avg_squared.V[1] << endl
              << "fm_param.V_0_1 " << this->fm_param.V[0] << ","
              << this->fm_param.V[1] << endl;

    // adamW:
    // Just adding the square of the weights to the loss function is *not*
    // the correct way of using L2 regularization/weight decay with Adam,
    // since that will interact with the m and v parameters in strange ways.
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

  static constexpr real_t eps = 1e-8;
  static constexpr real_t tolerance = 1e-5;
  static constexpr bool resetPolicy = true;
  static constexpr bool exactObjective = false;
  static constexpr bool bias_correction = true;

  static constexpr bool amsgrad = false; // 保留历史最大的v_t，记为v_{max}，每次计算都是用最大的v_{max}，否则是用当前v_t
  // amsgrad需要多一个保存一份历史最大值平方梯度v_{max}。 暂未实现
  // avg_grads = beta1 * avg_grads + (1-beta1) * w.grad
  // avg_squared = beta2 * (avg_squared) + (1-beta2) * (w.grad * w.grad)
  // max_squared = max(avg_squared, max_squared)
  // w = w - lr * avg_grads / sqrt(max_squared)

};

