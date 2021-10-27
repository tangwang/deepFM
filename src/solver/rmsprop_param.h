/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "solver/parammeter_container.h"

class RmspropParamUnit {
 public:
  FMParamUnit fm_param;
  FMParamUnit avg_squared; // 2nd raw momentum (the uncentered variance) of the gradient

  RmspropParamUnit() {
    fm_param.w = 0.0;
    avg_squared.w = 1e-7;
    for (int f = 0; f < DIM; ++f) {
      fm_param.V[f] = utils::gaussian(0.0, train_opt.init_stdev);
      avg_squared.V[f] = 1e-7;
    }
  }

  void update(const FMParamUnit &grad) {
        const real_tlr = train_opt.rmsprop.lr;
        const real_tl2_norm_w = train_opt.rmsprop.l2_norm_w;
        const real_tl2_norm_V = train_opt.rmsprop.l2_norm_V;
        const real_tbeta2 = train_opt.rmsprop.beta2;
      // update w
      real_t &w = this->fm_param.w;
      real_t &wv = this->avg_squared.w;

      wv = beta2 * wv + (1 - beta2) * grad.w * grad.w;

      DEBUG_OUT << "rmsprop_solver: grad:" << grad << " decayed_lr" << lr / (std::sqrt(wv) + eps)
                << " count " << param_node.count
                << " wv:" << wv << " update:" 
                << lr * (grad.w + l2_norm_w * w) / (std::sqrt(wv) + eps)
                << endl;

      w -= lr * (grad.w + l2_norm_w * w)  / (std::sqrt(wv) + eps);

      // update V
      for (int f = 0; f < DIM; ++f) {
        real_t &vf = this->fm_param.V[f];
        real_t &vvf = this->avg_squared.V[f];
        real_t vgf = grad.V[f];

        vvf = beta2 * vvf + (1 - beta2) * vgf * vgf;
        vf -= lr * (vgf + l2_norm_V * vf)  / (std::sqrt(vvf) + eps);
      }
  }

  static constexpr real_t eps = 1e-7;
  static constexpr bool amsgrad = false; // 保留历史最大的v_t，记为v_{max}，每次计算都是用最大的v_{max}，否则是用当前v_t
  // amsgrad需要多一个保存一份历史最大值平方梯度v_{max}。 暂未实现
  // avg_grads = beta1 * avg_grads + (1-beta1) * w.grad
  // avg_squared = beta2 * (avg_squared) + (1-beta2) * (w.grad * w.grad)
  // max_squared = max(avg_squared, max_squared)
  // w = w - lr * avg_grads / sqrt(max_squared)

};
