/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "solver/parammeter_container.h"

class AdagradParamUnit {
 public:
  FMParamUnit fm_param;
  FMParamUnit squared_sum; // 2nd raw momentum (the uncentered variance) of the gradient

  AdagradParamUnit() {
    fm_param.w = 0.0;
    // squared_sum原始论文是初始化为0
    // squared_sum初始化为1e-7相比于初始化为0（原始论文的实现）有提升。初始化为0.1或者1对其他维度的超参（lr, batch_size, l2norm）更为鲁棒，更容易收敛，但是最高AUC不如squared_sum初始化为0的情况。
    squared_sum.w = 1e-7;
    for (int f = 0; f < DIM; ++f) {
      fm_param.V[f] = utils::gaussian(0.0, train_opt.init_stdev);
      squared_sum.V[f] = 1e-7;
    }
  }

  void update(const FMParamUnit &grad) {
        const real_tlr = train_opt.adagrad.lr;
        const real_tl2_norm_w = train_opt.adagrad.l2_norm_w;
        const real_tl2_norm_V = train_opt.adagrad.l2_norm_V;
      // update w
      real_t &w = this->fm_param.w;
      real_t &wv = this->squared_sum.w;

      wv += grad.w * grad.w;

      DEBUG_OUT << "adagrad_solver: grad:" << grad << " decayed_lr" << lr / (std::sqrt(wv) + eps)
                << " count " << param_node.count
                << " wv:" << wv << " update:" 
                << lr * (grad.w + l2_norm_w * w) / (std::sqrt(wv) + eps)
                << endl;

      w -= lr * (grad.w + l2_norm_w * w)  / (std::sqrt(wv) + eps);

      // update V
      for (int f = 0; f < DIM; ++f) {
        real_t &vf = this->fm_param.V[f];
        real_t &vvf = this->squared_sum.V[f];
        real_t vgf = grad.V[f];

        vvf += vgf * vgf;
        vf -= lr * (vgf + l2_norm_V * vf)  / (std::sqrt(vvf) + eps);
      }

  }

  static constexpr real_t eps = 1e-7;
  static constexpr bool amsgrad = false; // 保留历史最大的v_t，记为v_{max}，每次计算都是用最大的v_{max}，否则是用当前v_t
  // amsgrad需要多一个保存一份历史最大值平方梯度v_{max}。 暂未实现
  // avg_grads = beta1 * avg_grads + (1-beta1) * w.grad
  // squared_sum = beta2 * (squared_sum) + (1-beta2) * (w.grad * w.grad)
  // max_squared = max(squared_sum, max_squared)
  // w = w - lr * avg_grads / sqrt(max_squared)

};
