/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "solver/parammeter_container.h"

class SgdmParamUnit {
public:
  FMParamUnit fm_param;
  FMParamUnit momentum;

  SgdmParamUnit() {
    fm_param.w = 0.0;
    momentum.w = 0.0;
    for (int f = 0; f < DIM; ++f) {
      fm_param.V[f] = utils::gaussian(0.0, train_opt.init_stdev);
      momentum.V[f] = 0.0;
    }
  }

  void update(const FMParamUnit &grad) {
        const real_tlr = train_opt.sgdm.lr;
        const real_tbeta1 = train_opt.sgdm.beta1;
        const real_tl1_reg_w = train_opt.sgdm.l1_reg_w;
        const real_tl1_reg_V = train_opt.sgdm.l1_reg_V;
        const real_tl2_reg_w = train_opt.sgdm.l2_reg_w;
        const real_tl2_reg_V = train_opt.sgdm.l2_reg_V;

      real_t & w = this->fm_param.w;
      real_t & wm = this->momentum.w;

      wm = beta1 * wm + (1-beta1) * grad.w;
      w -= lr * (wm  + w * l2_reg_w);

      for (int f = 0; f < DIM; ++f) {
        real_t &vf = this->fm_param.V[f];
        real_t & vmf = this->momentum.V[f];
        real_t vgf = grad.V[f];

        vmf = beta1 * vmf + (1-beta1) * vgf;
        vf -= lr * (vmf + vf * l2_reg_V);
      }


  }

};

