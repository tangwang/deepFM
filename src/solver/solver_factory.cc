/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#include "feature/feat_manager.h"
#include "solver/solver_factory.h"

shared_ptr<ParamContainerInterface> creatParamContainer(feat_id_t  feat_num, feat_id_t mutex_nums) {
  if (0 == strcasecmp(train_opt.solver.c_str(), "ftrl")) {
    return std::make_shared<ParamContainer<FtrlParamUnit>>(feat_num, mutex_nums);
  } else if (0 == strcasecmp(train_opt.solver.c_str(), "sgd") || 0 == strcasecmp(train_opt.solver.c_str(), "sgdm")) {
    return std::make_shared<ParamContainer<SgdmParamUnit>>(feat_num, mutex_nums);
  } else if (0 == strcasecmp(train_opt.solver.c_str(), "adagrad")) {
    return std::make_shared<ParamContainer<AdagradParamUnit>>(feat_num, mutex_nums);
  } else if (0 == strcasecmp(train_opt.solver.c_str(), "rmsprop")) {
    return std::make_shared<ParamContainer<RmspropParamUnit>>(feat_num, mutex_nums);
  } else if (0 == strcasecmp(train_opt.solver.c_str(), "adam")) {
    return std::make_shared<ParamContainer<AdamParamUnit>>(feat_num, mutex_nums);
  } else if (0 == strcasecmp(train_opt.solver.c_str(), "pred")) {
    return std::make_shared<ParamContainer<FMParamUnit>>(feat_num, mutex_nums);
  } else {
    cerr << "unknown solver, use adam by default." << endl;
    return std::make_shared<ParamContainer<AdamParamUnit>>(feat_num, mutex_nums);
  }
}
