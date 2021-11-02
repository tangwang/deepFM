/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#include "solver/base_solver.h"
#include "feature/feat_manager.h"

real_t Sample::forward() {
  logit = 0.0;
  
  for (int f = 0; f < DIM; ++f) {
    sum[f] = sum_sqr[f] = 0.0;
  }

  for (size_t i = 0; i < fm_layer_nodes_size; i++) {
    const auto & node = fm_layer_nodes[i];
    real_t x = 1.0;
    logit += node.forward.w * x;
    for (int f = 0; f < DIM; ++f) {
      real_t d = node.forward.V[f] * x;
      sum[f] += d;
      sum_sqr[f] += d * d;
    }
  }
  real_t sum_factors_score = 0.0;
  for (int f = 0; f < DIM; ++f) {
    sum_factors_score += (sum[f] * sum[f] - sum_sqr[f]);
  }

  logit += (0.5 * sum_factors_score);

  return logit;
}

void Sample::backward() {
  // 计算整体的梯度:
  // crossEntropyLoss = -log( sigmoid(label.i * fm_score(x) ) ) ， label.i = {-1, 1}
  // partitial(loss) / partitial(fm_score(x)) = -label.i * sigmoid( - fm_score(x) * label.i )，y*score
  real_t exp_y_logit = std::exp(logit * label.i);
  grad = -label.i / (1 + exp_y_logit);
  loss = - std::log(1 - 1/(1+std::max(exp_y_logit, 1e-10)));
  
  FMParamUnit backward_grad;
  for (size_t i = 0; i < fm_layer_nodes_size; i++) {
    auto & node = fm_layer_nodes[i];
    //  partitial(fm_score(x)) / partitial(fm_node)
    real_t xi = 1.0;
    real_t grad_i = grad * xi;
    backward_grad.w = grad_i;
    for (int f = 0; f < DIM; ++f) {
      real_t &vf = node.forward.V[f];
      real_t vgf = grad_i * (sum[f] - vf * xi);
      backward_grad.V[f] = vgf;
    }
    // 计算每个fmParamUnit的梯度： partitial(fm_score(x)) / partitial(\theta),  theata = {w_i, V_i1, Vi2, ... Vif} for i in {0, 1, ... N }
    node.feat->backWard(backward_grad);
  }
}

BaseSolver::BaseSolver(const FeatManager &feat_manager)
    : feat_manager_(feat_manager), batch_size(train_opt.batch_size) {
  for (const auto &v : feat_manager_.dense_feat_cfgs) {
    dense_feats.emplace_back(v);
  }
  for (const auto &v : feat_manager_.sparse_feat_cfgs) {
    sparse_feats.emplace_back(v);
  }
  for (const auto &v : feat_manager_.varlen_feat_cfgs) {
    varlen_feats.emplace_back(v);
  }

  sample.fm_layer_nodes.resize(dense_feats.size() + sparse_feats.size() + varlen_feats.size());

  for (auto &v : dense_feats) {
    feat_map[v.feat_cfg->name] = &v;
  }
  for (auto &v : sparse_feats) {
    feat_map[v.feat_cfg->name] = &v;
  }
  for (auto &v : varlen_feats) {
    feat_map[v.feat_cfg->name] = &v;
  }
  if (train_opt.data_formart == TrainOption::DataFormart_CSV) {
    utils::split_string(train_opt.csv_columns, ',', csv_columns);

    VERBOSE_OUT(1) << "csv_columns size: " << csv_columns.size() << endl;
    VERBOSE_OUT(1) << "csv_columns : " << csv_columns << endl;

    assert(!csv_columns.empty());
    
    for (size_t i = 0; i < csv_columns.size(); i++) {
      const string &column = csv_columns[i];
      auto got = feat_map.find(column);
      if (got != feat_map.end()) {
        feat_entries.push_back(make_pair(i, got->second));
     }
    }
    if (feat_entries.size() != feat_map.size()) {
        cerr << "feature names not match csv_columns, check your feature_config or your csv header line, exit." << endl;
        std::exit(1);
     }
  }
}

real_t BaseSolver::feedLine_libSVM(const string & aline) {
  // label统一为1， -1的形式
  // label.i = atoi(line) > 0 ? 1 : -1;
  char line[aline.size() + 1];
  memcpy(line, aline.c_str(), aline.size() + 1);
  char * pos = line;
  char * feat_beg = NULL;
  char * feat_end = strchr(pos, train_opt.feat_seperator);
  char * feat_kv_pos = NULL;
  if (unlikely(*pos == '\0' || feat_end == NULL)) {
    return -1;
  }

  // parse label
  sample.label.i = int(pos) > 0 ? 1 : -1;

  // parse featrues
  size_t fm_node_idx = 0;
  do {
    feat_beg = feat_end + 1;
    feat_end = strchr(feat_beg, train_opt.feat_seperator);
    feat_kv_pos = strchr(feat_beg, train_opt.feat_kv_seperator);
    int feat_len = (feat_end == NULL) ? (int)strlen(feat_kv_pos + 1) : feat_end - (feat_kv_pos + 1);
    if (likely(feat_kv_pos != NULL && feat_len > 0)) {
      *feat_kv_pos = '\0';
      if (feat_end) *feat_end = '\0';
      auto got = feat_map.find(feat_beg);
      if (got != feat_map.end()) {
        DEBUG_OUT << " feed : "<< fm_node_idx << " " << feat_beg << " " << feat_kv_pos + 1 << endl;
        got->second->feedSample(feat_kv_pos + 1, feat_len, sample.fm_layer_nodes[fm_node_idx++]);
      }
    }
  } while (feat_end != NULL);
  sample.fm_layer_nodes_size = fm_node_idx;

  return sample.forward();
}

real_t BaseSolver::feedLine_CSV(const string & aline) {
  // label统一为1， -1的形式
  // label.i = atoi(line) > 0 ? 1 : -1;
  line_split_buff.clear();
  utils::split_string(aline, train_opt.feat_seperator, line_split_buff);

  if (unlikely(line_split_buff.size() < csv_columns.size() || line_split_buff[0].empty())) {
    return -1;
  }
  
  // parse label
  sample.label.i = int(pos) > 0 ? 1 : -1;

  // parse featrues
  size_t fm_node_idx = 0;
  for (const auto &feat_entrie : feat_entries) {
    const string & feat_str = line_split_buff[feat_entrie.first];
        feat_entrie.second->feedSample(feat_str.c_str(), feat_str.size(), sample.fm_layer_nodes[fm_node_idx++]);
  }
  sample.fm_layer_nodes_size = fm_node_idx;

  return sample.forward();
}

void BaseSolver::train(const string & line, int &y, real_t &logit, real_t & loss, real_t & grad) {
  // feedLine and forward (calc score , loss)
  if (train_opt.data_formart == TrainOption::DataFormart_CSV) {
    feedLine_CSV(line);
  } else {
    feedLine_libSVM(line);
  }
  // backward ( calc the grad layer by layer to each param )
  sample.backward();
  
  y = sample.label.i;
  logit = sample.logit;
  loss = sample.loss;
  grad = sample.grad;

  DEBUG_OUT << "BaseSolver::train "
            << " y " << y << " logit " << logit << endl;
}

void BaseSolver::test(const string & line, int &y, real_t &logit) {
  if (train_opt.data_formart == TrainOption::DataFormart_CSV) {
    feedLine_CSV(line);
  } else {
    feedLine_libSVM(line);
  }
  y = sample.label.i;
  logit = sample.logit;
}

