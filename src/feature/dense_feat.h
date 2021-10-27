/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "feature/common_feat.h"

class DenseFeatConfig : public CommonFeatConfig {
 public:
  real_t min;
  real_t max;
  real_t add;
  real_t multiply;
  real_t pow;
  real_t log_base;
  real_t log_divisor;
  
  real_t default_value;

  // 配置的等频分桶桶宽
  vector<int> sparse_by_wide_bins_numbs;
  // 配置的分桶
  vector<vector<real_t>> sparse_by_splits;

  // 以下3个vector，长度一致，按位置一一对应
  vector<real_t> all_splits;                        // 分隔值
  vector<vector<feat_id_t>>  feat_ids_of_each_buckets;  // 分隔值对应的onehot ID列表
  vector<vector<FMParamUnit *>>
       feat_params_of_each_buckets;  // 分隔值对应的onehot ID列表 所对应的参数位置

  const vector<feat_id_t> &get_feat_ids(real_t x) const {
    if (x == default_value) {
      return  feat_ids_of_each_buckets[feat_ids_of_each_buckets.size() - 1];
    }
    x += add;
    if (multiply != 0.0) {
      x *= multiply;
    }
    if (pow != 0.0) {
      x = std::pow(x, pow);
    }
    if (log_divisor != 0.0) {
      x = std::log(x) / log_divisor;
    }

    int bucket_id = lower_bound(all_splits.begin(), all_splits.end(), x) -
                    all_splits.begin();
    // TODO check
    if (bucket_id == (int)all_splits.size()) --bucket_id;
    /* gdb debug
     p  feat_params_of_each_buckets[bucket_id]
     拿到param地址后：
     p (*(FMParamUnit *)0x6c8138)
     p (*(FMParamUnit *)0x6c8138).buff@24
     */
    return  feat_ids_of_each_buckets[bucket_id];
  }

  int getFeaBucketId(real_t x) const {
    assert(x != default_value);
    int bucket_id = lower_bound(all_splits.begin(), all_splits.end(), x) -
                    all_splits.begin();

    // TODO是否要加这个
    if (bucket_id == (int)all_splits.size()) --bucket_id;

    return bucket_id;
  }

  bool initParams(unordered_map<string, shared_ptr<ParamContainerInterface>> & shared_param_container_map);

  friend ostream & operator << (ostream &out, const DenseFeatConfig & cfg) {
    out << " DenseFeatConfig name <" << cfg.name << ">" << endl;
    out << " sparse_by_splits: " << endl << cfg.sparse_by_splits << endl;
    out << " sparse_by_wide_bins_numbs: " << endl << cfg.sparse_by_wide_bins_numbs << endl;
    out << " all_splits: " << endl << cfg.all_splits << endl;
    out << "  feat_ids_of_each_buckets: " << endl << cfg.feat_ids_of_each_buckets << endl;
    out << ">\n min <" << cfg.min << "> max <" << cfg.max << ">" << endl;
    out << " default_value <" << cfg.default_value << ">" << endl;
    return out;
  }

  DenseFeatConfig();
  ~DenseFeatConfig();
};

void to_json(json &j, const DenseFeatConfig &p);
void from_json(const json &j, DenseFeatConfig &p);

class DenseFeatContext : public CommonFeatContext {
 public:
  real_t orig_x;
  const vector<FMParamUnit *> *feat_params;

  const DenseFeatConfig &cfg_;

  int feedSample(const char *feat_str, size_t feat_str_len, FmLayerNode & fm_node);

  virtual void backWard(FMParamUnit & backward_grad);
  int sample_idx;
  // pair.first = bucket_id, pair=second = grad
  vector<pair<int, FMParamUnit>> batch_backward_nodes;

  bool valid() const {
    // TODO 暂时只支持离散特征
    return orig_x != cfg_.default_value && !cfg_.all_splits.empty();
  }

  DenseFeatContext(const DenseFeatConfig &cfg);
  ~DenseFeatContext();
};
