/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "train/train_opt.h"
#include "utils/base.h"
#include "utils/utils.h"
#include "synchronize/mutex_adapter.h"
#include "utils/stopwatch.h"


// 通用参数的头部结构
struct FMParamUnit {
  FMParamUnit() { }
  real_t w;
  real_t V[DIM];
  void operator+=(const struct FMParamUnit &other) {
    w += other.w;
    for (int i = 0; i < DIM; i++) {
      V[i] += other.V[i];
    }
  }
  void operator/=(real_t div) {
    w /= div;
    for (int i = 0; i < DIM; i++) {
      V[i] /= div;
    }
  }
  void operator*=(real_t value) {
    w *= value;
    for (int i = 0; i < DIM; i++) {
      V[i] *= value;
    }
  }
  void clear() {
    w = 0.0;
    for (int i = 0; i < DIM; i++) {
      V[i] = 0.0;
    }
  }

  friend std::ostream &operator<<(std::ostream &os, const FMParamUnit &p) {
    return os << "(w: " << p.w << " V[0]: " << p.V[0] << ")";
  }
};

// node of fm_socre layer ( backward_param (sum/avg to)-> forward_param -> fm_score -> sigmoid+crossEntopy <- label )
struct FmLayerNode {
  FMParamUnit forward;
  CommonFeatContext * feat;
};

class ParamContainerInterface {
 public:
  ParamContainerInterface(feat_id_t total_feat_num, feat_id_t _mutex_nums, size_t _param_size_of_one_fea)
      : mutexes(_mutex_nums),
        param_size_of_one_fea(_param_size_of_one_fea),
         feat_num(total_feat_num),
        mutex_nums(_mutex_nums)
  {
        param_base_addr = (unsigned char *)malloc((feat_num + 1) * param_size_of_one_fea);
  }

  virtual ~ParamContainerInterface() {
    free(param_base_addr);
  }

  bool isBadID(feat_id_t id) const { return id >  feat_num || id < 0; }
  feat_id_t getUNKnownID() const { return  feat_num; }

  FMParamUnit *get() {
    return (FMParamUnit *)param_base_addr;
  }

  FMParamUnit *get(feat_id_t id) {
    if (unlikely(isBadID(id))) {
      id = getUNKnownID();
    }
    return (FMParamUnit *)(param_base_addr + param_size_of_one_fea * id);
  }

  size_t getParamUnitSize() const {
    return param_size_of_one_fea;
  }

  int load(string path, string model_fmt) {
    feat_id_t total_feat_num =  feat_num + 1;
    int ret = 0;
    if (model_fmt == "bin") {
      ifstream ifs;
      ifs.open(path, std::ifstream::binary);
      feat_id_t i = 0;
      for (; i < total_feat_num; i++) {
        // load params befor solver threads created, so donnot need to lock. if solver workers is created, must lock for each params
        FMParamUnit *p = get(i);
        ifs.read((char *)p, sizeof(FMParamUnit));
        if (!ifs) {
          std::cerr << "load model faild, size not match: " << path << " feat_id: " << i <<  std::endl;
          return -1;
        }
      }
      if (i != total_feat_num) {
          std::cerr << "load model faild, size not match: " << path << " feat_id: " << i <<  std::endl;
          return -2;
      }
    } else {
      ifstream ifs;
      ifs.open(path);
      string line;
      feat_id_t i = 0;
      for (; i < total_feat_num; i++) {
        FMParamUnit *p = get(i);
        if (!std::getline(ifs, line)) {
          std::cerr << "load model faild, size not match: " << path << " feat_id: " << i <<  std::endl;
          return -1;
        }
        real_t * read_end = utils::split_string(line, ' ', (real_t *)p);
        if (read_end - (real_t *)p != 1 + DIM) {
          std::cerr << "load model faild, param size not match in line: " << i << " path: " << path << std::endl;
          return -2;
        }
      }
      if (i != total_feat_num) {
          std::cerr << "load model faild, size not match: " << path << " feat_id: " << i <<  std::endl;
          return -2;
      }
    }
    std::cout << "load model ok: " << path <<  " total_feat_num: " << total_feat_num <<  std::endl;
    return ret;
  }

  int dump(string path, string model_fmt) {
    int ret = 0;
    feat_id_t total_feat_num =  feat_num + 1;
    if (model_fmt == "bin") {
      ofstream ofs(path, std::ios::out | std::ios::binary);
      if (!ofs) {
        return -1;
      }

      utils::Stopwatch stopwatch;
      stopwatch.start();

      for (feat_id_t i = 0; i < total_feat_num; i++) {
        const FMParamUnit *p = get(i);
        ParamMutex_t *param_mutex = GetMutexByFeatID(i);
        param_mutex->readLock();
        ofs.write((const char *)p, sizeof(FMParamUnit));
        param_mutex->unlock();
      }
      ofs.close();

      int tm = stopwatch.get_elapsed_by_ms();
      cout << " dump model : feat_num " << total_feat_num <<  " feat_size " << sizeof(FMParamUnit) << " cost " << tm
           << " ms " << endl;

    } else {
      ofstream ofs(path);
      if (!ofs) {
        return -1;
      }
      for (feat_id_t i = 0; i < total_feat_num; i++) {
        const FMParamUnit *p = get(i);
        ParamMutex_t *param_mutex = GetMutexByFeatID(i);
        param_mutex->readLock();
        ofs << p->w;
        for (int i = 0; i < DIM; i++) {
          ofs << " " << p->V[i];
        }
        param_mutex->unlock();
        ofs << std::endl;
      }
      ofs.close();
    }
    return ret;
  }

  unsigned char * param_base_addr;
  vector<ParamMutex_t> mutexes;
  const size_t param_size_of_one_fea;
  const feat_id_t  feat_num;
  const int mutex_nums;

  // mutexes
  ParamMutex_t* GetMutexByFeatID(feat_id_t id) {
    return &mutexes[id % mutex_nums];
  }

 private:
  // disable copy
  ParamContainerInterface(const ParamContainerInterface &ohter);
  ParamContainerInterface &operator=(const ParamContainerInterface &that);
  ParamContainerInterface(ParamContainerInterface &ohter);
  ParamContainerInterface &operator=(ParamContainerInterface &that);
};

template <class ParamUnit>
class ParamContainer : public ParamContainerInterface {
 public:
  ParamContainer(feat_id_t total_feat_num, feat_id_t mutex_nums)
      : ParamContainerInterface(total_feat_num, mutex_nums, sizeof(ParamUnit)) {
    for (feat_id_t i = 0; i <  feat_num + 1; i++) {
      ParamUnit *p = (ParamUnit *)get(i);
      new (p) ParamUnit();
    }
  }
  virtual ~ParamContainer() {}
};
