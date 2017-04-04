#ifndef CAFFE_RPN_ANNOTATOR_OHEM_LAYER_HPP_
#define CAFFE_RPN_ANNOTATOR_OHEM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

 /**
 * @brief RpnAnnotatorOHEMLayer: Annotate rpn labels for Online Hard Example Mining (OHEM) training
 */
  template <typename Dtype>
  class RpnAnnotatorOHEMLayer :public Layer<Dtype>{
   public:
    explicit RpnAnnotatorOHEMLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "RpnAnnotatorOHEM"; }

    virtual inline int ExactNumBottomBlobs() const { return 3; }
    virtual inline int ExactNumTopBlobs() const { return 2; }

   protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    int num_;
    int height_;
    int width_;
    int spatial_dim_;

    int rpn_per_img_;
    int ignore_label_;
  };

}  // namespace caffe

#endif  // CAFFE_RPN_ANNOTATOR_OHEM_LAYER_HPP_
