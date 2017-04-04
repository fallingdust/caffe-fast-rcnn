#include <cfloat>

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/rpn_annotator_ohem_layer.hpp"
#include "caffe/proto/caffe.pb.h"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

  template <typename Dtype>
  void RpnAnnotatorOHEMLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    RpnAnnotatorOHEMParameter rpn_anno_param =
      this->layer_param_.rpn_annotator_ohem_param();
    rpn_per_img_ = rpn_anno_param.rpn_per_img();
    CHECK_GT(rpn_per_img_, 0);
    ignore_label_ = rpn_anno_param.ignore_label();
  }

  template <typename Dtype>
  void RpnAnnotatorOHEMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    num_ = bottom[0]->num();
    CHECK_EQ(bottom[0]->channels(), 1);
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    spatial_dim_ = height_*width_;

    CHECK_EQ(bottom[1]->num(), num_);
    CHECK_EQ(bottom[1]->channels(), 1);
    CHECK_EQ(bottom[1]->height(), height_);
    CHECK_EQ(bottom[1]->width(), width_);

    CHECK_EQ(bottom[2]->num(), num_);
    CHECK_EQ(bottom[2]->channels(), 4);
    CHECK_EQ(bottom[2]->height(), height_);
    CHECK_EQ(bottom[2]->width(), width_);

    // Labels for scoring
    top[0]->Reshape(num_, 1, height_, width_);
    // Loss weights for bbox regression
    top[1]->Reshape(num_, 4, height_, width_);
  }

  template <typename Dtype>
  void RpnAnnotatorOHEMLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    NOT_IMPLEMENTED;
  }

  template <typename Dtype>
  void RpnAnnotatorOHEMLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }


#ifdef CPU_ONLY
  STUB_GPU(RpnAnnotatorOHEMLayer);
#endif

  INSTANTIATE_CLASS(RpnAnnotatorOHEMLayer);
  REGISTER_LAYER_CLASS(RpnAnnotatorOHEM);

}  // namespace caffe
