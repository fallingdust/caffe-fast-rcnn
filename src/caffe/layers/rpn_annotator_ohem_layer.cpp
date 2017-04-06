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
    fg_fraction_ = rpn_anno_param.fg_fraction();
    ignore_label_ = rpn_anno_param.ignore_label();
    positive_label_ = rpn_anno_param.positive_label();
    negative_label_ = rpn_anno_param.negative_label();
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
    const Dtype* bottom_loss = bottom[0]->cpu_data();
    const Dtype* bottom_labels = bottom[1]->cpu_data();
    const Dtype* bottom_bbox_loss_weights = bottom[2]->cpu_data();
    Dtype* top_labels = top[0]->mutable_cpu_data();
    Dtype* top_bbox_loss_weights = top[1]->mutable_cpu_data();
    caffe_set(top[0]->count(), Dtype(ignore_label_), top_labels);
    caffe_set(top[1]->count(), Dtype(0), top_bbox_loss_weights);

    int num_rpns_ = bottom[0]->count();

    // Find rois with max loss
    vector<int> sorted_idx(num_rpns_);
    for (int i = 0; i < num_rpns_; i++) {
      sorted_idx[i] = i;
    }
    std::sort(sorted_idx.begin(), sorted_idx.end(),
              [bottom_loss](int i1, int i2) {
                return bottom_loss[i1] > bottom_loss[i2];
              });

    // Generate output labels for scoring and loss_weights for bbox regression
    int number_pos_left = int(rpn_per_img_ * fg_fraction_ + 0.5);
    for (int i = 0; i < num_rpns_; i++) {
      int index = sorted_idx[i];
      if (bottom_labels[index] == positive_label_) {
        if (number_pos_left > 0) {
          number_pos_left--;
          top_labels[index] = bottom_labels[index];
          int s = index % (width_*height_);
          int n = index / (width_*height_);
          for (int j = 0; j < 4; j++) {
            int bbox_index = (n*4+j)*spatial_dim_+s;
            top_bbox_loss_weights[bbox_index] = bottom_bbox_loss_weights[bbox_index];
          }
        }
      }
    }
    int number_neg_left = rpn_per_img_ - int(rpn_per_img_ * fg_fraction_ + 0.5) + number_pos_left;
    for (int i = 0; i < num_rpns_; i++) {
      int index = sorted_idx[i];
      if (bottom_labels[index] == negative_label_) {
        if (number_neg_left > 0) {
          number_neg_left--;
          top_labels[index] = bottom_labels[index];
          int s = index % (width_*height_);
          int n = index / (width_*height_);
          for (int j = 0; j < 4; j++) {
            int bbox_index = (n*4+j)*spatial_dim_+s;
            top_bbox_loss_weights[bbox_index] =
                bottom_bbox_loss_weights[bbox_index];
          }
        }
      }
    }
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
