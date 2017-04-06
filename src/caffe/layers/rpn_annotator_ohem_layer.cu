#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/rpn_annotator_ohem_layer.hpp"

using std::max;
using std::min;

namespace caffe {
  template <typename Dtype>
  void RpnAnnotatorOHEMLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
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
    int number_left = rpn_per_img_;
    for (int i = 0; i < num_rpns_; i++) {
      int index = sorted_idx[i];
      int s = index % (width_*height_);
      int n = index / (width_*height_);
      if (bottom_labels[index] == ignore_label_) {
        continue;
      }
      if (number_left > 0) {
        number_left--;
        top_labels[index] = bottom_labels[index];
        for (int j = 0; j < 4; j++) {
          int bbox_index = (n*4+j)*spatial_dim_+s;
          top_bbox_loss_weights[bbox_index] =
            bottom_bbox_loss_weights[bbox_index];
        }
      }
    }
  }

  template <typename Dtype>
  void RpnAnnotatorOHEMLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    return;
  }

  INSTANTIATE_LAYER_GPU_FUNCS(RpnAnnotatorOHEMLayer);

}  // namespace caffe
