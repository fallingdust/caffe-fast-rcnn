#include <vector>

#include "caffe/layers/deformable_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void DeformableConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); i += 2) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    const Dtype* offset_data = bottom[i + 1]->gpu_data();
    Dtype* top_data = top[i / 2]->mutable_gpu_data();
    if (bottom.size() > 2) {
      reshape_variables(bottom[i], bottom[i + 1], top[i / 2]);
    }
    for (int n = 0; n < num_; ++n) {
      forward_gpu_gemm(bottom_data + n * bottom_dim_, offset_data + n * offset_dim_, weight,
          top_data + n * top_dim_);
      if (bias_term_) {
        const Dtype* bias = blobs_[1]->gpu_data();
        forward_gpu_bias(top_data + n * top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void DeformableConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = blobs_[0]->gpu_data();
  Dtype* weight_diff = blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    if (top.size() > 1) {
      reshape_variables(bottom[2 * i], bottom[2 * i + 1], top[i]);
    }
    // Bias gradient, if necessary.
    if (bias_term_ && param_propagate_down_[1]) {
      Dtype* bias_diff = blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < num_; ++n) {
        backward_gpu_bias(bias_diff, top_diff + n * top_dim_);
      }
    }
    if (param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[2 * i]->gpu_data();
      const Dtype* offset_data = bottom[2 * i + 1]->gpu_data();
      Dtype* bottom_diff = bottom[2 * i]->mutable_gpu_diff();
      Dtype* offset_diff = bottom[2 * i + 1]->mutable_gpu_diff();
      for (int n = 0; n < num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (param_propagate_down_[0]) {
          weight_gpu_gemm(bottom_data + n * bottom_dim_, offset_data + n * offset_dim_,
              top_diff + n * top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[2 * i] && propagate_down[2 * i + 1]) {
          backward_gpu_gemm(top_diff + n * top_dim_, bottom_data + n * bottom_dim_, offset_data + n * offset_dim_, weight,
              bottom_diff + n * bottom_dim_, offset_diff + n * offset_dim_);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DeformableConvolutionLayer);

}  // namespace caffe
