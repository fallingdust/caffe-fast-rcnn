#include <algorithm>
#include <vector>

#include "caffe/layers/batch_norm_fixed_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BatchNormFixedLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int num = bottom[0]->shape(0);
  int spatial_dim = bottom[0]->count()/(channels_*bottom[0]->shape(0));

  if (bottom[0] != top[0]) {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }

  // use the stored mean/variance estimates.
  const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ?
      0 : 1 / this->blobs_[2]->cpu_data()[0];
  caffe_gpu_scale(variance_.count(), scale_factor,
      this->blobs_[0]->gpu_data(), mean_.mutable_gpu_data());
  caffe_gpu_scale(variance_.count(), scale_factor,
      this->blobs_[1]->gpu_data(), variance_.mutable_gpu_data());

  // subtract mean
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.gpu_data(), mean_.gpu_data(), 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, -1, num_by_chans_.gpu_data(),
      spatial_sum_multiplier_.gpu_data(), 1., top_data);

  // normalize variance
  caffe_gpu_add_scalar(variance_.count(), eps_, variance_.mutable_gpu_data());
  caffe_gpu_sqrt(variance_.count(), variance_.gpu_data(),
      variance_.mutable_gpu_data());

  Blob<Dtype> temp(bottom[0]->shape());
  // replicate variance to input size
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.gpu_data(), variance_.gpu_data(), 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.gpu_data(),
      spatial_sum_multiplier_.gpu_data(), 0., temp.mutable_gpu_data());
  caffe_gpu_div(temp.count(), top_data, temp.gpu_data(), top_data);
}

template <typename Dtype>
void BatchNormFixedLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  int num = bottom[0]->shape(0);
  int spatial_dim = bottom[0]->count()/(bottom[0]->shape(0)*channels_);

  Blob<Dtype> temp(bottom[0]->shape());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.gpu_data(),
      spatial_sum_multiplier_.gpu_data(), 0., temp.mutable_gpu_data());
  caffe_gpu_div(bottom[0]->count(), top[0]->gpu_diff(), temp.gpu_data(), bottom[0]->mutable_gpu_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(BatchNormFixedLayer);


}  // namespace caffe
