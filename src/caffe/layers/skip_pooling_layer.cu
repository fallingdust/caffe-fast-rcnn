#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/skip_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SkipPoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % (channels * 4);
    const int n = index / pooled_width / pooled_height / (channels * 4);
    int hstart = ph * 2;
    int wstart = pw * 2;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c / 4) * height * width;
    int h_index = hstart + c % 4 / 2;
    int w_index = wstart + c % 4 % 2;
    top_data[index] = (h_index >= height || w_index >= width) ? Dtype(0) : bottom_slice[h_index * width + w_index];
  }
}


template <typename Dtype>
void SkipPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SkipPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, bottom[0]->num(), channels_,
      height_, width_, pooled_height_, pooled_width_, top_data);

  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
void SkipPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}


INSTANTIATE_LAYER_GPU_FUNCS(SkipPoolingLayer);


}  // namespace caffe
