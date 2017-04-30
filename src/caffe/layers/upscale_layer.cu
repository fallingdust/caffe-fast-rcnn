#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/upscale_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void UpscaleForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int bottom_height, const int bottom_width, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * bottom_height * bottom_width;
    const int bh = int(h * bottom_height / float(height));
    const int hw = int(w * bottom_width / float(width));
    top_data[index] = bottom_slice[bh * bottom_width + hw];
  }
}


template <typename Dtype>
void UpscaleingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  UpscaleForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top[0]->num(), top[0]->channels(),
      top[0]->height(), top[0]->width(), bottom[0]->height(), bottom[0]->width(), top_data);

  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
void UpscaleingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}


INSTANTIATE_LAYER_GPU_FUNCS(UpscaleingLayer);


}  // namespace caffe
