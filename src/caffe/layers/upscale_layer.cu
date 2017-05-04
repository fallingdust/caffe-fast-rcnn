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
__global__ void UpscaleBackward(const int nthreads,
    const Dtype* const top_diff, const int num, const int channels,
    const int height, const int width, const int top_height, const int top_width, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const Dtype* const top_slice =
        top_diff + (n * channels + c) * top_height * top_width;
    const int factor_h = int(ceil(top_height / float(height)));
    const int factor_w = int(ceil(top_width / float(width)));
    for (int i = h * factor_h; i < min((h + 1) * factor_h, top_height); i++) {
      for (int j = w * factor_w; j < min((w + 1) * factor_w, top_width); j++) {
        bottom_diff[index] += top_slice[i * top_width + j];
      }
    }
  }
}

template <typename Dtype>
void UpscaleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
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
void UpscaleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  // NOLINT_NEXT_LINE(whitespace/operators)
  UpscaleBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, bottom[0]->num(), bottom[0]->channels(),
    bottom[0]->height(), bottom[0]->width(), top[0]->height(), top[0]->width(), bottom_diff);

  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(UpscaleLayer);


}  // namespace caffe

