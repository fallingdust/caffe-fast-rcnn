#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/skip_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void SkipPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void SkipPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  pooled_height_ = static_cast<int>(ceil(static_cast<float>(height_ - 2) / 2)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(width_ - 2) / 2)) + 1;
  top[0]->Reshape(bottom[0]->num(), channels_ * 4, pooled_height_, pooled_width_);
}

template <typename Dtype>
void SkipPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();

  caffe_set(top_count, Dtype(0), top_data);
  // The main loop
  for (int n = 0; n < bottom[0]->num(); ++n) {
    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          int hstart = ph * 2;
          int wstart = pw * 2;
          int hend = hstart + 2;
          int wend = wstart + 2;
          const int pool_index = ph * pooled_width_ + pw;
          int channel_offset = 0;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              if (h < height_ && w < width_) {
                const int index = h * width_ + w;
                top_data[pool_index + channel_offset] = bottom_data[index];
              }
              channel_offset += top[0]->offset(0, 1);
            }
          }
        }
      }
      // compute offset
      bottom_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 4);
    }
  }
}

template <typename Dtype>
void SkipPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}


#ifdef CPU_ONLY
STUB_GPU(SkipPoolingLayer);
#endif

INSTANTIATE_CLASS(SkipPoolingLayer);
REGISTER_LAYER_CLASS(SkipPooling);

}  // namespace caffe
