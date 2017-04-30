#include <algorithm>
#include <vector>

#include "caffe/layers/upscale_layer.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void UpscaleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void UpscaleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[1]->height(), bottom[1]->width());
}

template <typename Dtype>
void UpscaleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  float factor_h = top[0]->height() / float(bottom[0]->height());
  float factor_w = top[0]->width() / float(bottom[0]->width());

  // The main loop
  for (int n = 0; n < top[0]->num(); ++n) {
    for (int c = 0; c < top[0]->channels(); ++c) {
      for (int h = 0; h < top[0]->height(); ++h) {
        int bh = int(h / factor_h);
        for (int w = 0; w < top[0]->width(); ++w) {
          int bw = int(w / factor_w);
          top_data[h * top[0]->width() + w] = bottom_data[bh * bottom[0]->width() + bw];
        }
      }
      // compute offset
      bottom_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
    }
  }
}

template <typename Dtype>
void UpscaleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}


#ifdef CPU_ONLY
STUB_GPU(UpscaleLayer);
#endif

INSTANTIATE_CLASS(UpscaleLayer);
REGISTER_LAYER_CLASS(Upscale);

}  // namespace caffe
