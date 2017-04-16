#include <algorithm>
#include <vector>

#include "caffe/layers/scale_fixed_layer.hpp"

namespace caffe {

template <typename Dtype>
void ScaleFixedLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ScaleParameter& param = this->layer_param_.scale_param();
  // scale is a learned parameter; initialize it
  axis_ = bottom[0]->CanonicalAxisIndex(param.axis());
  const int num_axes = param.num_axes();
  CHECK_GE(num_axes, -1) << "num_axes must be non-negative, "
                         << "or -1 to extend to the end of bottom[0]";
  if (num_axes >= 0) {
    CHECK_GE(bottom[0]->num_axes(), axis_ + num_axes)
      << "scale blob's shape extends past bottom[0]'s shape when applied "
      << "starting with bottom[0] axis = " << axis_;
  }
  has_bias_ = param.bias_term();
  this->blobs_.resize(has_bias_ ? 2 : 1);
  const vector<int>::const_iterator& shape_start =
      bottom[0]->shape().begin() + axis_;
  const vector<int>::const_iterator& shape_end =
      (num_axes == -1) ? bottom[0]->shape().end() : (shape_start + num_axes);
  vector<int> scale_shape(shape_start, shape_end);
  this->blobs_[0].reset(new Blob<Dtype>(scale_shape));
  if (has_bias_) {
    this->blobs_[1].reset(new Blob<Dtype>(scale_shape));
  }
  // Fix scale and bias.
  for (int i = 0; i < this->blobs_.size(); ++i) {
    if (this->layer_param_.param_size() == i) {
      ParamSpec* fixed_param_spec = this->layer_param_.add_param();
      fixed_param_spec->set_lr_mult(0.f);
    } else {
      CHECK_EQ(this->layer_param_.param(i).lr_mult(), 0.f)
        << "Cannot configure learning rate for scale fixed layer.";
    }
  }
}

template <typename Dtype>
void ScaleFixedLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ScaleParameter& param = this->layer_param_.scale_param();
  Blob<Dtype>* scale = this->blobs_[0].get();
  // Always set axis_ == 0 in special case where scale is a scalar
  // (num_axes == 0). Mathematically equivalent for any choice of axis_, so the
  // actual setting can be safely ignored; and computation is most efficient
  // with axis_ == 0 and (therefore) outer_dim_ == 1. (Setting axis_ to
  // bottom[0]->num_axes() - 1, giving inner_dim_ == 1, would be equally
  // performant.)
  axis_ = (scale->num_axes() == 0) ?
      0 : bottom[0]->CanonicalAxisIndex(param.axis());
  CHECK_GE(bottom[0]->num_axes(), axis_ + scale->num_axes())
      << "scale blob's shape extends past bottom[0]'s shape when applied "
      << "starting with bottom[0] axis = " << axis_;
  for (int i = 0; i < scale->num_axes(); ++i) {
    CHECK_EQ(bottom[0]->shape(axis_ + i), scale->shape(i))
        << "dimension mismatch between bottom[0]->shape(" << axis_ + i
        << ") and scale->shape(" << i << ")";
  }
  outer_dim_ = bottom[0]->count(0, axis_);
  scale_dim_ = scale->count();
  inner_dim_ = bottom[0]->count(axis_ + scale->num_axes());
  if (bottom[0] != top[0]) {
    top[0]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void ScaleFixedLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* scale_data = this->blobs_[0].get()->cpu_data();
  const Dtype* bias_data = has_bias_ ? this->blobs_[1].get()->cpu_data() : nullptr;
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < outer_dim_; ++n) {
    for (int d = 0; d < scale_dim_; ++d) {
      const Dtype factor = scale_data[d];
      caffe_cpu_scale(inner_dim_, factor, bottom_data, top_data);
      if (has_bias_) {
        const Dtype bias = bias_data[d];
        caffe_add_scalar(inner_dim_, bias, top_data);
      }
      bottom_data += inner_dim_;
      top_data += inner_dim_;
    }
  }
}

template <typename Dtype>
void ScaleFixedLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Blob<Dtype>* scale = this->blobs_[0].get();
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* scale_data = scale->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int n = 0; n < outer_dim_; ++n) {
      for (int d = 0; d < scale_dim_; ++d) {
        const Dtype factor = scale_data[d];
        caffe_cpu_scale(inner_dim_, factor, top_diff, bottom_diff);
        bottom_diff += inner_dim_;
        top_diff += inner_dim_;
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ScaleFixedLayer);
#endif

INSTANTIATE_CLASS(ScaleFixedLayer);
REGISTER_LAYER_CLASS(ScaleFixed);

}  // namespace caffe
