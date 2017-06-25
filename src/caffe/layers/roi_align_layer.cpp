
#include <cfloat>

#include "caffe/layers/roi_align_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

/*
 * There are two bottom layers, 0 (actual data) and 1 (ROIs).
 *
 * A ROI is defined as [batch_index x1 y1 x2 y2]
 * The ROI layer (Channels x Width x Height) must == 5
 */
template <typename Dtype>
void ROIAlignLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ROIPoolingParameter roi_pool_param = this->layer_param_.roi_pooling_param();
  CHECK_GT(roi_pool_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(roi_pool_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = roi_pool_param.pooled_h();
  pooled_width_ = roi_pool_param.pooled_w();
  spatial_scale_ = roi_pool_param.spatial_scale();
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 2) << "number of bottom layers must be == 2";
  CHECK_EQ(bottom[1]->channels() * bottom[1]->height() * bottom[1]->width(), 5)
    << "ROI layer C x W x H must be == 5";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
  max_idx_x_.Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
  max_idx_y_.Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  // Number of ROIs
  int num_rois = bottom[1]->num();
  int batch_size = bottom[0]->num();
  int top_count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* argmax_data_x = max_idx_x_.mutable_cpu_data();
  Dtype* argmax_data_y = max_idx_y_.mutable_cpu_data();

  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  caffe_set(top_count, Dtype(-1), argmax_data_x);
  caffe_set(top_count, Dtype(-1), argmax_data_y);

  // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
  for (int n = 0; n < num_rois; ++n) {
    int roi_batch_ind = bottom_rois[0];
    Dtype roi_start_w = min(max(bottom_rois[1] * spatial_scale_, Dtype(0)), static_cast<Dtype>(height_ - 1));
    Dtype roi_start_h = min(max(bottom_rois[2] * spatial_scale_, Dtype(0)), static_cast<Dtype>(height_ - 1));
    Dtype roi_end_w = min(max(bottom_rois[3] * spatial_scale_, Dtype(0)), static_cast<Dtype>(height_ - 1));
    Dtype roi_end_h = min(max(bottom_rois[4] * spatial_scale_, Dtype(0)), static_cast<Dtype>(height_ - 1));
    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_size);

    Dtype roi_height = roi_end_h - roi_start_h;
    Dtype roi_width = roi_end_w - roi_start_w;
    const Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height_);
    const Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width_);

    const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);

    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          // Compute pooling region for this output unit:
          Dtype hstart = static_cast<Dtype>(ph) * bin_size_h + roi_start_h;
          Dtype wstart = static_cast<Dtype>(pw) * bin_size_w + roi_start_w;
          Dtype hend = static_cast<Dtype>(ph + 1) * bin_size_h + roi_start_h;
          Dtype wend = static_cast<Dtype>(pw + 1) * bin_size_w + roi_start_w;

          bool is_empty = (hend <= hstart) || (wend <= wstart);

          const int pool_index = ph * pooled_width_ + pw;
          if (is_empty) {
            top_data[pool_index] = 0;
            argmax_data_x[pool_index] = -1;
            argmax_data_y[pool_index] = -1;
          }

          // Selecting four regular locations for bilinear interpolation
          for (Dtype h = hstart + bin_size_h / Dtype(4); h < hend; h += bin_size_h / Dtype(2)) {
            for (Dtype w = wstart + bin_size_w / Dtype(4); w < wend; w += bin_size_w / Dtype(2)) {
              int x_left = floor(w);
              int x_right = ceil(w);
              if (x_left == x_right) {
                x_right = x_left + 1;
              }
              int y_bottom = floor(h);
              int y_top = ceil(h);
              if (y_bottom == y_top) {
                y_top = y_bottom + 1;
              }

              int top_left_index = y_top * width_ + x_left;
              int top_right_index = y_top * width_ + x_right;
              int bottom_left_index = y_bottom * width_ + x_left;
              int bottom_right_index = y_bottom * width_ + x_right;

              Dtype val = 0;
              val += (1 - w + x_left) * (1 - y_top + h) * batch_data[top_left_index];
              val += (1 - x_right + w) * (1 - y_top + h) * batch_data[top_right_index];
              val += (1 - w + x_left) * (1 - h + y_bottom) * batch_data[bottom_left_index];
              val += (1 - x_right + w) * (1 - h + y_bottom) * batch_data[bottom_right_index];

              if (val > top_data[pool_index]) {
                top_data[pool_index] = val;
                argmax_data_x[pool_index] = w;
                argmax_data_y[pool_index] = h;
              }
            }
          }
        }
      }
      // Increment all data pointers by one channel
      batch_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
      argmax_data_x += max_idx_x_.offset(0, 1);
      argmax_data_y += max_idx_y_.offset(0, 1);
    }
    // Increment ROI data pointer
    bottom_rois += bottom[1]->offset(1);
  }
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0])
    return;

  // Number of ROIs
  const int num_rois = bottom[1]->num();
  CHECK_EQ(num_rois, top[0]->num());
  const int bottom_count = bottom[0]->count();

  const Dtype* bottom_rois = bottom[1]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* argmax_data_x = max_idx_x_.cpu_data();
  const Dtype* argmax_data_y = max_idx_y_.cpu_data();

  caffe_set(bottom_count, Dtype(0), bottom_diff);

  for (int n = 0; n < num_rois; ++n) {
    int roi_batch_ind = bottom_rois[0];
    int roi_start_w = floor(bottom_rois[1] * spatial_scale_);
    int roi_start_h = floor(bottom_rois[2] * spatial_scale_);
    int roi_end_w = ceil(bottom_rois[3] * spatial_scale_);
    int roi_end_h = ceil(bottom_rois[4] * spatial_scale_);

    // Skip if ROI doesn't include (h, w)
    int start_h = max(0, roi_start_h);
    int end_h = min(height_, roi_end_h + 1);
    int start_w = max(0, roi_start_w);
    int end_w = min(width_, roi_end_w + 1);

    // Reverse engineer indices of elements pooled by this ROI
    Dtype* offset_bottom_diff = bottom_diff + bottom[0]->offset(roi_batch_ind);

    for (int c = 0; c < channels_; ++c) {
      for (int h = start_h; h < end_h; ++h) {
        for (int w = start_w; w < end_w; ++w) {
          int index = h * width_ + w;

          Dtype gradient = Dtype(0);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) reduction(+:gradient)
#endif
          for (int ph = 0; ph < pooled_height_; ++ph) {
            for (int pw = 0; pw < pooled_width_; ++pw) {
              int pindex = ph * pooled_width_ + pw;
              Dtype max_x = argmax_data_x[pindex];
              Dtype max_y = argmax_data_y[pindex];

              int x_left = floor(max_x);
              int x_right = ceil(max_x);
              if (x_left == x_right) {
                x_right = x_left + 1;
              }
              int y_bottom = floor(max_y);
              int y_top = ceil(max_y);
              if (y_bottom == y_top) {
                y_top = y_bottom + 1;
              }

              if (x_left == w && y_top == h)
                gradient += (1 - max_x + x_left) * (1 - y_top + max_y) * top_diff[pindex];
              else if (x_left == w && y_bottom == h)
                gradient += (1 - max_x + x_left) * (1 - max_y + y_bottom) * top_diff[pindex];
              else if (x_right == w && y_top == h)
                gradient += (1 - x_right + max_x) * (1 - y_top + max_y) * top_diff[pindex];
              else if (x_right == w && y_bottom == h)
                gradient += (1 - x_right + max_x) * (1 - max_y + y_bottom) * top_diff[pindex];
            }
          }

          offset_bottom_diff[index] += gradient;
        }
      }

      // Increment all data pointers by one channel
      offset_bottom_diff += bottom[0]->offset(0, 1);
      top_diff += top[0]->offset(0, 1);
      argmax_data_x += max_idx_x_.offset(0, 1);
      argmax_data_y += max_idx_y_.offset(0, 1);
    }

    // Increment ROI data pointer
    bottom_rois += bottom[1]->offset(1);
  }
}

#ifdef CPU_ONLY
STUB_GPU(ROIAlignLayer);
#endif

INSTANTIATE_CLASS(ROIAlignLayer);
REGISTER_LAYER_CLASS(ROIAlign);

}  // namespace caffe
