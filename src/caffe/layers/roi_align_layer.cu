
#include <cfloat>

#include "caffe/layers/roi_align_layer.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__global__ void ROIAlignForward(const int nthreads, const Dtype* bottom_data,
    const Dtype spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois, Dtype* top_data, Dtype* argmax_data_x, Dtype* argmax_data_y) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];
    Dtype roi_start_w = bottom_rois[1] * spatial_scale;
    Dtype roi_start_h = bottom_rois[2] * spatial_scale;
    Dtype roi_end_w = bottom_rois[3] * spatial_scale;
    Dtype roi_end_h = bottom_rois[4] * spatial_scale;

    Dtype roi_width = roi_end_w - roi_start_w;
    Dtype roi_height = roi_end_h - roi_start_h;
    Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

    Dtype hstart = static_cast<Dtype>(ph) * bin_size_h;
    Dtype wstart = static_cast<Dtype>(pw) * bin_size_w;
    Dtype hend = static_cast<Dtype>(ph + 1) * bin_size_h;
    Dtype wend = static_cast<Dtype>(pw + 1) * bin_size_w;

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, Dtype(0)), static_cast<Dtype>(height - 1));
    hend = min(max(hend + roi_start_h, Dtype(0)), static_cast<Dtype>(height - 1));
    wstart = min(max(wstart + roi_start_w, Dtype(0)), static_cast<Dtype>(width - 1));
    wend = min(max(wend + roi_start_w, Dtype(0)), static_cast<Dtype>(width - 1));
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    Dtype maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    Dtype maxidx_x = -1;
    Dtype maxidx_y = -1;
    bottom_data += (roi_batch_ind * channels + c) * height * width;
    // Selecting the center locations for bilinear interpolation
    Dtype h = hstart + bin_size_h / Dtype(2);
    Dtype w = wstart + bin_size_w / Dtype(2);
    int x_left = floor(w);
    int x_right = ceil(w);
    if (x_right == x_left) {
      x_right = x_left + 1;
    }
    int y_bottom = floor(h);
    int y_top = ceil(h);
    if (y_top == y_bottom) {
      y_top = y_bottom + 1;
    }

    int top_left_index = y_top * width + x_left;
    int top_right_index = y_top * width + x_right;
    int bottom_left_index = y_bottom * width + x_left;
    int bottom_right_index = y_bottom * width + x_right;

    Dtype val = 0;
    val += (1 - w + x_left) * (1 - y_top + h) * bottom_data[top_left_index];
    val += (1 - x_right + w) * (1 - y_top + h) * bottom_data[top_right_index];
    val += (1 - w + x_left) * (1 - h + y_bottom) * bottom_data[bottom_left_index];
    val += (1 - x_right + w) * (1 - h + y_bottom) * bottom_data[bottom_right_index];

    if (val > maxval) {
      maxval = val;
      maxidx_x = w;
      maxidx_y = h;
    }
    top_data[index] = maxval;
    argmax_data_x[index] = maxidx_x;
    argmax_data_y[index] = maxidx_y;
  }
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* argmax_data_x = max_idx_x_.mutable_gpu_data();
  Dtype* argmax_data_y = max_idx_y_.mutable_gpu_data();
  int count = top[0]->count();
  if (bottom[1]->num() == 0) {
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  ROIAlignForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, spatial_scale_, channels_, height_, width_,
      pooled_height_, pooled_width_, bottom_rois, top_data, argmax_data_x, argmax_data_y);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void ROIAlignBackward(const int nthreads, const Dtype* top_diff,
    const Dtype* argmax_data_x, const Dtype* argmax_data_y, const int num_rois, const Dtype spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, Dtype* bottom_diff,
    const Dtype* bottom_rois) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    Dtype gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
      int roi_batch_ind = offset_bottom_rois[0];
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind) {
        continue;
      }

      int roi_start_w = floor(offset_bottom_rois[1] * spatial_scale);
      int roi_start_h = floor(offset_bottom_rois[2] * spatial_scale);
      int roi_end_w = ceil(offset_bottom_rois[3] * spatial_scale);
      int roi_end_h = ceil(offset_bottom_rois[4] * spatial_scale);

      // Skip if ROI doesn't include (h, w)
      const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                           h >= roi_start_h && h <= roi_end_h);
      if (!in_roi) {
        continue;
      }

      int offset = (roi_n * channels + c) * pooled_height * pooled_width;
      const Dtype* offset_top_diff = top_diff + offset;
      const Dtype* offset_argmax_data_x = argmax_data_x + offset;
      const Dtype* offset_argmax_data_y = argmax_data_y + offset;

      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          int pindex = ph * pooled_width + pw;
          Dtype max_x = offset_argmax_data_x[pindex];
          Dtype max_y = offset_argmax_data_y[pindex];

          int x_left = floor(max_x);
          int x_right = ceil(max_x);
          if (x_right == x_left) {
            x_right = x_left + 1;
          }
          int y_bottom = floor(max_y);
          int y_top = ceil(max_y);
          if (y_top == y_bottom) {
            y_top = y_bottom + 1;
          }

          if (x_left == w && y_top == h)
            gradient += (1 - max_x + x_left) * (1 - y_top + max_y) * offset_top_diff[pindex];
          else if (x_left == w && y_bottom == h)
            gradient += (1 - max_x + x_left) * (1 - max_y + y_bottom) * offset_top_diff[pindex];
          else if (x_right == w && y_top == h)
            gradient += (1 - x_right + max_x) * (1 - y_top + max_y) * offset_top_diff[pindex];
          else if (x_right == w && y_bottom == h)
            gradient += (1 - x_right + max_x) * (1 - max_y + y_bottom) * offset_top_diff[pindex];
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0] || top[0]->num() == 0) {
    return;
  }
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  const Dtype* argmax_data_x = max_idx_x_.gpu_data();
  const Dtype* argmax_data_y = max_idx_y_.gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ROIAlignBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, argmax_data_x, argmax_data_y, top[0]->num(), spatial_scale_, channels_,
      height_, width_, pooled_height_, pooled_width_, bottom_diff, bottom_rois);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(ROIAlignLayer);

}  // namespace caffe
