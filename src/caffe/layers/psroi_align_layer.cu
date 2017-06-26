// --------------------------------------------------------
// R-FCN
// --------------------------------------------------------

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/psroi_align_layer.hpp"
#include "caffe/util/gpu_util.cuh"

using std::max;
using std::min;

namespace caffe {

  template <typename Dtype>
  __global__ void PSROIAlignForward(
    const int nthreads,
    const Dtype* bottom_data,
    const Dtype spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois,
    const int output_dim,
    const int group_size,
    Dtype* top_data,
    int* mapping_channel) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      // The output is in order (n, ctop, ph, pw)
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int ctop = (index / pooled_width / pooled_height) % output_dim;
      int n = index / pooled_width / pooled_height / output_dim;

      bottom_rois += n * 5;
      int roi_batch_ind = bottom_rois[0];
      Dtype roi_start_w = min(max(bottom_rois[1] * spatial_scale, Dtype(0)), static_cast<Dtype>(width - 1));
      Dtype roi_start_h = min(max(bottom_rois[2] * spatial_scale, Dtype(0)), static_cast<Dtype>(height - 1));
      Dtype roi_end_w = min(max(bottom_rois[3] * spatial_scale, Dtype(0)), static_cast<Dtype>(width - 1));
      Dtype roi_end_h = min(max(bottom_rois[4] * spatial_scale, Dtype(0)), static_cast<Dtype>(height - 1));

      Dtype roi_width = roi_end_w - roi_start_w;
      Dtype roi_height =roi_end_h - roi_start_h;

      Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

      Dtype hstart = static_cast<Dtype>(ph) * bin_size_h + roi_start_h;
      Dtype wstart = static_cast<Dtype>(pw)* bin_size_w + roi_start_w;
      Dtype hend = static_cast<Dtype>(ph + 1) * bin_size_h + roi_start_h;
      Dtype wend = static_cast<Dtype>(pw + 1) * bin_size_w + roi_start_w;

      bool is_empty = (hend <= hstart) || (wend <= wstart);

      int gw = pw;
      int gh = ph;
      int c = (ctop*group_size + gh)*group_size + gw;

      bottom_data += (roi_batch_ind * channels + c) * height * width;
      Dtype out_sum = 0;
      // Selecting four regular locations for bilinear interpolation
      for (Dtype h = hstart + bin_size_h / Dtype(4); h < hend; h += bin_size_h / Dtype(2)) {
        for (Dtype w = wstart + bin_size_w / Dtype(4); w < wend; w += bin_size_w / Dtype(2)) {
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

          out_sum += val;
        }
      }

      top_data[index] = is_empty? 0. : out_sum / 4;
      mapping_channel[index] = c;
    }
  }

  template <typename Dtype>
  void PSROIAlignLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* bottom_rois = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    int* mapping_channel_ptr = mapping_channel_.mutable_gpu_data();
    int count = top[0]->count();
    caffe_gpu_set(count, Dtype(0), top_data);
    caffe_gpu_set(count, -1, mapping_channel_ptr);
    // NOLINT_NEXT_LINE(whitespace/operators)
    PSROIAlignForward<Dtype> << <CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS >> >(count, bottom_data, spatial_scale_,
      channels_, height_, width_, pooled_height_,
      pooled_width_, bottom_rois, output_dim_, group_size_,
      top_data, mapping_channel_ptr);
    CUDA_POST_KERNEL_CHECK;
  }

  template <typename Dtype>
  __global__ void PSROIAlignBackwardAtomic(
    const int nthreads,
    const Dtype* top_diff,
    const int* mapping_channel,
    const int num_rois,
    const Dtype spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int output_dim,
    Dtype* bottom_diff,
    const Dtype* bottom_rois) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      // The output is in order (n, ctop, ph, pw)
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int n = index / pooled_width / pooled_height / output_dim;

      bottom_rois += n * 5;
      int roi_batch_ind = bottom_rois[0];
      Dtype roi_start_w = min(max(bottom_rois[1] * spatial_scale, Dtype(0)), static_cast<Dtype>(width - 1));
      Dtype roi_start_h = min(max(bottom_rois[2] * spatial_scale, Dtype(0)), static_cast<Dtype>(height - 1));
      Dtype roi_end_w = min(max(bottom_rois[3] * spatial_scale, Dtype(0)), static_cast<Dtype>(width - 1));
      Dtype roi_end_h = min(max(bottom_rois[4] * spatial_scale, Dtype(0)), static_cast<Dtype>(height - 1));

      Dtype roi_width = roi_end_w - roi_start_w;
      Dtype roi_height = roi_end_h - roi_start_h;

      Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

      Dtype hstart = static_cast<Dtype>(ph) * bin_size_h + roi_start_h;
      Dtype wstart = static_cast<Dtype>(pw)* bin_size_w + roi_start_w;
      Dtype hend = static_cast<Dtype>(ph + 1) * bin_size_h + roi_start_h;
      Dtype wend = static_cast<Dtype>(pw + 1) * bin_size_w + roi_start_w;

      bool is_empty = (hend <= hstart) || (wend <= wstart);

      // Compute c at bottom
      int c = mapping_channel[index];
      Dtype* offset_bottom_diff = bottom_diff +
        (roi_batch_ind * channels + c) * height * width;
      Dtype diff_val = is_empty ? 0. : top_diff[index] / 4;
      // Selecting four regular locations for bilinear interpolation
      for (Dtype h = hstart + bin_size_h / Dtype(4); h < hend; h += bin_size_h / Dtype(2)) {
        for (Dtype w = wstart + bin_size_w / Dtype(4); w < wend; w += bin_size_w / Dtype(2)) {
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

          caffe_gpu_atomic_add(diff_val * (x_right - w) * (h - y_bottom), offset_bottom_diff + top_left_index);
          caffe_gpu_atomic_add(diff_val * (w - x_left) * (h - y_bottom), offset_bottom_diff + top_right_index);
          caffe_gpu_atomic_add(diff_val * (x_right - w) * (y_top - h), offset_bottom_diff + bottom_left_index);
          caffe_gpu_atomic_add(diff_val * (w - x_left) * (y_top - h), offset_bottom_diff + bottom_right_index);
        }
      }
    }
  }

  template <typename Dtype>
  void PSROIAlignLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (!propagate_down[0]) {
      return;
    }

    const Dtype* bottom_rois = bottom[1]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int bottom_count = bottom[0]->count();
    const int* mapping_channel_ptr = mapping_channel_.gpu_data();
    caffe_gpu_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_gpu_diff());
    caffe_gpu_set(bottom_count, Dtype(0), bottom_diff);
    const int count = top[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    PSROIAlignBackwardAtomic<Dtype> << <CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS >> >(count, top_diff, mapping_channel_ptr,
      top[0]->num(), spatial_scale_, channels_, height_, width_,
      pooled_height_, pooled_width_, output_dim_, bottom_diff,
      bottom_rois);
    CUDA_POST_KERNEL_CHECK;
  }

  INSTANTIATE_LAYER_GPU_FUNCS(PSROIAlignLayer);

}  // namespace caffe
