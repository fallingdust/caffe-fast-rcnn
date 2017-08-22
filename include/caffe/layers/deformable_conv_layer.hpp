#ifndef CAFFE_DEFORMABLE_CONV_LAYER_HPP_
#define CAFFE_DEFORMABLE_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/deformable_im2col.hpp"

namespace caffe {

template <typename Dtype>
class DeformableConvolutionLayer : public Layer<Dtype> {
 public:
  /**
   * @param param provides DeformableConvolutionParameter convolution_param,
   *    with ConvolutionLayer options:
   *  - num_output. The number of filters.
   *  - kernel_size / kernel_h / kernel_w. The filter dimensions, given by
   *  kernel_size for square filters or kernel_h and kernel_w for rectangular
   *  filters.
   *  - stride / stride_h / stride_w (\b optional, default 1). The filter
   *  stride, given by stride_size for equal dimensions or stride_h and stride_w
   *  for different strides. By default the convolution is dense with stride 1.
   *  - pad / pad_h / pad_w (\b optional, default 0). The zero-padding for
   *  convolution, given by pad for equal dimensions or pad_h and pad_w for
   *  different padding. Input padding is computed implicitly instead of
   *  actually padding.
   *  - dilation (\b optional, default 1). The filter
   *  dilation, given by dilation_size for equal dimensions for different
   *  dilation. By default the convolution has dilation 1.
   *  - group (\b optional, default 1). The number of filter groups. Group
   *  convolution is a method for reducing parameterization by selectively
   *  connecting input and output channels. The input and output channel dimensions must be divisible
   *  by the number of groups. For group @f$ \geq 1 @f$, the
   *  convolutional filters' input and output channels are separated s.t. each
   *  group takes 1 / group of the input channels and makes 1 / group of the
   *  output channels. Concretely 4 input channels, 8 output channels, and
   *  2 groups separate input channels 1-2 and output channels 1-4 into the
   *  first group and input channels 3-4 and output channels 5-8 into the second
   *  group.
   *  - deformable_group (\b optional, default 1).
   *  - bias_term (\b optional, default true). Whether to have a bias.
   */
  explicit DeformableConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DeformableConvolution"; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void compute_output_shape();
  void reshape_variables(const Blob<Dtype>* bottom, const Blob<Dtype>* offset, const Blob<Dtype>* top);
  // Helper functions that abstract away the column buffer and gemm arguments.
  // The last argument in forward_cpu_gemm is so that we can skip the im2col if
  // we just called weight_cpu_gemm with the same input.
  void forward_cpu_gemm(const Dtype* input, const Dtype* offset, const Dtype* weights, Dtype* output);
  void forward_cpu_bias(Dtype* output, const Dtype* bias);
  void backward_cpu_gemm(const Dtype* output, const Dtype* input, const Dtype* offset, const Dtype* weights,
                         Dtype* input_grad, Dtype* offset_grad, bool input_propagate, bool offset_propagate);
  void weight_cpu_gemm(const Dtype* input, const Dtype* offset, const Dtype* output, Dtype* weights);
  void backward_cpu_bias(Dtype* bias, const Dtype* input);

#ifndef CPU_ONLY
  void forward_gpu_gemm(const Dtype* input, const Dtype* offset, const Dtype* weights, Dtype* output);
  void forward_gpu_bias(Dtype* output, const Dtype* bias);
  void backward_gpu_gemm(const Dtype* output, const Dtype* input, const Dtype* offset, const Dtype* weights,
                         Dtype* input_grad, Dtype* offset_grad, bool input_propagate, bool offset_propagate);
  void weight_gpu_gemm(const Dtype* input, const Dtype* offset, const Dtype* output, Dtype* weights);
  void backward_gpu_bias(Dtype* bias, const Dtype* input);
#endif

  /// @brief The spatial dimensions of the input.
  inline int input_shape(int i) {
    return (*bottom_shape_)[channel_axis_ + i];
  }

  /// @brief The spatial dimensions of a filter kernel.
  Blob<int> kernel_shape_;
  /// @brief The spatial dimensions of the stride.
  Blob<int> stride_;
  /// @brief The spatial dimensions of the padding.
  Blob<int> pad_;
  /// @brief The spatial dimensions of the dilation.
  Blob<int> dilation_;
  /// @brief The spatial dimensions of the convolution input.
  Blob<int> conv_input_shape_;
  /// @brief The spatial dimensions of the col_buffer.
  vector<int> col_buffer_shape_;
  /// @brief The spatial dimensions of the output.
  vector<int> output_shape_;
  const vector<int>* bottom_shape_;

  int num_spatial_axes_;
  int bottom_dim_;
  int offset_dim_;
  int top_dim_;

  int channel_axis_;
  int num_;
  int channels_;
  int group_;
  int deformable_group_;
  int out_spatial_dim_;
  int weight_offset_;
  int num_output_;
  bool bias_term_;

 private:
  // wrap im2col/col2im so we don't have to remember the (long) argument lists
  inline void conv_im2col_cpu(const Dtype* data, const Dtype* offset, Dtype* col_buff) {
    deformable_im2col(data, offset, num_spatial_axes_, conv_input_shape_.cpu_data(),
                      col_buffer_shape_.data(), kernel_shape_.cpu_data(),
                      pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), deformable_group_, col_buff);
  }
  inline void conv_col2im_cpu(const Dtype* col_buff, const Dtype* offset, Dtype* data_grad) {
    deformable_col2im(col_buff, offset, num_spatial_axes_, conv_input_shape_.cpu_data(),
                      col_buffer_shape_.data(), kernel_shape_.cpu_data(),
                      pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), deformable_group_, data_grad);
  }
  inline void conv_col2im_coord_cpu(const Dtype* col_buff, const Dtype* data, const Dtype* offset, Dtype* offset_grad) {
    deformable_col2im_coord(col_buff, data, offset, conv_input_shape_.cpu_data(),
                            col_buffer_shape_.data(), kernel_shape_.cpu_data(),
                            pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), deformable_group_, offset_grad);
  }
#ifndef CPU_ONLY
  inline void conv_im2col_gpu(const Dtype* data, const Dtype* offset, Dtype* col_buff) {
    deformable_im2col(data, offset, num_spatial_axes_, num_kernels_im2col_,
                      conv_input_shape_.cpu_data(), col_buffer_shape_.data(),
                      kernel_shape_.cpu_data(), pad_.cpu_data(),
                      stride_.cpu_data(), dilation_.cpu_data(), deformable_group_, col_buff);
  }
  inline void conv_col2im_gpu(const Dtype* col_buff, const Dtype* offset, Dtype* data_grad) {
    deformable_col2im(col_buff, offset, num_spatial_axes_, num_kernels_col2im_,
                      conv_input_shape_.cpu_data(), col_buffer_shape_.data(),
                      kernel_shape_.cpu_data(), pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(),
                      deformable_group_, data_grad);
  }
  inline void conv_col2im_coord_gpu(const Dtype* col_buff, const Dtype* data, const Dtype* offset, Dtype* offset_grad) {
    deformable_col2im_coord(col_buff, data, offset, num_spatial_axes_, num_kernels_col2im_coord_,
                            conv_input_shape_.cpu_data(), col_buffer_shape_.data(),
                            kernel_shape_.cpu_data(), pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(),
                            deformable_group_, offset_grad);
  }
#endif

  int num_kernels_im2col_;
  int num_kernels_col2im_;
  int num_kernels_col2im_coord_;
  int conv_out_channels_;
  int conv_in_channels_;
  int conv_out_spatial_dim_;
  int kernel_dim_;
  int col_offset_;
  int output_offset_;

  Blob<Dtype> col_buffer_;
  Blob<Dtype> bias_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_DEFORMABLE_CONV_LAYER_HPP_
