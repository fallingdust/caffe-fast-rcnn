#include "caffe/util/deformable_im2col.hpp"
#include "caffe/common.hpp"

namespace caffe {

/*!\brief 
 * cpu function of deformable_im2col algorithm
 * \param data_im pointer of an image (C, H, W, ...) in the image batch
 * \param data_offset pointer of offset (C, H, W, ...) in the offset batch
 * \param im_shape input image shape in dimensions (N, C, H, W,)
 * \param col_shape column buffer shape (#channels, output_im_height, output_im_width, ...)
 * \param kernel_shape kernel filter shape
 * \param pad pad shape
 * \param stride stride shape
 * \param dilation dilation shape
 * \param deformable_group #offset group that deformable convolution use
 * \param data_col column buffer pointer
 */
template <typename DType>
void deformable_im2col_cpu(
  const DType* data_im, const DType* data_offset,
  const int num_spatial_axes,
  const int* im_shape, const int* col_shape, const int* kernel_shape,
  const int* pad, const int* stride, const int* dilation, 
  const int deformable_group, DType* data_col) {
	  NOT_IMPLEMENTED;
}

// Explicit instantiation
template void deformable_im2col_cpu<float>(
  const float* data_im, const float* data_offset,
  const int num_spatial_axes,
  const int* im_shape, const int* col_shape, const int* kernel_shape,
  const int* pad, const int* stride, const int* dilation, 
  const int deformable_group, float* data_col);
template void deformable_im2col_cpu<double>(
  const double* data_im, const double* data_offset,
  const int num_spatial_axes,
  const int* im_shape, const int* col_shape, const int* kernel_shape,
  const int* pad, const int* stride, const int* dilation, 
  const int deformable_group, double* data_col);


/*!\brief
 * cpu function of deformable_col2im algorithm
 * \param data_col start pointer of the column buffer to be filled
 * \param data_offset pointer of offset (C, H, W, ...) in the offset batch
 * \param im_shape input image shape in dimensions (N, C, H, W,)
 * \param col_shape column buffer shape
 * \param kernel_shape kernel filter shape
 * \param pad pad shape
 * \param stride stride shape
 * \param dilation dilation shape
 * \param deformable_group #offset group that deformable convolution use
 * \param grad_im pointer of a image (C, H, W,...) in the image batch
 */
template <typename DType>
void deformable_col2im_cpu(
  const DType* data_col, const DType* data_offset,
  const int num_spatial_axes,
  const int* im_shape, const int* col_shape, const int* kernel_shape,
  const int* pad, const int* stride,
  const int* dilation, const int deformable_group,
  DType* grad_im) {
    NOT_IMPLEMENTED;
}

// Explicit instantiation
template void deformable_col2im_cpu<float>(
  const float* data_col, const float* data_offset,
  const int num_spatial_axes,
  const int* im_shape, const int* col_shape, const int* kernel_shape,
  const int* pad, const int* stride,
  const int* dilation, const int deformable_group,
  float* grad_im);
template void deformable_col2im_cpu<double>(
  const double* data_col, const double* data_offset,
  const int num_spatial_axes,
  const int* im_shape, const int* col_shape, const int* kernel_shape,
  const int* pad, const int* stride,
  const int* dilation, const int deformable_group,
  double* grad_im);

/*!\brief
 * cpu function of deformable_col2im_coord algorithm
 * \param data_col start pointer of the column buffer to be filled
 * \param data_im pointer of an image (C, H, W, ...) in the image batch
 * \param data_offset pointer of offset (C, H, W, ...) in the offset batch
 * \param im_shape input image shape in dimensions (N, C, H, W,)
 * \param col_shape column buffer shape
 * \param kernel_shape kernel filter shape
 * \param pad pad shape
 * \param stride stride shape
 * \param dilation dilation shape
 * \param deformable_group #offset group that deformable convolution use
 * \param grad_offset pointer of the offset (C, H, W,...) in the offset batch
 */

template <typename DType>
void deformable_col2im_coord_cpu(
  const DType* data_col, const DType* data_im, const DType* data_offset, const int* im_shape,
  const int* col_shape, const int* kernel_shape,
  const int* pad, const int* stride,
  const int* dilation, const int deformable_group, DType* grad_offset) {
    NOT_IMPLEMENTED;
}

// Explicit instantiation
template void deformable_col2im_coord_cpu<float>(
  const float* data_col, const float* data_im, const float* data_offset, const int* im_shape,
  const int* col_shape, const int* kernel_shape,
  const int* pad, const int* stride,
  const int* dilation, const int deformable_group, float* grad_offset);
template void deformable_col2im_coord_cpu<double>(
  const double* data_col, const double* data_im, const double* data_offset, const int* im_shape,
  const int* col_shape, const int* kernel_shape,
  const int* pad, const int* stride,
  const int* dilation, const int deformable_group, double* grad_offset);

}  // namespace caffe
