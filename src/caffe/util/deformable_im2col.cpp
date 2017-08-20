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
inline void deformable_im2col(
  const DType* data_im, const DType* data_offset,
  const int num_spatial_axes,
  const int* im_shape, const int* col_shape, const int* kernel_shape,
  const int* pad, const int* stride, const int* dilation, 
  const int deformable_group, DType* data_col) {
	  NOT_IMPLEMENTED;
}


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
inline void deformable_col2im(
  const DType* data_col, const DType* data_offset,
  const int num_spatial_axes,
  const int* im_shape, const int* col_shape, const int* kernel_shape,
  const int* pad, const int* stride,
  const int* dilation, const int deformable_group,
  DType* grad_im) {
    NOT_IMPLEMENTED;
}


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
inline void deformable_col2im_coord(
  const DType* data_col, const DType* data_im, const DType* data_offset, const int* im_shape,
  const int* col_shape, const int* kernel_shape,
  const int* pad, const int* stride,
  const int* dilation, const int deformable_group, DType* grad_offset) {
    NOT_IMPLEMENTED;
}

}  // namespace caffe
