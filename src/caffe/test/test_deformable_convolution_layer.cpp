#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/deformable_conv_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY

template <typename TypeParam>
class DeformableConvolutionLayerTest : public GPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  DeformableConvolutionLayerTest()
    : blob_bottom_(new Blob<Dtype>(2, 1, 2, 3)),
      blob_bottom_offset_(new Blob<Dtype>(2, 18, 2, 3)),
      blob_top_(new Blob<Dtype>()) {}

  virtual void SetUp() {
    for (int i = 0; i < blob_bottom_->count() / 2; i++) {
      blob_bottom_->mutable_cpu_data()[i] = i + 1;
      blob_bottom_->mutable_cpu_data()[blob_bottom_->count() / 2 + i] = i + 1;
    }
    for (int i = 0; i < blob_bottom_offset_->count(); i++) {
      blob_bottom_offset_->mutable_cpu_data()[i] = 0;
    }
    for (int i = 0; i < 2 * 3 * 2; i++) {
      blob_bottom_offset_->mutable_cpu_data()[blob_bottom_offset_->count() / 2 + i] = 0.5;
    }
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_offset_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~DeformableConvolutionLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_offset_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_offset_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(DeformableConvolutionLayerTest, TestDtypesAndDevices);

TYPED_TEST(DeformableConvolutionLayerTest, TestSimple) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  DeformableConvolutionParameter* deformable_convolution_param = 
      layer_param.mutable_deformable_convolution_param();
  deformable_convolution_param->add_kernel_size(3);
  deformable_convolution_param->add_stride(1);
  deformable_convolution_param->add_pad(1);
  deformable_convolution_param->set_num_output(1);
  deformable_convolution_param->mutable_weight_filler()->set_type("gaussian");
  deformable_convolution_param->mutable_bias_filler()->set_type("constant");
  deformable_convolution_param->mutable_bias_filler()->set_value(1);
  shared_ptr<Layer<Dtype>> layer(new DeformableConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // [1, 2, 1]
  // [2, 3, 2]
  // [1, 2, 1]
  Dtype* kernel_weights = layer->blobs()[0]->mutable_cpu_data();
  for (int i = 0; i < 1; i++) {
    kernel_weights[i * 9] = 1 + i;
    kernel_weights[i * 9 + 1] = 2 + i;
    kernel_weights[i * 9 + 2] = 1 + i;
    kernel_weights[i * 9 + 3] = 2 + i;
    kernel_weights[i * 9 + 4] = 3 + i;
    kernel_weights[i * 9 + 5] = 2 + i;
    kernel_weights[i * 9 + 6] = 1 + i;
    kernel_weights[i * 9 + 7] = 2 + i;
    kernel_weights[i * 9 + 8] = 1 + i;
  }
  EXPECT_EQ(this->blob_top_->shape(0), 2);
  EXPECT_EQ(this->blob_top_->shape(1), 1);
  EXPECT_EQ(this->blob_top_->shape(2), 2);
  EXPECT_EQ(this->blob_top_->shape(3), 3);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_NEAR(this->blob_top_->cpu_data()[0], 21, 1e-5);
  EXPECT_NEAR(this->blob_top_->cpu_data()[1], 35, 1e-5);
  EXPECT_NEAR(this->blob_top_->cpu_data()[2], 31, 1e-5);
  EXPECT_NEAR(this->blob_top_->cpu_data()[3], 27, 1e-5);
  EXPECT_NEAR(this->blob_top_->cpu_data()[4], 44, 1e-5);
  EXPECT_NEAR(this->blob_top_->cpu_data()[5], 37, 1e-5);

  EXPECT_NEAR(this->blob_top_->cpu_data()[6], 21, 1e-5);  // out of boundary
  EXPECT_NEAR(this->blob_top_->cpu_data()[7], 35, 1e-5);  // out of boundary
  EXPECT_NEAR(this->blob_top_->cpu_data()[8], 31, 1e-5);  // out of boundary
  EXPECT_NEAR(this->blob_top_->cpu_data()[9], 27, 1e-5);  // out of boundary
  EXPECT_NEAR(this->blob_top_->cpu_data()[10], 46, 1e-5);
  EXPECT_NEAR(this->blob_top_->cpu_data()[11], 39, 1e-5);
}

TYPED_TEST(DeformableConvolutionLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  DeformableConvolutionParameter* deformable_convolution_param =
      layer_param.mutable_deformable_convolution_param();
  deformable_convolution_param->add_kernel_size(3);
  deformable_convolution_param->add_stride(1);
  deformable_convolution_param->add_pad(1);
  deformable_convolution_param->set_num_output(1);
  deformable_convolution_param->mutable_weight_filler()->set_type("gaussian");
  deformable_convolution_param->mutable_bias_filler()->set_type("gaussian");
  DeformableConvolutionLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

#endif

}
