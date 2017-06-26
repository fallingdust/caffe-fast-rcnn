// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/psroi_align_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class PSROIAlignLayerTest : public GPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  PSROIAlignLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(1, 4, 5, 5)),
        blob_bottom_rois_(new Blob<Dtype>(1, 5, 1, 1)),
        blob_top_data_(new Blob<Dtype>()) {
    // fill the values
//    FillerParameter filler_param;
//    filler_param.set_std(10);
//    GaussianFiller<Dtype> filler(filler_param);
//    filler.Fill(this->blob_bottom_data_);
    for (int i = 0; i < blob_bottom_data_->count(); ++i) {
      blob_bottom_data_->mutable_cpu_data()[i] = i + 1;
    }
    blob_bottom_vec_.push_back(blob_bottom_data_);

    blob_top_vec_.push_back(blob_top_data_);
  }
  virtual ~PSROIAlignLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_rois_;
    delete blob_top_data_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_rois_;
  Blob<Dtype>* const blob_top_data_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(PSROIAlignLayerTest, TestDtypesAndDevices);

TYPED_TEST(PSROIAlignLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PSROIPoolingParameter* psroi_pooling_param = layer_param.mutable_psroi_pooling_param();
  psroi_pooling_param->set_group_size(2);
  psroi_pooling_param->set_output_dim(1);
  psroi_pooling_param->set_spatial_scale(1);
  PSROIAlignLayer<Dtype> layer(layer_param);
  this->blob_bottom_rois_->mutable_cpu_data()[0] = 0;
  this->blob_bottom_rois_->mutable_cpu_data()[1] = 0;
  this->blob_bottom_rois_->mutable_cpu_data()[2] = 0;
  this->blob_bottom_rois_->mutable_cpu_data()[3] = 3;
  this->blob_bottom_rois_->mutable_cpu_data()[4] = 3;
  this->blob_bottom_vec_.push_back(this->blob_bottom_rois_);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 1);
  EXPECT_EQ(this->blob_top_data_->channels(), 1);
  EXPECT_EQ(this->blob_top_data_->height(), 2);
  EXPECT_EQ(this->blob_top_data_->width(), 2);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_NEAR(this->blob_top_data_->cpu_data()[0], 5.5, 1e-5);
  EXPECT_NEAR(this->blob_top_data_->cpu_data()[1], 32, 1e-5);
}

TYPED_TEST(PSROIAlignLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PSROIPoolingParameter* psroi_pooling_param = layer_param.mutable_psroi_pooling_param();
  psroi_pooling_param->set_group_size(2);
  psroi_pooling_param->set_output_dim(1);
  psroi_pooling_param->set_spatial_scale(1);
  PSROIAlignLayer<Dtype> layer(layer_param);
    this->blob_bottom_rois_->mutable_cpu_data()[0] = 0;
    this->blob_bottom_rois_->mutable_cpu_data()[1] = 0;
    this->blob_bottom_rois_->mutable_cpu_data()[2] = 0;
    this->blob_bottom_rois_->mutable_cpu_data()[3] = 3;
    this->blob_bottom_rois_->mutable_cpu_data()[4] = 3;
    this->blob_bottom_vec_.push_back(this->blob_bottom_rois_);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
