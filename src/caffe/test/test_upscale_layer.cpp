#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/upscale_layer.hpp"


#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"


namespace caffe {

template <typename TypeParam>
class UpscaleLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  UpscaleLayerTest()
      : blob_bottom_0(new Blob<Dtype>()),
        blob_bottom_1(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_0->Reshape(2, 3, 2, 2);
    blob_bottom_1->Reshape(3, 4, 3, 4);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_0);
    blob_bottom_vec_.push_back(blob_bottom_0);
    blob_bottom_vec_.push_back(blob_bottom_1);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~UpscaleLayerTest() {
    delete blob_bottom_0;
    delete blob_bottom_1;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_0;
  Blob<Dtype>* const blob_bottom_1;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  void TestForward() {
    LayerParameter layer_param;
    const int num = blob_bottom_0->num();
    const int channels = blob_bottom_0->channels();
    // Input: 2 x 3 channels of:
    //     [1 2]
    //     [3 4]
    for (int i = 0; i < 4 * num * channels; i += 4) {
      blob_bottom_0->mutable_cpu_data()[i + 0] = 1;
      blob_bottom_0->mutable_cpu_data()[i + 1] = 2;
      blob_bottom_0->mutable_cpu_data()[i + 2] = 3;
      blob_bottom_0->mutable_cpu_data()[i + 3] = 4;
    }
    UpscaleLayer <Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Expected output: 2 x 3 channels of:
    //     [1 1 2 2]
    //     [1 1 2 2]
    //     [3 3 4 4]
    for (int i = 0; i < 12 * num * channels; i += 12) {
      EXPECT_EQ(blob_top_->cpu_data()[i + 0], 1);
      EXPECT_EQ(blob_top_->cpu_data()[i + 1], 1);
      EXPECT_EQ(blob_top_->cpu_data()[i + 2], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 3], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 4], 1);
      EXPECT_EQ(blob_top_->cpu_data()[i + 5], 1);
      EXPECT_EQ(blob_top_->cpu_data()[i + 6], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 7], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 8], 3);
      EXPECT_EQ(blob_top_->cpu_data()[i + 9], 3);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 4);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 4);
    }
  }

  void TestBackward() {
    LayerParameter layer_param;
    const int num = blob_bottom_0->num();
    const int channels = blob_bottom_0->channels();
    // Input: 2 x 3 channels of:
    //     [1 2 3 4]
    //     [5 6 7 8]
    //     [9 10 11 12]
    blob_top_->Reshape(2, 3, 3, 4);
    for (int i = 0; i < 12 * num * channels; i += 12) {
      blob_top_->mutable_cpu_diff()[i + 0] = 1;
      blob_top_->mutable_cpu_diff()[i + 1] = 2;
      blob_top_->mutable_cpu_diff()[i + 2] = 3;
      blob_top_->mutable_cpu_diff()[i + 3] = 4;
      blob_top_->mutable_cpu_diff()[i + 4] = 5;
      blob_top_->mutable_cpu_diff()[i + 5] = 6;
      blob_top_->mutable_cpu_diff()[i + 6] = 7;
      blob_top_->mutable_cpu_diff()[i + 7] = 8;
      blob_top_->mutable_cpu_diff()[i + 8] = 9;
      blob_top_->mutable_cpu_diff()[i + 9] = 10;
      blob_top_->mutable_cpu_diff()[i + 10] = 11;
      blob_top_->mutable_cpu_diff()[i + 11] = 12;
    }
    UpscaleLayer <Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    vector<bool> propagate_down(1, true);
    layer.Backward(blob_top_vec_, propagate_down, blob_bottom_vec_);
    // Expected output: 2 x 3 channels of:
    //     [14 22]
    //     [19 23]
    for (int i = 0; i < 4 * num * channels; i += 4) {
      EXPECT_EQ(blob_bottom_0->cpu_diff()[i + 0], 14);
      EXPECT_EQ(blob_bottom_0->cpu_diff()[i + 1], 22);
      EXPECT_EQ(blob_bottom_0->cpu_diff()[i + 2], 19);
      EXPECT_EQ(blob_bottom_0->cpu_diff()[i + 3], 23);
    }
  }
};

TYPED_TEST_CASE(UpscaleLayerTest, TestDtypesAndDevices);

TYPED_TEST(UpscaleLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UpscaleLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_0->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_0->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_1->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_1->width());
}

TYPED_TEST(UpscaleLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(UpscaleLayerTest, TestBackword) {
  this->TestBackward();
}

TYPED_TEST(UpscaleLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UpscaleLayer <Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
