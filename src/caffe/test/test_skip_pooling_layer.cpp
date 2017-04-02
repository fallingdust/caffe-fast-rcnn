#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/skip_pooling_layer.hpp"


#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class SkipPoolingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SkipPoolingLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 3, 6, 5);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~SkipPoolingLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  // Test for 2x 2 square pooling layer
  void TestForwardSquare() {
    LayerParameter layer_param;
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 3, 5);
    // Input: 2x 2 channels of:
    //     [1 2 5 2 3]
    //     [9 4 1 4 8]
    //     [1 2 5 2 3]
    for (int i = 0; i < 15 * num * channels; i += 15) {
      blob_bottom_->mutable_cpu_data()[i + 0] = 1;
      blob_bottom_->mutable_cpu_data()[i + 1] = 2;
      blob_bottom_->mutable_cpu_data()[i + 2] = 5;
      blob_bottom_->mutable_cpu_data()[i + 3] = 2;
      blob_bottom_->mutable_cpu_data()[i + 4] = 3;
      blob_bottom_->mutable_cpu_data()[i + 5] = 9;
      blob_bottom_->mutable_cpu_data()[i + 6] = 4;
      blob_bottom_->mutable_cpu_data()[i + 7] = 1;
      blob_bottom_->mutable_cpu_data()[i + 8] = 4;
      blob_bottom_->mutable_cpu_data()[i + 9] = 8;
      blob_bottom_->mutable_cpu_data()[i + 10] = 1;
      blob_bottom_->mutable_cpu_data()[i + 11] = 2;
      blob_bottom_->mutable_cpu_data()[i + 12] = 5;
      blob_bottom_->mutable_cpu_data()[i + 13] = 2;
      blob_bottom_->mutable_cpu_data()[i + 14] = 3;
    }
    SkipPoolingLayer <Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels * 4);
    EXPECT_EQ(blob_top_->height(), 2);
    EXPECT_EQ(blob_top_->width(), 3);
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Expected output: 2 x 2 channels of:
    //     [1 5 3] [2 2 0] [9 1 8] [4 4 0]
    //     [1 5 3] [2 2 0] [0 0 0] [0 0 0]
    for (int i = 0; i < 24 * num * channels; i += 24) {
      EXPECT_EQ(blob_top_->cpu_data()[i + 0], 1);
      EXPECT_EQ(blob_top_->cpu_data()[i + 1], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 2], 3);
      EXPECT_EQ(blob_top_->cpu_data()[i + 3], 1);
      EXPECT_EQ(blob_top_->cpu_data()[i + 4], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 5], 3);

      EXPECT_EQ(blob_top_->cpu_data()[i + 6], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 7], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 8], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 9], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 0);

      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 9);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 1);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 8);
      EXPECT_EQ(blob_top_->cpu_data()[i + 15], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 16], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 17], 0);
      
      EXPECT_EQ(blob_top_->cpu_data()[i + 18], 4);
      EXPECT_EQ(blob_top_->cpu_data()[i + 19], 4);
      EXPECT_EQ(blob_top_->cpu_data()[i + 20], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 21], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 22], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 23], 0);
    }
  }
};

TYPED_TEST_CASE(SkipPoolingLayerTest, TestDtypesAndDevices);

TYPED_TEST(SkipPoolingLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SkipPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels() * 4);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

TYPED_TEST(SkipPoolingLayerTest, TestForward) {
  this->TestForwardSquare();
}

}  // namespace caffe
