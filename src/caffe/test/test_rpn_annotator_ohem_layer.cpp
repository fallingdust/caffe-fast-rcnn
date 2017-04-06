#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/layers/rpn_annotator_ohem_layer.hpp"


#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class RpnAnnotatorOHEMLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
    RpnAnnotatorOHEMLayerTest()
      : blob_bottom_a_(new Blob<Dtype>()), blob_bottom_b_(new Blob<Dtype>()), blob_bottom_c_(new Blob<Dtype>()),
        blob_top_a_(new Blob<Dtype>()), blob_top_b_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_a_->Reshape(5, 1, 1, 1);
    blob_bottom_b_->Reshape(5, 1, 1, 1);
    blob_bottom_c_->Reshape(5, 4, 1, 1);
    blob_bottom_vec_.push_back(blob_bottom_a_);
    blob_bottom_vec_.push_back(blob_bottom_b_);
    blob_bottom_vec_.push_back(blob_bottom_c_);
    blob_top_vec_.push_back(blob_top_a_);
    blob_top_vec_.push_back(blob_top_b_);
  }
  virtual ~RpnAnnotatorOHEMLayerTest() {
    delete blob_bottom_a_;
    delete blob_bottom_b_;
    delete blob_bottom_c_;
    delete blob_top_a_;
    delete blob_top_b_;
  }
  Blob<Dtype>* const blob_bottom_a_;
  Blob<Dtype>* const blob_bottom_b_;
  Blob<Dtype>* const blob_bottom_c_;
  Blob<Dtype>* const blob_top_a_;
  Blob<Dtype>* const blob_top_b_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  void TestForward() {
    blob_bottom_a_->mutable_cpu_data()[0] = 0.3;
    blob_bottom_a_->mutable_cpu_data()[1] = 0.1;
    blob_bottom_a_->mutable_cpu_data()[2] = 0.5;
    blob_bottom_a_->mutable_cpu_data()[3] = 0.2;
    blob_bottom_a_->mutable_cpu_data()[4] = 0.4;

    blob_bottom_b_->mutable_cpu_data()[0] = 1;
    blob_bottom_b_->mutable_cpu_data()[1] = 0;
    blob_bottom_b_->mutable_cpu_data()[2] = -1;
    blob_bottom_b_->mutable_cpu_data()[3] = 1;
    blob_bottom_b_->mutable_cpu_data()[4] = 1;

    blob_bottom_c_->mutable_cpu_data()[0] = 1;
    blob_bottom_c_->mutable_cpu_data()[1] = 1;
    blob_bottom_c_->mutable_cpu_data()[2] = 1;
    blob_bottom_c_->mutable_cpu_data()[3] = 1;
    blob_bottom_c_->mutable_cpu_data()[4] = 0;
    blob_bottom_c_->mutable_cpu_data()[5] = 0;
    blob_bottom_c_->mutable_cpu_data()[6] = 0;
    blob_bottom_c_->mutable_cpu_data()[7] = 0;
    blob_bottom_c_->mutable_cpu_data()[8] = 0;
    blob_bottom_c_->mutable_cpu_data()[9] = 0;
    blob_bottom_c_->mutable_cpu_data()[10] = 0;
    blob_bottom_c_->mutable_cpu_data()[11] = 0;
    blob_bottom_c_->mutable_cpu_data()[12] = 1;
    blob_bottom_c_->mutable_cpu_data()[13] = 1;
    blob_bottom_c_->mutable_cpu_data()[14] = 1;
    blob_bottom_c_->mutable_cpu_data()[15] = 1;
    blob_bottom_c_->mutable_cpu_data()[16] = 1;
    blob_bottom_c_->mutable_cpu_data()[17] = 1;
    blob_bottom_c_->mutable_cpu_data()[18] = 1;
    blob_bottom_c_->mutable_cpu_data()[19] = 1;

    LayerParameter layer_param;
    RpnAnnotatorOHEMParameter* rpn_annotator_ohem_param = layer_param.mutable_rpn_annotator_ohem_param();
    rpn_annotator_ohem_param->set_ignore_label(-1);
    rpn_annotator_ohem_param->set_positive_label(1);
    rpn_annotator_ohem_param->set_negative_label(0);
    rpn_annotator_ohem_param->set_rpn_per_img(2);
    rpn_annotator_ohem_param->set_fg_fraction(0.5);
    RpnAnnotatorOHEMLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(blob_bottom_vec_, blob_top_vec_);

    EXPECT_EQ(blob_top_a_->cpu_data()[0], -1);
    EXPECT_EQ(blob_top_a_->cpu_data()[1], 0);
    EXPECT_EQ(blob_top_a_->cpu_data()[2], -1);
    EXPECT_EQ(blob_top_a_->cpu_data()[3], -1);
    EXPECT_EQ(blob_top_a_->cpu_data()[4], 1);

    EXPECT_EQ(blob_top_b_->cpu_data()[0], 0);
    EXPECT_EQ(blob_top_b_->cpu_data()[1], 0);
    EXPECT_EQ(blob_top_b_->cpu_data()[2], 0);
    EXPECT_EQ(blob_top_b_->cpu_data()[3], 0);
    EXPECT_EQ(blob_top_b_->cpu_data()[4], 0);
    EXPECT_EQ(blob_top_b_->cpu_data()[5], 0);
    EXPECT_EQ(blob_top_b_->cpu_data()[6], 0);
    EXPECT_EQ(blob_top_b_->cpu_data()[7], 0);
    EXPECT_EQ(blob_top_b_->cpu_data()[8], 0);
    EXPECT_EQ(blob_top_b_->cpu_data()[9], 0);
    EXPECT_EQ(blob_top_b_->cpu_data()[10], 0);
    EXPECT_EQ(blob_top_b_->cpu_data()[11], 0);
    EXPECT_EQ(blob_top_b_->cpu_data()[12], 0);
    EXPECT_EQ(blob_top_b_->cpu_data()[13], 0);
    EXPECT_EQ(blob_top_b_->cpu_data()[14], 0);
    EXPECT_EQ(blob_top_b_->cpu_data()[15], 0);
    EXPECT_EQ(blob_top_b_->cpu_data()[16], 1);
    EXPECT_EQ(blob_top_b_->cpu_data()[17], 1);
    EXPECT_EQ(blob_top_b_->cpu_data()[18], 1);
    EXPECT_EQ(blob_top_b_->cpu_data()[19], 1);
  }
};

TYPED_TEST_CASE(RpnAnnotatorOHEMLayerTest, TestDtypesAndDevices);

TYPED_TEST(RpnAnnotatorOHEMLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  RpnAnnotatorOHEMParameter* rpn_annotator_ohem_param = layer_param.mutable_rpn_annotator_ohem_param();
  rpn_annotator_ohem_param->set_ignore_label(-1);
  rpn_annotator_ohem_param->set_positive_label(1);
  rpn_annotator_ohem_param->set_negative_label(0);
  rpn_annotator_ohem_param->set_rpn_per_img(2);
  rpn_annotator_ohem_param->set_fg_fraction(0.5);

  RpnAnnotatorOHEMLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_a_->num(), this->blob_bottom_b_->num());
  EXPECT_EQ(this->blob_top_a_->channels(), this->blob_bottom_b_->channels());
  EXPECT_EQ(this->blob_top_a_->height(), this->blob_bottom_b_->height());
  EXPECT_EQ(this->blob_top_a_->width(), this->blob_bottom_b_->width());

  EXPECT_EQ(this->blob_top_b_->num(), this->blob_bottom_c_->num());
  EXPECT_EQ(this->blob_top_b_->channels(), this->blob_bottom_c_->channels());
  EXPECT_EQ(this->blob_top_b_->height(), this->blob_bottom_c_->height());
  EXPECT_EQ(this->blob_top_b_->width(), this->blob_bottom_c_->width());
}

TYPED_TEST(RpnAnnotatorOHEMLayerTest, TestForward) {
  this->TestForward();
}

}  // namespace caffe
