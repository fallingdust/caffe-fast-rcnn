#include <algorithm>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/batch_norm_fixed_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

  template <typename TypeParam>
  class BatchNormFixedLayerTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;
   protected:
    BatchNormFixedLayerTest()
        : blob_bottom_(new Blob<Dtype>(5, 2, 3, 4)),
          blob_top_(new Blob<Dtype>()) {
      // fill the values
      FillerParameter filler_param;
      GaussianFiller<Dtype> filler(filler_param);
      filler.Fill(this->blob_bottom_);
      blob_bottom_vec_.push_back(blob_bottom_);
      blob_top_vec_.push_back(blob_top_);
    }
    virtual ~BatchNormFixedLayerTest() { delete blob_bottom_; delete blob_top_; }
    Blob<Dtype>* const blob_bottom_;
    Blob<Dtype>* const blob_top_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
  };

  TYPED_TEST_CASE(BatchNormFixedLayerTest, TestDtypesAndDevices);

  TYPED_TEST(BatchNormFixedLayerTest, TestForward) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;

    BatchNormFixedLayer<Dtype> layer(layer_param);
    layer.blobs().resize(3);
    vector<int> sz;
    sz.push_back(this->blob_bottom_->shape(1));
    layer.blobs()[0].reset(new Blob<Dtype>(sz));
    layer.blobs()[1].reset(new Blob<Dtype>(sz));
    sz[0] = 1;
    layer.blobs()[2].reset(new Blob<Dtype>(sz));
    caffe_set(layer.blobs()[0]->count(), Dtype(1.), layer.blobs()[0]->mutable_cpu_data());
    caffe_set(layer.blobs()[1]->count(), Dtype(2.), layer.blobs()[1]->mutable_cpu_data());
    caffe_set(layer.blobs()[2]->count(), Dtype(3.), layer.blobs()[2]->mutable_cpu_data());
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // Test mean
    int num = this->blob_bottom_->num();
    int channels = this->blob_bottom_->channels();
    int height = this->blob_bottom_->height();
    int width = this->blob_bottom_->width();

    for (int j = 0; j < channels; ++j) {
      Dtype mean = layer.blobs()[0]->cpu_data()[j] / layer.blobs()[2]->cpu_data()[0];
      Dtype var = layer.blobs()[1]->cpu_data()[j] / layer.blobs()[2]->cpu_data()[0];
      for (int i = 0; i < num; ++i) {
        for ( int k = 0; k < height; ++k ) {
          for ( int l = 0; l < width; ++l ) {
            Dtype data = this->blob_top_->data_at(i, j, k, l);
            Dtype data_ = this->blob_bottom_->data_at(i, j, k, l);
            EXPECT_NEAR(data, (data_ - mean) / sqrt(var), 0.001);
          }
        }
      }
    }
  }

  TYPED_TEST(BatchNormFixedLayerTest, TestForwardInplace) {
    typedef typename TypeParam::Dtype Dtype;
    Blob<Dtype> blob_inplace(5, 2, 3, 4);
    vector<Blob<Dtype>*> blob_bottom_vec;
    vector<Blob<Dtype>*> blob_top_vec;
    LayerParameter layer_param;
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(&blob_inplace);
    blob_bottom_vec.push_back(&blob_inplace);
    blob_top_vec.push_back(&blob_inplace);

    BatchNormFixedLayer<Dtype> layer(layer_param);
    layer.blobs().resize(3);
    vector<int> sz;
    sz.push_back(this->blob_bottom_->shape(1));
    layer.blobs()[0].reset(new Blob<Dtype>(sz));
    layer.blobs()[1].reset(new Blob<Dtype>(sz));
    sz[0] = 1;
    layer.blobs()[2].reset(new Blob<Dtype>(sz));
    caffe_set(layer.blobs()[0]->count(), Dtype(4.), layer.blobs()[0]->mutable_cpu_data());
    caffe_set(layer.blobs()[1]->count(), Dtype(5.), layer.blobs()[1]->mutable_cpu_data());
    caffe_set(layer.blobs()[2]->count(), Dtype(6.), layer.blobs()[2]->mutable_cpu_data());

    layer.SetUp(blob_bottom_vec, blob_top_vec);
    Blob<Dtype> temp(blob_inplace.shape());
    caffe_copy(temp.count(), blob_inplace.cpu_data(), temp.mutable_cpu_data());
    layer.Forward(blob_bottom_vec, blob_top_vec);

    // Test mean
    int num = blob_inplace.num();
    int channels = blob_inplace.channels();
    int height = blob_inplace.height();
    int width = blob_inplace.width();

    for (int j = 0; j < channels; ++j) {
      Dtype mean = layer.blobs()[0]->cpu_data()[j] / layer.blobs()[2]->cpu_data()[0];
      Dtype var = layer.blobs()[1]->cpu_data()[j] / layer.blobs()[2]->cpu_data()[0];
      for (int i = 0; i < num; ++i) {
        for ( int k = 0; k < height; ++k ) {
          for ( int l = 0; l < width; ++l ) {
            Dtype data = blob_inplace.data_at(i, j, k, l);
            Dtype data_ = temp.data_at(i, j, k, l);
            EXPECT_NEAR(data, (data_ - mean) / sqrt(var), 0.001);
          }
        }
      }
    }
  }

  TYPED_TEST(BatchNormFixedLayerTest, TestBackward) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;

    BatchNormFixedLayer<Dtype> layer(layer_param);
    layer.blobs().resize(3);
    vector<int> sz;
    sz.push_back(this->blob_bottom_->shape(1));
    layer.blobs()[0].reset(new Blob<Dtype>(sz));
    layer.blobs()[1].reset(new Blob<Dtype>(sz));
    sz[0] = 1;
    layer.blobs()[2].reset(new Blob<Dtype>(sz));
    caffe_set(layer.blobs()[0]->count(), Dtype(1.), layer.blobs()[0]->mutable_cpu_data());
    caffe_set(layer.blobs()[1]->count(), Dtype(2.), layer.blobs()[1]->mutable_cpu_data());
    caffe_set(layer.blobs()[2]->count(), Dtype(3.), layer.blobs()[2]->mutable_cpu_data());
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    caffe_set(this->blob_top_->count(), Dtype(1.), this->blob_top_->mutable_cpu_diff());
    vector<bool> propagate_down(this->blob_bottom_vec_.size(), true);
    layer.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);

    int num = this->blob_bottom_->num();
    int channels = this->blob_bottom_->channels();
    int height = this->blob_bottom_->height();
    int width = this->blob_bottom_->width();

    for (int j = 0; j < channels; ++j) {
      Dtype var = layer.blobs()[1]->cpu_data()[j] / layer.blobs()[2]->cpu_data()[0];
      for (int i = 0; i < num; ++i) {
        for ( int k = 0; k < height; ++k ) {
          for ( int l = 0; l < width; ++l ) {
            Dtype diff = this->blob_top_->diff_at(i, j, k, l);
            Dtype diff_ = this->blob_bottom_->diff_at(i, j, k, l);
            EXPECT_NEAR(diff_, diff / sqrt(var), 0.001);
          }
        }
      }
    }
  }

  TYPED_TEST(BatchNormFixedLayerTest, TestBackwardInplace) {
    typedef typename TypeParam::Dtype Dtype;
    Blob<Dtype> blob_inplace(5, 2, 3, 4);
    vector<Blob<Dtype>*> blob_bottom_vec;
    vector<Blob<Dtype>*> blob_top_vec;
    LayerParameter layer_param;
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(&blob_inplace);
    blob_bottom_vec.push_back(&blob_inplace);
    blob_top_vec.push_back(&blob_inplace);

    BatchNormFixedLayer<Dtype> layer(layer_param);
    layer.blobs().resize(3);
    vector<int> sz;
    sz.push_back(blob_inplace.shape(1));
    layer.blobs()[0].reset(new Blob<Dtype>(sz));
    layer.blobs()[1].reset(new Blob<Dtype>(sz));
    sz[0] = 1;
    layer.blobs()[2].reset(new Blob<Dtype>(sz));
    caffe_set(layer.blobs()[0]->count(), Dtype(1.), layer.blobs()[0]->mutable_cpu_data());
    caffe_set(layer.blobs()[1]->count(), Dtype(2.), layer.blobs()[1]->mutable_cpu_data());
    caffe_set(layer.blobs()[2]->count(), Dtype(3.), layer.blobs()[2]->mutable_cpu_data());

    layer.SetUp(blob_bottom_vec, blob_top_vec);
    layer.Forward(blob_bottom_vec, blob_top_vec);

    caffe_set(blob_inplace.count(), Dtype(1.), blob_inplace.mutable_cpu_diff());
    vector<bool> propagate_down(blob_bottom_vec.size(), true);
    Blob<Dtype> temp(blob_inplace.shape());
    caffe_copy(temp.count(), blob_inplace.cpu_diff(), temp.mutable_cpu_diff());
    const Dtype* diff = blob_top_vec[0]->cpu_diff();
    layer.Backward(blob_top_vec, propagate_down, blob_bottom_vec);

    int num = blob_inplace.num();
    int channels = blob_inplace.channels();
    int height = blob_inplace.height();
    int width = blob_inplace.width();

    for (int j = 0; j < channels; ++j) {
      Dtype var = layer.blobs()[1]->cpu_data()[j] / layer.blobs()[2]->cpu_data()[0];
      for (int i = 0; i < num; ++i) {
        for ( int k = 0; k < height; ++k ) {
          for ( int l = 0; l < width; ++l ) {
            Dtype diff = temp.diff_at(i, j, k, l);
            Dtype diff_ = blob_inplace.diff_at(i, j, k, l);
            EXPECT_NEAR(diff_, diff / sqrt(var), 0.001);
          }
        }
      }
    }
  }

  TYPED_TEST(BatchNormFixedLayerTest, TestGradient) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;

    BatchNormFixedLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-4);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  }

}  // namespace caffe
