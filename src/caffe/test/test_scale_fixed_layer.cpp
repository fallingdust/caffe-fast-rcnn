#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/scale_fixed_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ScaleFixedLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ScaleFixedLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    filler_param.set_min(1);
    filler_param.set_max(10);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~ScaleFixedLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ScaleFixedLayerTest, TestDtypesAndDevices);

TYPED_TEST(ScaleFixedLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ScaleParameter* param = layer_param.mutable_scale_param();
  param->set_bias_term(true);
  shared_ptr<ScaleFixedLayer<Dtype> > layer(new ScaleFixedLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  caffe_set(layer->blobs()[0]->count(), Dtype(2.), layer->blobs()[0]->mutable_cpu_data());
  caffe_set(layer->blobs()[1]->count(), Dtype(1.), layer->blobs()[1]->mutable_cpu_data());
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
    const Dtype scale = layer->blobs()[0]->cpu_data()[c];
    const Dtype bias = layer->blobs()[1]->cpu_data()[c];
    for (int n = 0; n < this->blob_bottom_->num(); ++n) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_NEAR(this->blob_top_->data_at(n, c, h, w),
                      this->blob_bottom_->data_at(n, c, h, w) * scale + bias,
                      1e-5);
        }
      }
    }
  }
}

TYPED_TEST(ScaleFixedLayerTest, TestForwardInplace) {
  typedef typename TypeParam::Dtype Dtype;

  Blob<Dtype> blob_inplace(5, 2, 3, 4);
  vector<Blob<Dtype>*> blob_bottom_vec;
  vector<Blob<Dtype>*> blob_top_vec;
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(&blob_inplace);
  blob_bottom_vec.push_back(&blob_inplace);
  blob_top_vec.push_back(&blob_inplace);

  LayerParameter layer_param;
  ScaleParameter* param = layer_param.mutable_scale_param();
  param->set_bias_term(true);
  shared_ptr<ScaleFixedLayer<Dtype> > layer(new ScaleFixedLayer<Dtype>(layer_param));
  layer->SetUp(blob_bottom_vec, blob_top_vec);
  caffe_set(layer->blobs()[0]->count(), Dtype(2.), layer->blobs()[0]->mutable_cpu_data());
  caffe_set(layer->blobs()[1]->count(), Dtype(1.), layer->blobs()[1]->mutable_cpu_data());
  Blob<Dtype> temp(blob_inplace.shape());
  caffe_copy(temp.count(), blob_inplace.cpu_data(), temp.mutable_cpu_data());
  layer->Forward(blob_bottom_vec, blob_top_vec);
  for (int c = 0; c < blob_inplace.channels(); ++c) {
    const Dtype scale = layer->blobs()[0]->cpu_data()[c];
    const Dtype bias = layer->blobs()[1]->cpu_data()[c];
    for (int n = 0; n < blob_inplace.num(); ++n) {
      for (int h = 0; h < blob_inplace.height(); ++h) {
        for (int w = 0; w < blob_inplace.width(); ++w) {
          EXPECT_NEAR(blob_inplace.data_at(n, c, h, w),
                      temp.data_at(n, c, h, w) * scale + bias,
                      1e-5);
        }
      }
    }
  }
}

TYPED_TEST(ScaleFixedLayerTest, TestBackward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ScaleParameter* param = layer_param.mutable_scale_param();
  param->set_bias_term(true);
  shared_ptr<ScaleFixedLayer<Dtype> > layer(new ScaleFixedLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  caffe_set(layer->blobs()[0]->count(), Dtype(2.), layer->blobs()[0]->mutable_cpu_data());
  caffe_set(layer->blobs()[1]->count(), Dtype(1.), layer->blobs()[1]->mutable_cpu_data());
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  caffe_set(this->blob_top_->count(), Dtype(1.), this->blob_top_->mutable_cpu_diff());
  vector<bool> propagate_down(this->blob_bottom_vec_.size(), true);
  layer->Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);

  for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
    const Dtype scale = layer->blobs()[0]->cpu_data()[c];
    for (int n = 0; n < this->blob_bottom_->num(); ++n) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_NEAR(this->blob_bottom_->diff_at(n, c, h, w),
                      this->blob_top_->diff_at(n, c, h, w) * scale,
                      1e-5);
        }
      }
    }
  }
}

TYPED_TEST(ScaleFixedLayerTest, TestBackwardInplace) {
  typedef typename TypeParam::Dtype Dtype;

  Blob<Dtype> blob_inplace(5, 2, 3, 4);
  vector<Blob<Dtype>*> blob_bottom_vec;
  vector<Blob<Dtype>*> blob_top_vec;
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(&blob_inplace);
  blob_bottom_vec.push_back(&blob_inplace);
  blob_top_vec.push_back(&blob_inplace);

  LayerParameter layer_param;
  ScaleParameter* param = layer_param.mutable_scale_param();
  param->set_bias_term(true);
  shared_ptr<ScaleFixedLayer<Dtype> > layer(new ScaleFixedLayer<Dtype>(layer_param));
  layer->SetUp(blob_bottom_vec, blob_top_vec);
  caffe_set(layer->blobs()[0]->count(), Dtype(2.), layer->blobs()[0]->mutable_cpu_data());
  caffe_set(layer->blobs()[1]->count(), Dtype(1.), layer->blobs()[1]->mutable_cpu_data());
  layer->Forward(blob_bottom_vec, blob_top_vec);
  caffe_set(blob_inplace.count(), Dtype(1.), blob_inplace.mutable_cpu_diff());
  vector<bool> propagate_down(blob_bottom_vec.size(), true);
  Blob<Dtype> temp(blob_inplace.shape());
  caffe_copy(temp.count(), blob_inplace.cpu_diff(), temp.mutable_cpu_diff());
  layer->Backward(blob_top_vec, propagate_down, blob_bottom_vec);

  for (int c = 0; c < blob_inplace.channels(); ++c) {
    const Dtype scale = layer->blobs()[0]->cpu_data()[c];
    for (int n = 0; n < blob_inplace.num(); ++n) {
      for (int h = 0; h < blob_inplace.height(); ++h) {
        for (int w = 0; w < blob_inplace.width(); ++w) {
          EXPECT_NEAR(blob_inplace.diff_at(n, c, h, w),
                      temp.diff_at(n, c, h, w) * scale,
                      1e-5);
        }
      }
    }
  }
}

}  // namespace caffe
