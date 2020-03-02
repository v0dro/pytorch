#include "gtest/gtest.h"

#include "ATen/ATen.h"
#include <caffe2/core/init.h>
#include <caffe2/core/operator.h>

TEST(TensorOperators, LogicalAND) {
  auto a = at::ones({5, 5}, at::dtype(at::kLong));
  auto b = at::zeros({5, 5}, at::dtype(at::kLong));
  auto c = a & b;
  
  EXPECT_TRUE(c, at::zeros({5, 5}, at::dtype(at::kBool)));
}

TEST(TensorOperators, LogicalOR) {
  auto a = at::ones({5, 5}, at::dtype(at::kLong));
  auto b = at::zeros({5, 5}, at::dtype(at::kLong));
  auto c = a | b;
  
  EXPECT_TRUE(c, at::ones({5, 5}, at::dtype(at::kBool)));  
}
