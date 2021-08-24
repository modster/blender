/* Apache License, Version 2.0 */

#include "testing/testing.h"

#include "FN_field.hh"

namespace blender::fn::tests {

class IndexInputField final : public InputField {
 private:
  //   blender::fn::GVArrayPtr data;

 public:
  GVArrayPtr get_data(IndexMask mask) const
  {
    auto index_func = [](int i) { return i; };
    return std::make_unique<
        fn::GVArray_For_EmbeddedVArray<int, VArray_For_Func<int, decltype(index_func)>>>(
        mask.size());
  }
};

class ConstantInputField final : public InputField {
  GPointer *value_;

 public:
  GVArrayPtr get_data(IndexMask mask) const
  {
    return std::make_unique<fn::GVArray_For_SingleValue>(
        *value_->type(), mask.size(), value_->get());
  }
};

TEST(field, SimpleTest)
{
}

}  // namespace blender::fn::tests
