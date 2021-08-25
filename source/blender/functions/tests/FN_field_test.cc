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
        mask.size(), mask.size(), index_func);
  }
};

class ConstantInputField final : public InputField {
  const GPointer value_;

 public:
  ConstantInputField(const GPointer &value) : InputField(*value.type()), value_(value)
  {
    BLI_assert(value_.get() != nullptr);
  }

  GVArrayPtr get_data(IndexMask mask) const
  {
    return std::make_unique<fn::GVArray_For_SingleValue>(
        *value_.type(), mask.size(), value_.get());
  }
};

TEST(field, ConstantFieldTest)
{
  const int value = 10;
  FieldPtr constant_field = std::make_unique<ConstantInputField>(
      GPointer(CPPType::get<int>(), &value));

  Array<int> result(4);
  GMutableSpan result_generic(result.as_mutable_span());
  MutableSpan<GMutableSpan> result_span{&result_generic, 1};
  evaluate_fields({std::move(constant_field)}, result_span, IndexMask(IndexRange(4)));

  ASSERT_EQ(result[0], 10);
  ASSERT_EQ(result[1], 10);
  ASSERT_EQ(result[2], 10);
  ASSERT_EQ(result[3], 10);
}

}  // namespace blender::fn::tests
