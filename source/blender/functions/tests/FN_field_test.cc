/* Apache License, Version 2.0 */

#include "testing/testing.h"

#include "FN_cpp_type.hh"
#include "FN_field.hh"
#include "FN_multi_function_builder.hh"

namespace blender::fn::tests {

TEST(field, ConstantFieldTest)
{
  CustomMF_Constant<int> const_fn{10};
  Field constant_field = Field(CPPType::get<int>(), const_fn);

  Array<int> result(4);
  GMutableSpan result_generic(result.as_mutable_span());
  // MutableSpan<GMutableSpan> result_span{&result_generic, 1};
  evaluate_fields({&constant_field, 1}, {&result_generic, 1}, IndexMask(IndexRange(4)));

  ASSERT_EQ(result[0], 10);
  ASSERT_EQ(result[1], 10);
  ASSERT_EQ(result[2], 10);
  ASSERT_EQ(result[3], 10);
}

}  // namespace blender::fn::tests
