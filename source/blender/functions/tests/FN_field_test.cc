/* Apache License, Version 2.0 */

#include "testing/testing.h"

#include "FN_cpp_type.hh"
#include "FN_field.hh"
#include "FN_multi_function_builder.hh"

namespace blender::fn::tests {

TEST(field, ConstantFunction)
{
  Field constant_field = Field(CPPType::get<int>(),
                               std::make_shared<FieldFunction>(FieldFunction(
                                   std::make_unique<CustomMF_Constant<int>>(10), {})),
                               0);

  Array<int> result(4);
  GMutableSpan result_generic(result.as_mutable_span());
  evaluate_fields({constant_field}, IndexMask(IndexRange(4)), {result_generic});

  EXPECT_EQ(result[0], 10);
  EXPECT_EQ(result[1], 10);
  EXPECT_EQ(result[2], 10);
  EXPECT_EQ(result[3], 10);
}

class IndexFieldInput final : public FieldInput {
  GVArrayPtr retrieve_data(IndexMask mask) const final
  {
    auto index_func = [](int i) { return i; };
    return std::make_unique<
        GVArray_For_EmbeddedVArray<int, VArray_For_Func<int, decltype(index_func)>>>(
        mask.min_array_size(), mask.min_array_size(), index_func);
  }
};

TEST(field, VArrayInput)
{
  Field index_field = Field(CPPType::get<int>(), std::make_shared<IndexFieldInput>());

  Array<int> result_1(4);
  GMutableSpan result_generic_1(result_1.as_mutable_span());
  evaluate_fields({index_field}, IndexMask(IndexRange(4)), {result_generic_1});
  EXPECT_EQ(result_1[0], 0);
  EXPECT_EQ(result_1[1], 1);
  EXPECT_EQ(result_1[2], 2);
  EXPECT_EQ(result_1[3], 3);

  /* Evaluate a second time, just to test that the first didn't break anything. */
  Array<int> result_2(10);
  GMutableSpan result_generic_2(result_2.as_mutable_span());
  evaluate_fields({index_field}, {2, 4, 6, 8}, {result_generic_2});
  EXPECT_EQ(result_2[2], 2);
  EXPECT_EQ(result_2[4], 4);
  EXPECT_EQ(result_2[6], 6);
  EXPECT_EQ(result_2[8], 8);
}

TEST(field, VArrayInputMultipleOutputs)
{
  std::shared_ptr<FieldInput> index_input = std::make_shared<IndexFieldInput>();
  Field field_1 = Field(CPPType::get<int>(), index_input);
  Field field_2 = Field(CPPType::get<int>(), index_input);

  Array<int> result_1(10);
  Array<int> result_2(10);
  GMutableSpan result_generic_1(result_1.as_mutable_span());
  GMutableSpan result_generic_2(result_2.as_mutable_span());

  evaluate_fields({field_1, field_2}, {2, 4, 6, 8}, {result_generic_1, result_generic_2});
  EXPECT_EQ(result_2[2], 2);
  EXPECT_EQ(result_2[4], 4);
  EXPECT_EQ(result_2[6], 6);
  EXPECT_EQ(result_2[8], 8);
}

}  // namespace blender::fn::tests
