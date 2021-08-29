/* Apache License, Version 2.0 */

#include "testing/testing.h"

#include "FN_cpp_type.hh"
#include "FN_field.hh"
#include "FN_multi_function_builder.hh"

namespace blender::fn::tests {

TEST(field, ConstantFunction)
{
  Field constant_field{CPPType::get<int>(),
                       std::make_shared<FieldFunction>(
                           FieldFunction(std::make_unique<CustomMF_Constant<int>>(10), {})),
                       0};

  Array<int> result(4);
  GMutableSpan result_generic(result.as_mutable_span());
  evaluate_fields({constant_field}, IndexMask(IndexRange(4)), {result_generic});

  EXPECT_EQ(result[0], 10);
  EXPECT_EQ(result[1], 10);
  EXPECT_EQ(result[2], 10);
  EXPECT_EQ(result[3], 10);
}

class IndexFieldInput final : public FieldInput {
  StringRef name_ = "Index"; /* TODO: I don't think this is a valid way to override the name. */
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
  Field index_field{CPPType::get<int>(), std::make_shared<IndexFieldInput>()};

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
  Field field_1{CPPType::get<int>(), index_input};
  Field field_2{CPPType::get<int>(), index_input};

  Array<int> result_1(10);
  Array<int> result_2(10);
  GMutableSpan result_generic_1(result_1.as_mutable_span());
  GMutableSpan result_generic_2(result_2.as_mutable_span());

  evaluate_fields({field_1, field_2}, {2, 4, 6, 8}, {result_generic_1, result_generic_2});
  EXPECT_EQ(result_1[2], 2);
  EXPECT_EQ(result_1[4], 4);
  EXPECT_EQ(result_1[6], 6);
  EXPECT_EQ(result_1[8], 8);
  EXPECT_EQ(result_2[2], 2);
  EXPECT_EQ(result_2[4], 4);
  EXPECT_EQ(result_2[6], 6);
  EXPECT_EQ(result_2[8], 8);
}

TEST(field, InputAndFunction)
{
  Field index_field{CPPType::get<int>(), std::make_shared<IndexFieldInput>()};

  Field output_field{CPPType::get<int>(),
                     std::make_shared<FieldFunction>(
                         FieldFunction(std::make_unique<CustomMF_SI_SI_SO<int, int, int>>(
                                           "add", [](int a, int b) { return a + b; }),
                                       {index_field, index_field})),
                     0};

  Array<int> result(10);
  GMutableSpan result_generic(result.as_mutable_span());
  evaluate_fields({output_field}, {2, 4, 6, 8}, {result_generic});
  EXPECT_EQ(result[2], 4);
  EXPECT_EQ(result[4], 8);
  EXPECT_EQ(result[6], 12);
  EXPECT_EQ(result[8], 16);
}

TEST(field, TwoFunctions)
{
  Field index_field{CPPType::get<int>(), std::make_shared<IndexFieldInput>()};

  Field add_field{CPPType::get<int>(),
                  std::make_shared<FieldFunction>(
                      FieldFunction(std::make_unique<CustomMF_SI_SI_SO<int, int, int>>(
                                        "add", [](int a, int b) { return a + b; }),
                                    {index_field, index_field})),
                  0};

  Field result_field{
      CPPType::get<int>(),
      std::make_shared<FieldFunction>(FieldFunction(
          std::make_unique<CustomMF_SI_SO<int, int>>("add_10", [](int a) { return a + 10; }),
          {add_field})),
      0};

  Array<int> result(10);
  GMutableSpan result_generic(result.as_mutable_span());
  evaluate_fields({result_field}, {2, 4, 6, 8}, {result_generic});
  EXPECT_EQ(result[2], 14);
  EXPECT_EQ(result[4], 18);
  EXPECT_EQ(result[6], 22);
  EXPECT_EQ(result[8], 26);
}

}  // namespace blender::fn::tests
