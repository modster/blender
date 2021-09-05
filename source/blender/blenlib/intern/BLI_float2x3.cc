#include "BLI_float2x3.hh"
#include "BLI_float3x2.hh"

namespace blender {

float2x2 operator*(const float2x3 &a, const float3x2 &b)
{
  float2x2 result;

  result.ptr()[0][0] = a.ptr()[0][0] * b.ptr()[0][0] + a.ptr()[1][0] * b.ptr()[0][1] +
                       a.ptr()[2][0] * b.ptr()[0][2];
  result.ptr()[0][1] = a.ptr()[0][1] * b.ptr()[0][0] + a.ptr()[1][1] * b.ptr()[0][1] +
                       a.ptr()[2][1] * b.ptr()[0][2];

  result.ptr()[1][0] = a.ptr()[0][0] * b.ptr()[1][0] + a.ptr()[1][0] * b.ptr()[1][1] +
                       a.ptr()[2][0] * b.ptr()[1][2];
  result.ptr()[1][1] = a.ptr()[0][1] * b.ptr()[1][0] + a.ptr()[1][1] * b.ptr()[1][1] +
                       a.ptr()[2][1] * b.ptr()[1][2];

  return result;
}

float3x2 float2x3::transpose() const
{
  float3x2 result;

  result.ptr()[0][0] = this->ptr()[0][0];
  result.ptr()[0][1] = this->ptr()[1][0];
  result.ptr()[0][2] = this->ptr()[2][0];

  result.ptr()[1][0] = this->ptr()[0][1];
  result.ptr()[1][1] = this->ptr()[1][1];
  result.ptr()[1][2] = this->ptr()[2][1];

  return result;
}

}  // namespace blender
