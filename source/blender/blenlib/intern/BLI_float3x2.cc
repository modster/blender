#include "BLI_float3x2.hh"
#include "BLI_float2x3.hh"

namespace blender {

float2x3 float3x2::transpose() const
{
  float2x3 res;

  res.ptr()[0][0] = this->ptr()[0][0];
  res.ptr()[0][1] = this->ptr()[1][0];

  res.ptr()[1][0] = this->ptr()[0][1];
  res.ptr()[1][1] = this->ptr()[1][1];

  res.ptr()[2][0] = this->ptr()[0][2];
  res.ptr()[2][1] = this->ptr()[1][2];

  return res;
}

}  // namespace blender
