#include "BLI_color.hh"
#include "BLI_math_base.h"
#include "BLI_math_color.h"
#include "BLI_sys_types.h"

#include "IMB_colormanagement.h"
#include "IMB_imbuf.h"

#include <type_traits>

namespace blender {
namespace paint {

using Color4f = blender::ColorSceneLinear4f<blender::eAlpha::Straight>;
using Color4b = blender::ColorSceneLinearByteEncoded4b<blender::eAlpha::Straight>;

struct ByteTraits {
  using ValueType = uchar;
  using BlendType = int;

  inline static const uchar range = 255; /* Zero-based maximum value. */
  inline static const float frange = 255.0f; /* Convienent floating-point version of range. */
  inline static const int cmpRange = 254;
  inline static const int expandedRange = 256; /* One-based maxium value. */ 

  inline static const int bytes = 1;
  inline static const float unit = 255.0f;

  static inline BlendType divide_round(BlendType a, BlendType b)
  {
    return divide_round_i(a, b);
  }

  static inline BlendType min(BlendType a, BlendType b)
  {
    return min_ii(a, b);
  }

  static inline BlendType max(BlendType a, BlendType b)
  {
    return max_ii(a, b);
  }
  /* Discretizes in steps of 1.0 / range */
  static inline ValueType round(float f)
  {
    return round_fl_to_uchar(f);
  }
};

struct FloatTraits {
  using ValueType = float;
  using BlendType = float;

  inline const static float range = 1.0f;
  inline const static float frange = 1.0f;
  inline const static float cmpRange = 0.9999f;
  inline static const int expandedRange = 1.0f;

  inline const static float unit = 1.0f;
  inline const static int bytes = 4;

  static inline BlendType divide_round(BlendType a, BlendType b)
  {
    return a / b;
  }

  static inline BlendType min(BlendType a, BlendType b)
  {
    return min_ff(a, b);
  }

  static inline BlendType max(BlendType a, BlendType b)
  {
    return min_ff(a, b);
  }

  /* Discretizes in steps of 1.0 / range */
  static inline ValueType round(float f)
  {
    return f;
  }
};

static float get_luminance(Color4f c)
{
  return IMB_colormanagement_get_luminance(&c.r);
}

static int get_luminance(Color4b c)
{
  return IMB_colormanagement_get_luminance_byte(&c.r);
}

#define EPS_SATURATION 0.0005f

/* -------------------------------------------------------------------- */
/** \name Color Blending Modes
 * \{ */

template<typename Color, typename Traits>
static Color mcol_blend(Color col_src, Color col_dst, typename Traits::BlendType fac)
{
  using Value = typename Traits::ValueType;
  using Blend = typename Traits::BlendType;

  Value *cp_src, *cp_dst, *cp_mix;
  Color col_mix(0, 0, 0, 0);
  Blend mfac;

  if (fac == 0) {
    return col_src;
  }

  if (fac >= Traits::range) {
    return col_dst;
  }

  mfac = Traits::range - fac;

  cp_src = &col_src.r;
  cp_dst = &col_dst.r;
  cp_mix = &col_mix.r;

  /* Updated to use the rgb squared color model which blends nicer. */
  Blend r1 = cp_src[0] * cp_src[0];
  Blend g1 = cp_src[1] * cp_src[1];
  Blend b1 = cp_src[2] * cp_src[2];
  Blend a1 = cp_src[3] * cp_src[3];

  Blend r2 = cp_dst[0] * cp_dst[0];
  Blend g2 = cp_dst[1] * cp_dst[1];
  Blend b2 = cp_dst[2] * cp_dst[2];
  Blend a2 = cp_dst[3] * cp_dst[3];

  cp_mix[0] = Traits::round(sqrtf(Traits::divide_round((mfac * r1 + fac * r2), Traits::range)));
  cp_mix[1] = Traits::round(sqrtf(Traits::divide_round((mfac * g1 + fac * g2), Traits::range)));
  cp_mix[2] = Traits::round(sqrtf(Traits::divide_round((mfac * b1 + fac * b2), Traits::range)));
  cp_mix[3] = Traits::round(sqrtf(Traits::divide_round((mfac * a1 + fac * a2), Traits::range)));
  return Color(col_mix[0], col_mix[1], col_mix[2], col_mix[3]);

  return col_mix;
}

template<typename Color, typename Traits>
static Color mcol_add(Color col_src, Color col_dst, typename Traits::BlendType fac)
{
  using Value = typename Traits::ValueType;
  using Blend = typename Traits::BlendType;

  Value *cp_src, *cp_dst, *cp_mix;
  Blend temp;
  Color col_mix(0, 0, 0, 0);

  if (fac == 0) {
    return col_src;
  }

  cp_src = (Value *)&col_src.r;
  cp_dst = (Value *)&col_dst.r;
  cp_mix = (Value *)&col_mix.r;

  temp = cp_src[0] + Traits::divide_round((fac * cp_dst[0]), Traits::range);
  cp_mix[0] = (temp > Traits::cmpRange) ? Traits::range : temp;
  temp = cp_src[1] + Traits::divide_round((fac * cp_dst[1]), Traits::range);
  cp_mix[1] = (temp > Traits::cmpRange) ? Traits::range : temp;
  temp = cp_src[2] + Traits::divide_round((fac * cp_dst[2]), Traits::range);
  cp_mix[2] = (temp > Traits::cmpRange) ? Traits::range : temp;
  temp = cp_src[3] + Traits::divide_round((fac * cp_dst[3]), Traits::range);
  cp_mix[3] = (temp > Traits::cmpRange) ? Traits::range : temp;

  return col_mix;
}

template<typename Color, typename Traits>
static Color mcol_sub(Color col_src, Color col_dst, typename Traits::BlendType fac)
{
  using Value = typename Traits::ValueType;
  using Blend = typename Traits::BlendType;

  Value *cp_src, *cp_dst, *cp_mix;
  Blend temp;
  Color col_mix(0, 0, 0, 0);

  cp_src = (Value *)&col_src.r;
  cp_dst = (Value *)&col_dst.r;
  cp_mix = (Value *)&col_mix.r;

  temp = cp_src[0] - Traits::divide_round((fac * cp_dst[0]), Traits::range);
  cp_mix[0] = (temp < 0) ? 0 : temp;
  temp = cp_src[1] - Traits::divide_round((fac * cp_dst[1]), Traits::range);
  cp_mix[1] = (temp < 0) ? 0 : temp;
  temp = cp_src[2] - Traits::divide_round((fac * cp_dst[2]), Traits::range);
  cp_mix[2] = (temp < 0) ? 0 : temp;
  temp = cp_src[3] - Traits::divide_round((fac * cp_dst[3]), Traits::range);
  cp_mix[3] = (temp < 0) ? 0 : temp;

  return col_mix;
}

template<typename Color, typename Traits>
static Color mcol_mul(Color col_src, Color col_dst, typename Traits::BlendType fac)
{
  using Value = typename Traits::ValueType;
  using Blend = typename Traits::BlendType;

  Value *cp_src, *cp_dst, *cp_mix;
  Blend mfac;
  Color col_mix(0, 0, 0, 0);

  if (fac == 0) {
    return col_src;
  }

  mfac = Traits::range - fac;

  cp_src = (Value *)&col_src;
  cp_dst = (Value *)&col_dst;
  cp_mix = (Value *)&col_mix;

  /* first mul, then blend the fac */
  cp_mix[0] = Traits::divide_round(mfac * cp_src[0] * Traits::range + fac * cp_dst[0] * cp_src[0],
                                   Traits::range * Traits::range);
  cp_mix[1] = Traits::divide_round(mfac * cp_src[1] * Traits::range + fac * cp_dst[1] * cp_src[1],
                                   Traits::range * Traits::range);
  cp_mix[2] = Traits::divide_round(mfac * cp_src[2] * Traits::range + fac * cp_dst[2] * cp_src[2],
                                   Traits::range * Traits::range);
  cp_mix[3] = Traits::divide_round(mfac * cp_src[3] * Traits::range + fac * cp_dst[3] * cp_src[3],
                                   Traits::range * Traits::range);

  return col_mix;
}

template<typename Color, typename Traits>
static Color mcol_lighten(Color col_src, Color col_dst, typename Traits::BlendType fac)
{
  using Value = typename Traits::ValueType;
  using Blend = typename Traits::BlendType;

  Value *cp_src, *cp_dst, *cp_mix;
  Blend mfac;
  Color col_mix(0, 0, 0, 0);

  if (fac == 0) {
    return col_src;
  }
  if (fac >= Traits::range) {
    return col_dst;
  }

  mfac = Traits::range - fac;

  cp_src = (Value *)&col_src;
  cp_dst = (Value *)&col_dst;
  cp_mix = (Value *)&col_mix;

  /* See if we're lighter, if so mix, else don't do anything.
   * if the paint color is darker then the original, then ignore */
  if (get_luminance(cp_src) > get_luminance(cp_dst)) {
    return col_src;
  }

  cp_mix[0] = Traits::divide_round(mfac * cp_src[0] + fac * cp_dst[0], Traits::range);
  cp_mix[1] = Traits::divide_round(mfac * cp_src[1] + fac * cp_dst[1], Traits::range);
  cp_mix[2] = Traits::divide_round(mfac * cp_src[2] + fac * cp_dst[2], Traits::range);
  cp_mix[3] = Traits::divide_round(mfac * cp_src[3] + fac * cp_dst[3], Traits::range);

  return col_mix;
}

template<typename Color, typename Traits>
static Color mcol_darken(Color col_src, Color col_dst, typename Traits::BlendType fac)
{
  using Value = typename Traits::ValueType;
  using Blend = typename Traits::BlendType;

  Value *cp_src, *cp_dst, *cp_mix;
  Blend mfac;
  Color col_mix(0, 0, 0, 0);

  if (fac == 0) {
    return col_src;
  }
  if (fac >= Traits::range) {
    return col_dst;
  }

  mfac = Traits::range - fac;

  cp_src = (Value *)&col_src;
  cp_dst = (Value *)&col_dst;
  cp_mix = (Value *)&col_mix;

  /* See if we're darker, if so mix, else don't do anything.
   * if the paint color is brighter then the original, then ignore */
  if (get_luminance(cp_src) < get_luminance(cp_dst)) {
    return col_src;
  }

  cp_mix[0] = Traits::divide_round((mfac * cp_src[0] + fac * cp_dst[0]), Traits::range);
  cp_mix[1] = Traits::divide_round((mfac * cp_src[1] + fac * cp_dst[1]), Traits::range);
  cp_mix[2] = Traits::divide_round((mfac * cp_src[2] + fac * cp_dst[2]), Traits::range);
  cp_mix[3] = Traits::divide_round((mfac * cp_src[3] + fac * cp_dst[3]), Traits::range);
  return col_mix;
}

template<typename Color, typename Traits>
static Color mcol_colordodge(Color col_src, Color col_dst, typename Traits::BlendType fac)
{
  using Value = typename Traits::ValueType;
  using Blend = typename Traits::BlendType;

  Value *cp_src, *cp_dst, *cp_mix;
  Blend mfac, temp;
  Color col_mix(0, 0, 0, 0);

  if (fac == 0) {
    return col_src;
  }

  mfac = Traits::range - fac;

  cp_src = (Value *)&col_src;
  cp_dst = (Value *)&col_dst;
  cp_mix = (Value *)&col_mix;

  Blend dodgefac = (Blend)((float)Traits::range * 0.885f); /* ~225/255 */

  temp = (cp_dst[0] == Traits::range) ?
             Traits::range :
             Traits::min((cp_src[0] * dodgefac) / (Traits::range - cp_dst[0]), Traits::range);
  cp_mix[0] = (mfac * cp_src[0] + temp * fac) / Traits::range;
  temp = (cp_dst[1] == Traits::range) ?
             Traits::range :
             Traits::min((cp_src[1] * dodgefac) / (Traits::range - cp_dst[1]), Traits::range);
  cp_mix[1] = (mfac * cp_src[1] + temp * fac) / Traits::range;
  temp = (cp_dst[2] == Traits::range) ?
             Traits::range :
             Traits::min((cp_src[2] * dodgefac) / (Traits::range - cp_dst[2]), Traits::range);
  cp_mix[2] = (mfac * cp_src[2] + temp * fac) / Traits::range;
  temp = (cp_dst[3] == Traits::range) ?
             Traits::range :
             Traits::min((cp_src[3] * dodgefac) / (Traits::range - cp_dst[3]), Traits::range);
  cp_mix[3] = (mfac * cp_src[3] + temp * fac) / Traits::range;
  return col_mix;
}

template<typename Color, typename Traits>
static Color mcol_difference(Color col_src, Color col_dst, typename Traits::BlendType fac)
{
  using Value = typename Traits::ValueType;
  using Blend = typename Traits::BlendType;

  Value *cp_src, *cp_dst, *cp_mix;
  Blend mfac, temp;
  Color col_mix(0, 0, 0, 0);

  if (fac == 0) {
    return col_src;
  }

  mfac = Traits::range - fac;

  cp_src = (Value *)&col_src;
  cp_dst = (Value *)&col_dst;
  cp_mix = (Value *)&col_mix;

  temp = abs(cp_src[0] - cp_dst[0]);
  cp_mix[0] = (mfac * cp_src[0] + temp * fac) / Traits::range;
  temp = abs(cp_src[1] - cp_dst[1]);
  cp_mix[1] = (mfac * cp_src[1] + temp * fac) / Traits::range;
  temp = abs(cp_src[2] - cp_dst[2]);
  cp_mix[2] = (mfac * cp_src[2] + temp * fac) / Traits::range;
  temp = abs(cp_src[3] - cp_dst[3]);
  cp_mix[3] = (mfac * cp_src[3] + temp * fac) / Traits::range;
  return col_mix;
}

template<typename Color, typename Traits>
static Color mcol_screen(Color col_src, Color col_dst, typename Traits::BlendType fac)
{
  using Value = typename Traits::ValueType;
  using Blend = typename Traits::BlendType;

  Value *cp_src, *cp_dst, *cp_mix;
  Blend mfac, temp;
  Color col_mix(0, 0, 0, 0);

  if (fac == 0) {
    return col_src;
  }

  mfac = Traits::range - fac;

  cp_src = (Value *)&col_src;
  cp_dst = (Value *)&col_dst;
  cp_mix = (Value *)&col_mix;

  temp = Traits::max(Traits::range - (((Traits::range - cp_src[0]) * (Traits::range - cp_dst[0])) /
                                      Traits::range),
                     0);
  cp_mix[0] = (mfac * cp_src[0] + temp * fac) / Traits::range;
  temp = Traits::max(Traits::range - (((Traits::range - cp_src[1]) * (Traits::range - cp_dst[1])) /
                                      Traits::range),
                     0);
  cp_mix[1] = (mfac * cp_src[1] + temp * fac) / Traits::range;
  temp = Traits::max(Traits::range - (((Traits::range - cp_src[2]) * (Traits::range - cp_dst[2])) /
                                      Traits::range),
                     0);
  cp_mix[2] = (mfac * cp_src[2] + temp * fac) / Traits::range;
  temp = Traits::max(Traits::range - (((Traits::range - cp_src[3]) * (Traits::range - cp_dst[3])) /
                                      Traits::range),
                     0);
  cp_mix[3] = (mfac * cp_src[3] + temp * fac) / Traits::range;
  return col_mix;
}

template<typename Color, typename Traits>
static Color mcol_hardlight(Color col_src, Color col_dst, typename Traits::BlendType fac)
{
  using Value = typename Traits::ValueType;
  using Blend = typename Traits::BlendType;

  Value *cp_src, *cp_dst, *cp_mix;
  Blend mfac, temp;
  Color col_mix(0, 0, 0, 0);

  if (fac == 0) {
    return col_src;
  }

  mfac = Traits::range - fac;

  cp_src = (Value *)&col_src;
  cp_dst = (Value *)&col_dst;
  cp_mix = (Value *)&col_mix;

  int i = 0;

  for (i = 0; i < 4; i++) {
    if (cp_dst[i] > (Traits::range / 2)) {
      temp = Traits::range - ((Traits::range - 2 * (cp_dst[i] - (Traits::range / 2))) *
                              (Traits::range - cp_src[i]) / Traits::range);
    }
    else {
      temp = (2 * cp_dst[i] * cp_src[i]) / Traits::expandedRange;
    }
    cp_mix[i] = Traits::min((mfac * cp_src[i] + temp * fac) / Traits::range, Traits::range);
  }
  return col_mix;
}

template<typename Color, typename Traits>
static Color mcol_overlay(Color col_src, Color col_dst, typename Traits::BlendType fac)
{
  using Value = typename Traits::ValueType;
  using Blend = typename Traits::BlendType;

  Value *cp_src, *cp_dst, *cp_mix;
  Blend mfac, temp;
  Color col_mix(0, 0, 0, 0);

  if (fac == 0) {
    return col_src;
  }

  mfac = Traits::range - fac;

  cp_src = (Value *)&col_src;
  cp_dst = (Value *)&col_dst;
  cp_mix = (Value *)&col_mix;

  int i = 0;

  for (i = 0; i < 4; i++) {
    if (cp_src[i] > (Traits::range / 2)) {
      temp = Traits::range - ((Traits::range - 2 * (cp_src[i] - (Traits::range / 2))) *
                              (Traits::range - cp_dst[i]) / Traits::range);
    }
    else {
      temp = (2 * cp_dst[i] * cp_src[i]) / Traits::expandedRange;
    }
    cp_mix[i] = Traits::min((mfac * cp_src[i] + temp * fac) / Traits::range, Traits::range);
  }
  return col_mix;
}

template<typename Color, typename Traits>
static Color mcol_softlight(Color col_src, Color col_dst, typename Traits::BlendType fac)
{
  using Value = typename Traits::ValueType;
  using Blend = typename Traits::BlendType;

  Value *cp_src, *cp_dst, *cp_mix;
  Blend mfac, temp;
  Color col_mix(0, 0, 0, 0);

  if (fac == 0) {
    return col_src;
  }

  mfac = Traits::range - fac;

  cp_src = (Value *)&col_src;
  cp_dst = (Value *)&col_dst;
  cp_mix = (Value *)&col_mix;

  /* Use divide_round so we don't alter original byte equations. */
  const int add = Traits::divide_round(Traits::range, 4);

  for (int i = 0; i < 4; i++) {
    if (cp_src[i] < (Traits::range / 2)) {
      temp = ((2 * ((cp_dst[i] / 2) + add)) * cp_src[i]) / Traits::range;
    }
    else {
      temp = Traits::range - (2 * (Traits::range - ((cp_dst[i] / 2) + add)) *
                              (Traits::range - cp_src[i]) / Traits::range);
    }
    cp_mix[i] = (temp * fac + cp_src[i] * mfac) / Traits::range;
  }
  return col_mix;
}

template<typename Color, typename Traits>
static Color mcol_exclusion(Color col_src, Color col_dst, typename Traits::BlendType fac)
{
  using Value = typename Traits::ValueType;
  using Blend = typename Traits::BlendType;

  Value *cp_src, *cp_dst, *cp_mix;
  Blend mfac, temp;
  Color col_mix(0, 0, 0, 0);

  if (fac == 0) {
    return col_src;
  }

  mfac = Traits::range - fac;

  cp_src = (Value *)&col_src;
  cp_dst = (Value *)&col_dst;
  cp_mix = (Value *)&col_mix;

  int i = 0;

  for (i = 0; i < 4; i++) {
    temp = (Traits::range / 2) -
           ((2 * (cp_src[i] - (Traits::range / 2)) * (cp_dst[i] - (Traits::range / 2))) /
            Traits::range);
    cp_mix[i] = (temp * fac + cp_src[i] * mfac) / Traits::range;
  }
  return col_mix;
}

template<typename Color, typename Traits>
static Color mcol_luminosity(Color col_src, Color col_dst, typename Traits::BlendType fac)
{
  using Value = typename Traits::ValueType;
  using Blend = typename Traits::BlendType;

  Value *cp_src, *cp_dst, *cp_mix;
  Blend mfac;
  Color col_mix(0, 0, 0, 0);

  if (fac == 0) {
    return col_src;
  }

  mfac = Traits::range - fac;

  cp_src = (Value *)&col_src;
  cp_dst = (Value *)&col_dst;
  cp_mix = (Value *)&col_mix;

  float h1, s1, v1;
  float h2, s2, v2;
  float r, g, b;
  rgb_to_hsv(cp_src[0] / Traits::frange,
             cp_src[1] / Traits::frange,
             cp_src[2] / Traits::frange,
             &h1,
             &s1,
             &v1);
  rgb_to_hsv(cp_dst[0] / Traits::frange,
             cp_dst[1] / Traits::frange,
             cp_dst[2] / Traits::frange,
             &h2,
             &s2,
             &v2);

  v1 = v2;

  hsv_to_rgb(h1, s1, v1, &r, &g, &b);

  cp_mix[0] = ((Blend)(r * Traits::frange) * fac + mfac * cp_src[0]) / Traits::range;
  cp_mix[1] = ((Blend)(g * Traits::frange) * fac + mfac * cp_src[1]) / Traits::range;
  cp_mix[2] = ((Blend)(b * Traits::frange) * fac + mfac * cp_src[2]) / Traits::range;
  cp_mix[3] = ((Blend)(cp_dst[3]) * fac + mfac * cp_src[3]) / Traits::range;
  return col_mix;
}

template<typename Color, typename Traits>
static Color mcol_saturation(Color col_src, Color col_dst, typename Traits::BlendType fac)
{
  using Value = typename Traits::ValueType;
  using Blend = typename Traits::BlendType;

  Value *cp_src, *cp_dst, *cp_mix;
  Blend mfac;
  Color col_mix(0, 0, 0, 0);

  if (fac == 0) {
    return col_src;
  }

  mfac = Traits::range - fac;

  cp_src = (Value *)&col_src;
  cp_dst = (Value *)&col_dst;
  cp_mix = (Value *)&col_mix;

  float h1, s1, v1;
  float h2, s2, v2;
  float r, g, b;
  rgb_to_hsv(cp_src[0] / Traits::frange,
             cp_src[1] / Traits::frange,
             cp_src[2] / Traits::frange,
             &h1,
             &s1,
             &v1);
  rgb_to_hsv(cp_dst[0] / Traits::frange,
             cp_dst[1] / Traits::frange,
             cp_dst[2] / Traits::frange,
             &h2,
             &s2,
             &v2);

  if (s1 > EPS_SATURATION) {
    s1 = s2;
  }

  hsv_to_rgb(h1, s1, v1, &r, &g, &b);

  cp_mix[0] = ((Blend)(r * Traits::frange) * fac + mfac * cp_src[0]) / Traits::range;
  cp_mix[1] = ((Blend)(g * Traits::frange) * fac + mfac * cp_src[1]) / Traits::range;
  cp_mix[2] = ((Blend)(b * Traits::frange) * fac + mfac * cp_src[2]) / Traits::range;
  return col_mix;
}

template<typename Color, typename Traits>
static Color mcol_hue(Color col_src, Color col_dst, typename Traits::BlendType fac)
{
  using Value = typename Traits::ValueType;
  using Blend = typename Traits::BlendType;

  Value *cp_src, *cp_dst, *cp_mix;
  Blend mfac;
  Color col_mix(0, 0, 0, 0);

  if (fac == 0) {
    return col_src;
  }

  mfac = Traits::range - fac;

  cp_src = (Value *)&col_src;
  cp_dst = (Value *)&col_dst;
  cp_mix = (Value *)&col_mix;

  float h1, s1, v1;
  float h2, s2, v2;
  float r, g, b;
  rgb_to_hsv(cp_src[0] / Traits::frange,
             cp_src[1] / Traits::frange,
             cp_src[2] / Traits::frange,
             &h1,
             &s1,
             &v1);
  rgb_to_hsv(cp_dst[0] / Traits::frange,
             cp_dst[1] / Traits::frange,
             cp_dst[2] / Traits::frange,
             &h2,
             &s2,
             &v2);

  h1 = h2;

  hsv_to_rgb(h1, s1, v1, &r, &g, &b);

  cp_mix[0] = ((Blend)(r * Traits::frange) * fac + mfac * cp_src[0]) / Traits::range;
  cp_mix[1] = ((Blend)(g * Traits::frange) * fac + mfac * cp_src[1]) / Traits::range;
  cp_mix[2] = ((Blend)(b * Traits::frange) * fac + mfac * cp_src[2]) / Traits::range;
  cp_mix[3] = ((Blend)(cp_dst[3]) * fac + mfac * cp_src[3]) / Traits::range;
  return col_mix;
}

template<typename Color, typename Traits>
static Color mcol_alpha_add(Color col_src, typename Traits::BlendType fac)
{
  using Value = typename Traits::ValueType;
  using Blend = typename Traits::BlendType;

  Value *cp_src, *cp_mix;
  Blend temp;
  Color col_mix = col_src;

  if (fac == 0) {
    return col_src;
  }

  cp_src = (Value *)&col_src;
  cp_mix = (Value *)&col_mix;

  temp = cp_src[3] + fac;
  cp_mix[3] = (temp > Traits::cmpRange) ? Traits::range : temp;

  return col_mix;
}

template<typename Color, typename Traits>
static Color mcol_alpha_sub(Color col_src, typename Traits::BlendType fac)
{
  using Value = typename Traits::ValueType;
  using Blend = typename Traits::BlendType;

  Value *cp_src, *cp_mix;
  Blend temp;
  Color col_mix = col_src;

  if (fac == 0) {
    return col_src;
  }

  cp_src = (Value *)&col_src;
  cp_mix = (Value *)&col_mix;

  temp = cp_src[3] - fac;
  cp_mix[3] = temp < 0 ? 0 : temp;

  return col_mix;
}

template<typename Color, typename Traits>
static Color mcol_pinlight(Color col_src, Color col_dst, typename Traits::BlendType fac)
{
  using Value = typename Traits::ValueType;
  using Blend = typename Traits::BlendType;

  Value *cp_src, *cp_dst, *cp_mix;
  Blend mfac;
  Color col_mix(0, 0, 0, 0);

  if (fac == 0) {
    return col_src;
  }

  mfac = Traits::range - fac;

  cp_src = (Value *)&col_src;
  cp_dst = (Value *)&col_dst;
  cp_mix = (Value *)&col_mix;

  const Blend cmp = Traits::range / 2;

  int i = 3;
  Blend temp;

  while (i--) {
    if (cp_dst[i] > cmp) {
      temp = Traits::max(2 * (cp_dst[i] - cmp), cp_src[i]);
    }
    else {
      temp = Traits::min(2 * cp_dst[i], cp_src[i]);
    }
    cp_mix[i] = (Value)((Traits::min(temp, Traits::range) * fac + cp_src[i] * mfac) /
                        Traits::range);
  }

  col_mix.a = col_src.a;
  return col_mix;
}

template<typename Color, typename Traits>
static Color mcol_linearlight(Color col_src, Color col_dst, typename Traits::BlendType fac)
{
  using Value = typename Traits::ValueType;
  using Blend = typename Traits::BlendType;

  Value *cp_src, *cp_dst, *cp_mix;
  Blend mfac;
  Color col_mix(0, 0, 0, 0);

  if (fac == 0) {
    return col_src;
  }

  mfac = Traits::range - fac;

  cp_src = (Value *)&col_src;
  cp_dst = (Value *)&col_dst;
  cp_mix = (Value *)&col_mix;

  const Blend cmp = Traits::range / 2;

  int i = 3;
  while (i--) {
    Blend temp;

    if (cp_dst[i] > cmp) {
      temp = Traits::min(cp_src[i] + 2 * (cp_dst[i] - cmp), Traits::range);
    }
    else {
      temp = Traits::max(cp_src[i] + 2 * cp_dst[i] - Traits::range, 0);
    }

    cp_mix[i] = (Value)((temp * fac + cp_src[i] * mfac) / Traits::range);
  }

  col_mix.a = col_src.a;
  return col_mix;
}

template<typename Color, typename Traits>
static Color mcol_vividlight(Color col_src, Color col_dst, typename Traits::BlendType fac)
{
  using Value = typename Traits::ValueType;
  using Blend = typename Traits::BlendType;

  Value *cp_src, *cp_dst, *cp_mix;
  Blend mfac;
  Color col_mix(0, 0, 0, 0);

  if (fac == 0) {
    return col_src;
  }

  mfac = Traits::range - fac;

  cp_src = (Value *)&col_src;
  cp_dst = (Value *)&col_dst;
  cp_mix = (Value *)&col_mix;

  const Blend cmp = Traits::range / 2;

  int i = 3;

  while (i--) {
    Blend temp;

    if (cp_dst[i] == Traits::range) {
      temp = (cp_src[i] == 0) ? cmp : Traits::range;
    }
    else if (cp_dst[i] == 0) {
      temp = (cp_src[i] == Traits::range) ? cmp : 0;
    }
    else if (cp_dst[i] > cmp) {
      temp = Traits::min(((cp_src[i]) * Traits::range) / (2 * (Traits::range - cp_dst[i])),
                         Traits::range);
    }
    else {
      temp = Traits::max(
          Traits::range - ((Traits::range - cp_src[i]) * Traits::range / (2 * cp_dst[i])), 0);
    }
    col_mix[i] = (Value)((temp * fac + cp_src[i] * mfac) / Traits::range);
  }

  col_mix.a = col_src.a;
  return col_mix;
}

template<typename Color, typename Traits>
static Color mcol_color(Color col_src, Color col_dst, typename Traits::BlendType fac)
{
  using Value = typename Traits::ValueType;
  using Blend = typename Traits::BlendType;

  Value *cp_src, *cp_dst, *cp_mix;
  Blend mfac;
  Color col_mix(0, 0, 0, 0);

  if (fac == 0) {
    return col_src;
  }

  mfac = Traits::range - fac;

  cp_src = (Value *)&col_src;
  cp_dst = (Value *)&col_dst;
  cp_mix = (Value *)&col_mix;

  float h1, s1, v1;
  float h2, s2, v2;
  float r, g, b;

  rgb_to_hsv(cp_src[0] / Traits::frange,
             cp_src[1] / Traits::frange,
             cp_src[2] / Traits::frange,
             &h1,
             &s1,
             &v1);
  rgb_to_hsv(cp_dst[0] / Traits::frange,
             cp_dst[1] / Traits::frange,
             cp_dst[2] / Traits::frange,
             &h2,
             &s2,
             &v2);

  h1 = h2;
  s1 = s2;

  hsv_to_rgb(h1, s1, v1, &r, &g, &b);

  cp_mix[0] = (Value)(((Blend)(r * Traits::frange) * fac + cp_src[0] * mfac) / Traits::range);
  cp_mix[1] = (Value)(((Blend)(g * Traits::frange) * fac + cp_src[1] * mfac) / Traits::range);
  cp_mix[2] = (Value)(((Blend)(b * Traits::frange) * fac + cp_src[2] * mfac) / Traits::range);

  col_mix.a = col_src.a;
  return col_mix;
}

template<typename Color, typename Traits>
static Color mcol_colorburn(Color col_src, Color col_dst, typename Traits::BlendType fac)
{
  using Value = typename Traits::ValueType;
  using Blend = typename Traits::BlendType;

  Value *cp_src, *cp_dst, *cp_mix;
  Blend mfac;
  Color col_mix(0, 0, 0, 0);

  if (fac == 0) {
    return col_src;
  }

  mfac = Traits::range - fac;

  cp_src = (Value *)&col_src;
  cp_dst = (Value *)&col_dst;
  cp_mix = (Value *)&col_mix;

  int i = 3;

  while (i--) {
    const Blend temp =
        (cp_dst[i] == 0) ?
            0 :
            Traits::max(Traits::range - ((Traits::range - cp_src[i]) * Traits::range) / cp_dst[i],
                        0);
    cp_mix[i] = (Value)((temp * fac + cp_src[i] * mfac) / Traits::range);
  }

  col_mix.a = col_src.a;
  return col_mix;
}

template<typename Color, typename Traits>
static Color mcol_linearburn(Color col_src, Color col_dst, typename Traits::BlendType fac)
{
  using Value = typename Traits::ValueType;
  using Blend = typename Traits::BlendType;

  Value *cp_src, *cp_dst, *cp_mix;
  Blend mfac;
  Color col_mix(0, 0, 0, 0);

  if (fac == 0) {
    return col_src;
  }

  mfac = Traits::range - fac;

  cp_src = (Value *)&col_src;
  cp_dst = (Value *)&col_dst;
  cp_mix = (Value *)&col_mix;

  int i = 3;

  while (i--) {
    const Blend temp = Traits::max(cp_src[i] + cp_dst[i] - Traits::range, 0);
    cp_mix[i] = (Value)((temp * fac + cp_src[i] * mfac) / Traits::range);
  }

  col_mix.a = col_src.a;
  return col_mix;
}

template<typename Color, typename Traits>
BLI_INLINE Color ED_vpaint_blend_tool2(const IMB_BlendMode tool,
                                       const Color col,
                                       const Color paintcol,
                                       const typename Traits::BlendType alpha_i)
{
  switch ((IMB_BlendMode)tool) {
    case IMB_BLEND_MIX:
      return mcol_blend<Color, Traits>(col, paintcol, alpha_i);
    case IMB_BLEND_ADD:
      return mcol_add<Color, Traits>(col, paintcol, alpha_i);
    case IMB_BLEND_SUB:
      return mcol_sub<Color, Traits>(col, paintcol, alpha_i);
    case IMB_BLEND_MUL:
      return mcol_mul<Color, Traits>(col, paintcol, alpha_i);
    case IMB_BLEND_LIGHTEN:
      return mcol_lighten<Color, Traits>(col, paintcol, alpha_i);
    case IMB_BLEND_DARKEN:
      return mcol_darken<Color, Traits>(col, paintcol, alpha_i);
    case IMB_BLEND_COLORDODGE:
      return mcol_colordodge<Color, Traits>(col, paintcol, alpha_i);
    case IMB_BLEND_COLORBURN:
      return mcol_colorburn<Color, Traits>(col, paintcol, alpha_i);
    case IMB_BLEND_DIFFERENCE:
      return mcol_difference<Color, Traits>(col, paintcol, alpha_i);
    case IMB_BLEND_SCREEN:
      return mcol_screen<Color, Traits>(col, paintcol, alpha_i);
    case IMB_BLEND_HARDLIGHT:
      return mcol_hardlight<Color, Traits>(col, paintcol, alpha_i);
    case IMB_BLEND_OVERLAY:
      return mcol_overlay<Color, Traits>(col, paintcol, alpha_i);
    case IMB_BLEND_SOFTLIGHT:
      return mcol_softlight<Color, Traits>(col, paintcol, alpha_i);
    case IMB_BLEND_EXCLUSION:
      return mcol_exclusion<Color, Traits>(col, paintcol, alpha_i);
    case IMB_BLEND_LUMINOSITY:
      return mcol_luminosity<Color, Traits>(col, paintcol, alpha_i);
    case IMB_BLEND_SATURATION:
      return mcol_saturation<Color, Traits>(col, paintcol, alpha_i);
    case IMB_BLEND_HUE:
      return mcol_hue<Color, Traits>(col, paintcol, alpha_i);
    /* non-color */
    case IMB_BLEND_ERASE_ALPHA:
      return mcol_alpha_sub<Color, Traits>(col, alpha_i);
    case IMB_BLEND_ADD_ALPHA:
      return mcol_alpha_add<Color, Traits>(col, alpha_i);
    case IMB_BLEND_PINLIGHT:
      return mcol_pinlight<Color, Traits>(col, paintcol, alpha_i);
    case IMB_BLEND_LINEARLIGHT:
      return mcol_linearlight<Color, Traits>(col, paintcol, alpha_i);
    case IMB_BLEND_VIVIDLIGHT:
      return mcol_vividlight<Color, Traits>(col, paintcol, alpha_i);
    case IMB_BLEND_COLOR:
      return mcol_color<Color, Traits>(col, paintcol, alpha_i);
    default:
      BLI_assert(0);
      return Color(0, 0, 0, 0);
  }
}
/** \} */

}  // namespace paint
}  // namespace blender
