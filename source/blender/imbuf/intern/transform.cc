/*
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * The Original Code is Copyright (C) 2001-2002 by NaN Holding BV.
 * All rights reserved.
 */

/** \file
 * \ingroup imbuf
 */

#include <type_traits>

#include "BLI_math.h"
#include "BLI_rect.h"

#include "IMB_imbuf.h"
#include "IMB_imbuf_types.h"

namespace blender::imbuf::transform {

struct TransformUserData {
  const ImBuf *src;
  ImBuf *dst;
  float start_uv[2];
  float add_x[2];
  float add_y[2];
  rctf src_crop;

  void init(const float transform_matrix[4][4])
  {
    init_start_uv(transform_matrix);
    init_add_x(transform_matrix);
    init_add_y(transform_matrix);
  }

 private:
  void init_start_uv(const float transform_matrix[4][4])
  {
    float start_uv_v3[3];
    float orig[3];
    zero_v3(orig);
    mul_v3_m4v3(start_uv_v3, transform_matrix, orig);
    copy_v2_v2(start_uv, start_uv_v3);
  }

  void init_add_x(const float transform_matrix[4][4])
  {
    const int width = src->x;
    float add_x_v3[3];
    float uv_max_x[3];
    zero_v3(uv_max_x);
    uv_max_x[0] = width;
    uv_max_x[1] = 0.0f;
    mul_v3_m4v3(add_x_v3, transform_matrix, uv_max_x);
    sub_v2_v2(add_x_v3, start_uv);
    mul_v2_fl(add_x_v3, 1.0f / width);
    copy_v2_v2(add_x, add_x_v3);
  }

  void init_add_y(const float transform_matrix[4][4])
  {
    const int height = src->y;
    float add_y_v3[3];
    float uv_max_y[3];
    zero_v3(uv_max_y);
    uv_max_y[0] = 0.0f;
    uv_max_y[1] = height;
    mul_v3_m4v3(add_y_v3, transform_matrix, uv_max_y);
    sub_v2_v2(add_y_v3, start_uv);
    mul_v2_fl(add_y_v3, 1.0f / height);
    copy_v2_v2(add_y, add_y_v3);
  }
};

/**
 * \brief Base class for source discarding.
 *
 * The class decides if a specific uv coordinate from the source buffer should be ignored.
 * This is used to mix multiple images over a single output buffer. Discarded pixels will
 * not change the output buffer.
 */
class BaseDiscard {
 public:
  virtual ~BaseDiscard() = default;

  /**
   * \brief Should the source pixel at the given uv coordinate be discarded.
   */
  virtual bool should_discard(const TransformUserData &user_data, const float uv[2]) = 0;
};

/**
 * \brief Crop uv-coordinates that are outside the user data src_crop rect.
 */
class CropSource : public BaseDiscard {
 public:
  /**
   * \brief Should the source pixel at the given uv coordinate be discarded.
   *
   * Uses user_data.src_crop to determine if the uv coordinate should be skipped.
   */
  virtual bool should_discard(const TransformUserData &user_data, const float uv[2])
  {
    return uv[0] < user_data.src_crop.xmin && uv[0] >= user_data.src_crop.xmax &&
           uv[1] < user_data.src_crop.ymin && uv[1] >= user_data.src_crop.ymax;
  }
};

/**
 * \brief Discard that does not discard anything.
 */
class NoDiscard : public BaseDiscard {
 public:
  /**
   * \brief Should the source pixel at the given uv coordinate be discarded.
   *
   * Will never discard any pixels.
   */
  virtual bool should_discard(const TransformUserData &UNUSED(user_data),
                              const float UNUSED(uv[2]))
  {
    return false;
  }
};

/**
 * \brief pointer to a texel to write or read serial.
 */
template<
    /**
     * \brief Kind of buffer.
     * Possible options: float, unsigned char.
     */
    typename StorageType = float,

    /**
     * \brief Number of channels of a single pixel.
     */
    int NumChannels = 4>
class TexelPointer {
  StorageType *pointer;

 public:
  void init_pixel_pointer(const ImBuf *image_buffer, int x, int y)
  {
    const size_t offset = (y * (size_t)image_buffer->x + x) * NumChannels;

    if constexpr (std::is_same_v<StorageType, float>) {
      pointer = image_buffer->rect_float + offset;
    }
    else if constexpr (std::is_same_v<StorageType, unsigned char>) {
      pointer = const_cast<unsigned char *>(
          static_cast<const unsigned char *>(static_cast<const void *>(image_buffer->rect)) +
          offset);
    }
    else {
      pointer = nullptr;
    }
  }
  /*
    float *get_float_pointer()
    {
      if constexpr (std::is_same_v<StorageType, float>) {
        return pointer;
      }
      else {
        return nullptr;
      }
    }
    unsigned char *get_uchar_pointer()
    {
      if constexpr (std::is_same_v<StorageType, unsigned char>) {
        return pointer;
      }
      else {
        return nullptr;
      }
    }
    */

  /**
   * \brief Get pointer to the current texel to write to.
   */
  StorageType *get_pointer()
  {
    return pointer;
  }

  void increase_pixel_pointer()
  {
    pointer += NumChannels;
  }
};

template<eIMBInterpolationFilterMode Filter, typename StorageType, int NumChannels> class Sampler {
 public:
  virtual void sample(const ImBuf *source,
                      const float u,
                      const float v,
                      StorageType r_sample[NumChannels])
  {
    if constexpr (Filter == IMB_FILTER_NEAREST && std::is_same_v<StorageType, float> &&
                  NumChannels == 4) {
      nearest_interpolation_color_fl(source, nullptr, r_sample, u, v);
    }
    else if constexpr (Filter == IMB_FILTER_BILINEAR && std::is_same_v<StorageType, float> &&
                       NumChannels == 4) {
      bilinear_interpolation_color_fl(source, nullptr, r_sample, u, v);
    }
    else if constexpr (Filter == IMB_FILTER_NEAREST &&
                       std::is_same_v<StorageType, unsigned char> && NumChannels == 4) {
      nearest_interpolation_color_char(source, r_sample, nullptr, u, v);
    }
    else if constexpr (Filter == IMB_FILTER_BILINEAR &&
                       std::is_same_v<StorageType, unsigned char> && NumChannels == 4) {
      bilinear_interpolation_color_char(source, r_sample, nullptr, u, v);
    }
    else {
      /* Unsupported sampler. */
      BLI_assert_unreachable();
    }
  }
};

/**
 * \brief Wrapping mode for the uv coordinates.
 *
 * Subclasses have the ability to change the UV coordinates before the source buffer will be
 * sampled.
 */
class BaseUVWrapping {
 public:
  /**
   * \brief modify the given u coordinate.
   */
  virtual float modify_u(const TransformUserData &user_data, float u) = 0;

  /**
   * \brief modify the given v coordinate.
   */
  virtual float modify_v(const TransformUserData &user_data, float v) = 0;
};

/**
 * \brief UVWrapping method that does not modify the UV coordinates.
 */
class PassThroughUV : public BaseUVWrapping {
 public:
  float modify_u(const TransformUserData &UNUSED(user_data), float u) override
  {
    return u;
  }

  float modify_v(const TransformUserData &UNUSED(user_data), float v) override
  {
    return v;
  }
};

/**
 * \brief UVWrapping method that wrap repeats the UV coordinates.
 */
class WrapRepeatUV : public BaseUVWrapping {
 public:
  float modify_u(const TransformUserData &user_data, float u) override

  {
    int x = (int)floor(u);
    x = x % user_data.src->x;
    if (x < 0) {
      x += user_data.src->x;
    }
    return x;
  }

  float modify_v(const TransformUserData &user_data, float v) override
  {
    int y = (int)floor(v);
    y = y % user_data.src->y;
    if (y < 0) {
      y += user_data.src->y;
    }
    return y;
  }
};

template<
    /**
     * \brief Discard function to use.
     *
     * \attention Should be a subclass of BaseDiscard.
     */
    typename Discard,

    /**
     * \brief Color interpolation function to read from the source buffer.
     */
    typename Sampler,

    /**
     * \brief Kernel to store to the destination buffer.
     * Should be an TexelPointer
     */
    typename OutputTexelPointer,

    /**
     * \brief Wrapping method to perform
     * Should be a subclass of BaseUVWrapper
     */
    typename UVWrapping>
class ScanlineProcessor {
  Discard discarder;
  UVWrapping uv_wrapping;
  OutputTexelPointer output;
  Sampler sampler;

 public:
  void process(const TransformUserData *user_data, int scanline)
  {
    const int width = user_data->dst->x;

    float uv[2];
    madd_v2_v2v2fl(uv, user_data->start_uv, user_data->add_y, scanline);

    output.init_pixel_pointer(user_data->dst, 0, scanline);
    for (int xi = 0; xi < width; xi++) {
      if (!discarder.should_discard(*user_data, uv)) {

        sampler.sample(user_data->src,
                       uv_wrapping.modify_u(*user_data, uv[0]),
                       uv_wrapping.modify_v(*user_data, uv[1]),
                       output.get_pointer());
      }

      add_v2_v2(uv, user_data->add_x);
      output.increase_pixel_pointer();
    }
  }
};

template<typename Processor> void transform_scanline_function(void *custom_data, int scanline)
{
  const TransformUserData *user_data = static_cast<const TransformUserData *>(custom_data);
  Processor processor;
  processor.process(user_data, scanline);
}

template<eIMBInterpolationFilterMode Filter, typename StorageType, int NumChannels>
ScanlineThreadFunc get_scanline_function(const eIMBTransformMode mode)

{
  switch (mode) {
    case IMB_TRANSFORM_MODE_REGULAR:
      return transform_scanline_function<
          ScanlineProcessor<NoDiscard,
                            Sampler<Filter, StorageType, NumChannels>,
                            TexelPointer<StorageType, NumChannels>,
                            PassThroughUV>>;
    case IMB_TRANSFORM_MODE_CROP_SRC:
      return transform_scanline_function<
          ScanlineProcessor<CropSource,
                            Sampler<Filter, StorageType, NumChannels>,
                            TexelPointer<StorageType, NumChannels>,
                            PassThroughUV>>;
    case IMB_TRANSFORM_MODE_WRAP_REPEAT:
      return transform_scanline_function<
          ScanlineProcessor<NoDiscard,
                            Sampler<Filter, StorageType, NumChannels>,
                            TexelPointer<StorageType, NumChannels>,
                            WrapRepeatUV>>;
  }

  BLI_assert_unreachable();
  return nullptr;
}

template<eIMBInterpolationFilterMode Filter>
static void transform(TransformUserData *user_data, const eIMBTransformMode mode)
{
  ScanlineThreadFunc scanline_func = nullptr;

  if (user_data->dst->rect_float) {
    scanline_func = get_scanline_function<Filter, float, 4>(mode);
  }
  else if (user_data->dst->rect) {
    scanline_func = get_scanline_function<Filter, unsigned char, 4>(mode);
  }

  if (scanline_func != nullptr) {
    IMB_processor_apply_threaded_scanlines(user_data->dst->y, scanline_func, user_data);
  }
}

}  // namespace blender::imbuf::transform

extern "C" {

using namespace blender::imbuf::transform;

void IMB_transform(const struct ImBuf *src,
                   struct ImBuf *dst,
                   const eIMBTransformMode mode,
                   const eIMBInterpolationFilterMode filter,
                   const float transform_matrix[4][4],
                   const struct rctf *src_crop)
{
  BLI_assert_msg(mode != IMB_TRANSFORM_MODE_CROP_SRC || src_crop != nullptr,
                 "No source crop rect given, but crop source is requested. Or source crop rect "
                 "was given, but crop source was not requested.");

  TransformUserData user_data;
  user_data.src = src;
  user_data.dst = dst;
  if (mode == IMB_TRANSFORM_MODE_CROP_SRC) {
    user_data.src_crop = *src_crop;
  }
  user_data.init(transform_matrix);

  if (filter == IMB_FILTER_NEAREST) {
    transform<IMB_FILTER_NEAREST>(&user_data, mode);
  }
  else {
    transform<IMB_FILTER_BILINEAR>(&user_data, mode);
  }
}
}
