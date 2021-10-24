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
 * Copyright 2021, Blender Foundation.
 */

/** \file
 * \ingroup eevee
 *
 * Module containing passes and parameters used for raytracing.
 * NOTE: For now only screen space raytracing is supported.
 */

#include <fstream>
#include <iostream>

#include "eevee_instance.hh"

#include "eevee_raytracing.hh"

namespace blender::eevee {

/* -------------------------------------------------------------------- */
/** \name Raytracing
 *
 * \{ */

void RaytracingModule::sync(void)
{
  SceneEEVEE &sce_eevee = inst_.scene->eevee;

  reflection_data_.thickness = sce_eevee.ssr_thickness;
  reflection_data_.brightness_clamp = (sce_eevee.ssr_firefly_fac < 1e-8f) ?
                                          FLT_MAX :
                                          sce_eevee.ssr_firefly_fac;
  reflection_data_.max_roughness = sce_eevee.ssr_max_roughness;
  reflection_data_.quality = 1.0f - 0.95f * sce_eevee.ssr_quality;
  reflection_data_.bias = 0.1f + reflection_data_.quality * 0.6f;
  reflection_data_.pool_offset = inst_.sampling.sample_get() / 5;

  refraction_data_ = static_cast<RaytraceData>(reflection_data_);
  // refraction_data_.thickness = 1e16;
  /* TODO(fclem): Clamp option for refraction. */
  /* TODO(fclem): bias option for refraction. */
  /* TODO(fclem): bias option for refraction. */

  reflection_data_.push_update();
  refraction_data_.push_update();

  enabled_ = (sce_eevee.flag & SCE_EEVEE_SSR_ENABLED) != 0;
}

void RaytracingModule::generate_sample_reuse_table(void)
{
  /** Following "Stochastic All The Things: Raytracing in Hybrid Real-Time Rendering"
   * by Tomasz Stachowiak
   * https://www.ea.com/seed/news/seed-dd18-presentation-slides-raytracing
   */

  const int samples_per_pool = 16;
  std::array<Vector<ivec2>, 4> pools;
  std::array<vec3, 4> pools_color = {vec3{1, 0, 0}, vec3{0, 1, 0}, vec3{0, 0, 1}, vec3{1, 1, 0}};
  Vector<vec3> debug_image(64 * 64);

  std::ofstream ppm;
  auto ppm_file_out = [&](const char *name, Vector<vec3> &debug_image) {
    ppm.open(name);
    ppm << "P3\n64 64\n255\n";
    for (auto &vec : debug_image) {
      uchar ucol[3];
      rgb_float_to_uchar(ucol, vec);
      ppm << (int)ucol[0] << " " << (int)ucol[1] << " " << (int)ucol[2] << "\n";
      /* Clear the image. */
      vec = vec3(0.0f);
    }
    ppm.close();
  };

  /* Using sample_reuse_1.ppm, set a center position where there is 4 different pool (color). */
  ivec2 center[4] = {ivec2(49, 47), ivec2(48, 46), ivec2(49, 46), ivec2(48, 47)};
  /* Remapping of pool order since the center samples may not match the pool order from shader.
   * Order is (0,0), (1,0), (0,1), (1,1).
   * IMPORTANT: the actual PPM picture has Y flipped compared to OpenGL. */
  std::array<int, 4> poolmap = {3, 0, 1, 2};

  for (auto x : IndexRange(64)) {
    for (auto y : IndexRange(64)) {
      uint px = y * 64 + x;
      float noise_value = blue_noise[px][0];
      int pool_id = floor(noise_value * 4.0f);
      ivec2 ofs = ivec2(x, y) - center[pool_id];
      pools[pool_id].append(ofs);

      debug_image[px] = pools_color[pool_id];
    }
  }
  ppm_file_out("sample_reuse_1.ppm", debug_image);

  for (auto pool_id : IndexRange(4)) {
    auto &pool = pools[pool_id];
    auto sort = [](const ivec2 &a, const ivec2 &b) {
      return len_manhattan_v2_int(a) < len_manhattan_v2_int(b);
    };
    std::sort(pool.begin(), pool.end(), sort);

    for (auto j : IndexRange(16)) {
      ivec2 pos = pool[j] + center[pool_id];
      uint px = pos.y * 64 + pos.x;
      debug_image[px] = pools_color[pool_id];
    }
  }
  ppm_file_out("sample_reuse_2.ppm", debug_image);

  /* TODO(fclem): Order the samples to have better spatial coherence. */

  std::ofstream table_out;
  table_out.open("sample_table.glsl");
  table_out << "const int resolve_sample_max = " << samples_per_pool << ";\n";
  table_out << "const vec2 resolve_sample_offsets[" << samples_per_pool * 4 << "] = vec2["
            << samples_per_pool * 4 << "](";
  for (auto pool_id : IndexRange(4)) {
    auto &pool = pools[poolmap[pool_id]];
    for (auto i : IndexRange(samples_per_pool)) {
      table_out << "    vec2(" << pool[i].x << ", " << pool[i].y << ")";
      if (i < samples_per_pool - 1) {
        table_out << ",\n";
      }
    }
    if (pool_id < 3) {
      table_out << ",\n";
    }
  }
  table_out << ");\n";
  table_out.close();
}

/** \} */

}  // namespace blender::eevee
