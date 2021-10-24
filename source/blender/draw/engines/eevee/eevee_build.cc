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
 * Compile time computation and code generation.
 */

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>

/* For blue_noise. */
#include "eevee_lut.h"

using namespace std;

typedef unsigned char uchar;
typedef float vec3[3];

const int samples_per_pool = 16;

static void raytrace_sample_reuse_table(string &output_name, bool debug)
{
  /** Following "Stochastic All The Things: Raytracing in Hybrid Real-Time Rendering"
   * by Tomasz Stachowiak
   * https://www.ea.com/seed/news/seed-dd18-presentation-slides-raytracing
   */

  struct sample {
    int x, y;
    sample(int _x, int _y) : x(_x), y(_y){};
  };

  array<vector<sample>, 4> pools;
  array<vec3, 4> pools_color = {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0};
  vector<vec3> debug_image(64 * 64);

  ofstream ppm;
  auto ppm_file_out = [&](const char *name, vector<vec3> &debug_image) {
    ppm.open(name);
    ppm << "P3\n64 64\n255\n";
    for (auto &vec : debug_image) {
      uchar ucol[3] = {(uchar)(vec[0] * 255), (uchar)(vec[1] * 255), (uchar)(vec[2] * 255)};
      ppm << (int)ucol[0] << " " << (int)ucol[1] << " " << (int)ucol[2] << "\n";
      /* Clear the image. */
      vec[0] = vec[1] = vec[2] = 0.0f;
    }
    ppm.close();
  };

  /* Using sample_reuse_1.ppm, set a center position where there is 4 different pool (color). */
  int center[4][2] = {{49, 47}, {48, 46}, {49, 46}, {48, 47}};
  /* Remapping of pool order since the center samples may not match the pool order from shader.
   * Order is (0,0), (1,0), (0,1), (1,1).
   * IMPORTANT: the actual PPM picture has Y flipped compared to OpenGL. */
  int poolmap[4] = {3, 0, 1, 2};

  for (int x = 0; x < 64; x++) {
    for (int y = 0; y < 64; y++) {
      int px = y * 64 + x;
      float noise_value = blue_noise[px][0];
      int pool_id = floorf(noise_value * 4.0f);
      sample ofs(x - center[pool_id][0], y - center[pool_id][1]);
      pools[pool_id].push_back(ofs);

      for (int i = 0; i < 3; i++) {
        debug_image[px][i] = pools_color[pool_id][i];
      }
    }
  }

  if (debug) {
    ppm_file_out("sample_reuse_1.ppm", debug_image);
  }

  for (int pool_id = 0; pool_id < 4; pool_id++) {
    auto &pool = pools[pool_id];
    auto sort = [](const sample &a, const sample &b) {
      /* Manhattan distance. */
      return abs(a.x) + abs(a.y) < abs(b.x) + abs(b.y);
    };
    std::sort(pool.begin(), pool.end(), sort);

    for (int j = 0; j < samples_per_pool; j++) {
      int pos[2] = {pool[j].x - center[pool_id][0], pool[j].y - center[pool_id][1]};
      int px = pos[1] * 64 + pos[0];
      for (int i = 0; i < 3; i++) {
        debug_image[px][i] = pools_color[pool_id][i];
      }
    }
  }

  if (debug) {
    ppm_file_out("sample_reuse_2.ppm", debug_image);
  }

  /* TODO(fclem): Order the samples to have better spatial coherence. */

  ofstream table_out;
  int total = samples_per_pool * 4;
  table_out.open(output_name);

  table_out << "\n/* Sample table generated at build time. */\n";
  table_out << "const int resolve_sample_max = " << samples_per_pool << ";\n";
  table_out << "const vec2 resolve_sample_offsets[" << total << "] = vec2[" << total << "](\n";
  for (int pool_id = 0; pool_id < 4; pool_id++) {
    auto &pool = pools[poolmap[pool_id]];
    for (int i = 0; i < samples_per_pool; i++) {
      table_out << "  vec2(" << pool[i].x << ", " << pool[i].y << ")";
      if (i < samples_per_pool - 1) {
        table_out << ",\n";
      }
    }
    if (pool_id < 3) {
      table_out << ",\n";
    }
  }
  table_out << ");\n\n";

  table_out.close();
}

int main(int argc, char **argv)
{
  if (argc != 3) {
    fprintf(stderr, "usage: eevee_build [--resolve_sample_table] output_file\n");
    return -1;
  }
  if (string(argv[1]) == "--resolve_sample_table") {
    string output_name(argv[2]);
    raytrace_sample_reuse_table(output_name, false);
  }
  return 0;
}
