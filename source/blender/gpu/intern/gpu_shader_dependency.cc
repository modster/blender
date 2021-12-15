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
 * The Original Code is Copyright (C) 2021 Blender Foundation.
 * All rights reserved.
 */

/** \file
 * \ingroup gpu
 *
 * Shader source dependency builder that make possible to support #include directive inside the
 * shader files.
 */

#include <iostream>

#include "BLI_map.hh"
#include "BLI_set.hh"
#include "BLI_string_ref.hh"

#include "gpu_shader_dependency_private.h"

extern "C" {
#define SHADER_SOURCE(datatoc, filename) extern char datatoc[];
#include "glsl_draw_source_list.h"
#include "glsl_gpu_source_list.h"
#undef SHADER_SOURCE
}

namespace blender::gpu {

using GPUSourceDictionnary = Map<StringRef, struct GPUSource *>;

struct GPUSource {
  StringRefNull filename;
  StringRefNull source;
  Set<GPUSource *> dependencies;
  bool dependencies_init = false;

  GPUSource(const char *file, const char *datatoc) : filename(file), source(datatoc){};

  void init_dependencies(const GPUSourceDictionnary &dict)
  {
    if (dependencies_init) {
      return;
    }
    dependencies_init = true;
    int64_t pos = 0;
    while (1) {
      pos = source.find("#pragma BLENDER_REQUIRE(", pos);
      if (pos == -1) {
        return;
      }
      int64_t start = source.find("(", pos) + 1;
      int64_t end = source.find(")", pos);
      if (end == -1) {
        /* TODO Use clog. */
        std::cout << "Error: " << filename << " : Malformed BLENDER_REQUIRE: Missing \")\"."
                  << std::endl;
        return;
      }
      StringRef dependency_name = source.substr(start, end - start);
      GPUSource *source = dict.lookup_default(dependency_name, nullptr);
      if (source == nullptr) {
        /* TODO Use clog. */
        std::cout << "Error: " << filename << " : Dependency not found \"" << dependency_name
                  << "\"." << std::endl;
        return;
      }
      /* Recursive. */
      source->init_dependencies(dict);

      for (auto dep : source->dependencies) {
        dependencies.add(dep);
      }
      dependencies.add(source);
      pos++;
    };
  }

  /* Returns the final string with all inlcudes done. */
  void build(std::string &str)
  {
    for (auto dep : dependencies) {
      str += dep->source;
    }
    str += source;
  }
};

}  // namespace blender::gpu

using namespace blender::gpu;

static GPUSourceDictionnary *g_sources = nullptr;

void gpu_shader_dependency_init()
{
  g_sources = new GPUSourceDictionnary();

#define SHADER_SOURCE(datatoc, filename) \
  g_sources->add_new(filename, new GPUSource(filename, datatoc));
#include "glsl_draw_source_list.h"
#include "glsl_gpu_source_list.h"
#undef SHADER_SOURCE

  for (auto value : g_sources->values()) {
    value->init_dependencies(*g_sources);
  }
}

void gpu_shader_dependency_exit()
{
  for (auto value : g_sources->values()) {
    delete value;
  }
  delete g_sources;
}

char *gpu_shader_dependency_get_resolved_source(const char *shader_source_name)
{
  GPUSource *source = g_sources->lookup(shader_source_name);
  std::string str;
  source->build(str);
  return strdup(str.c_str());
}
