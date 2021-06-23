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
 * The Original Code is Copyright (C) 2021 by Blender Foundation.
 * All rights reserved.
 */

#pragma once

#include <stdint.h>
#include <string>
#include <vector>

#ifdef WITH_CXX_GUARDEDALLOC
#  include "MEM_guardedalloc.h"
#endif

namespace shader_compiler {

enum class SourceType {
  GlslVertexShader,
  GlslGeometryShader,
  GlslFragmentShader,
  GlslComputeShader,
};

enum class TargetType {
  SpirV,
  // SpirVAssembly,
};

enum class OptimizationLevel {
  NotOptimized,
  SizeOptimized,
  SpeedOptimized,
};

enum class StatusCode {
  Ok,
  CompilationError,
};

class Job;
class Result;

class Compiler {
 protected:
  Compiler(){};

 public:
  virtual ~Compiler(){};

  static Compiler *create_default();
  virtual Result *compile(const Job &job) = 0;

#ifdef WITH_CXX_GUARDEDALLOC
  MEM_CXX_CLASS_ALLOC_FUNCS("ShadeCompiler:Compiler")
#endif
};

class Job {
 public:
  const char *name = nullptr;
  const char *source = nullptr;
  SourceType source_type;
  TargetType compilation_target;
  OptimizationLevel optimization_level = OptimizationLevel::NotOptimized;

#ifdef WITH_CXX_GUARDEDALLOC
  MEM_CXX_CLASS_ALLOC_FUNCS("ShaderCompiler:Job")
#endif
};

class Result {
 public:
  TargetType type;
  StatusCode status_code;
  std::string error_log;
  std::vector<uint32_t> bin;

  virtual ~Result(){};

 protected:
  void init(const Job &job);

#ifdef WITH_CXX_GUARDEDALLOC
  MEM_CXX_CLASS_ALLOC_FUNCS("ShaderCompiler:Result")
#endif
};

void init();
void free();

}  // namespace shader_compiler
