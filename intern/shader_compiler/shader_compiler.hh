
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
