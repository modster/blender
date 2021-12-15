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
 * The Original Code is Copyright (C) 2020 Blender Foundation.
 * All rights reserved.
 */

/** \file
 * \ingroup gpu
 */

#include "BKE_global.h"

#include "BLI_string.h"
#include "BLI_vector.hh"

#include "GPU_capabilities.h"
#include "GPU_platform.h"

#include "gl_backend.hh"
#include "gl_debug.hh"
#include "gl_vertex_buffer.hh"

#include "gl_shader.hh"
#include "gl_shader_interface.hh"

using namespace blender;
using namespace blender::gpu;
using namespace blender::gpu::shader;

/* -------------------------------------------------------------------- */
/** \name Creation / Destruction
 * \{ */

GLShader::GLShader(const char *name) : Shader(name)
{
#if 0 /* Would be nice to have, but for now the Deferred compilation \
       * does not have a GPUContext. */
  BLI_assert(GLContext::get() != NULL);
#endif
  shader_program_ = glCreateProgram();

  debug::object_label(GL_PROGRAM, shader_program_, name);
}

GLShader::~GLShader()
{
#if 0 /* Would be nice to have, but for now the Deferred compilation \
       * does not have a GPUContext. */
  BLI_assert(GLContext::get() != NULL);
#endif
  /* Invalid handles are silently ignored. */
  glDeleteShader(vert_shader_);
  glDeleteShader(geom_shader_);
  glDeleteShader(frag_shader_);
  glDeleteShader(compute_shader_);
  glDeleteProgram(shader_program_);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Create Info
 * \{ */

static inline std::ostream &operator<<(std::ostream &stream, const Interpolation &interp)
{
  switch (interp) {
    case SMOOTH:
      stream << "smooth";
      break;
    case FLAT:
      stream << "flat";
      break;
    case NO_PERSPECTIVE:
      stream << "no_perspective";
      break;
  }
  return stream;
}

static inline std::ostream &operator<<(std::ostream &stream, const Type &type)
{
  switch (type) {
    case FLOAT:
      stream << "float";
      break;
    case VEC2:
      stream << "vec2";
      break;
    case VEC3:
      stream << "vec3";
      break;
    case VEC4:
      stream << "vec4";
      break;
    case MAT4:
      stream << "mat4";
      break;
    case UINT:
      stream << "uint";
      break;
    case UVEC2:
      stream << "uvec2";
      break;
    case UVEC3:
      stream << "uvec3";
      break;
    case UVEC4:
      stream << "uvec4";
      break;
    case INT:
      stream << "int";
      break;
    case IVEC2:
      stream << "ivec2";
      break;
    case IVEC3:
      stream << "ivec3";
      break;
    case IVEC4:
      stream << "ivec4";
      break;
    case BOOL:
      stream << "bool";
      break;
  }
  return stream;
}

static inline void print_image_type(std::ostream &stream,
                                    const ImageType &type,
                                    const bool is_image)
{
  switch (type) {
    case INT_BUFFER:
    case INT_1D:
    case INT_1D_ARRAY:
    case INT_2D:
    case INT_2D_ARRAY:
    case INT_3D:
    case INT_CUBE:
    case INT_CUBE_ARRAY:
      stream << "i";
      break;
    case UINT_BUFFER:
    case UINT_1D:
    case UINT_1D_ARRAY:
    case UINT_2D:
    case UINT_2D_ARRAY:
    case UINT_3D:
    case UINT_CUBE:
    case UINT_CUBE_ARRAY:
      stream << "u";
      break;
    default:
      break;
  }

  if (is_image) {
    stream << "image";
  }
  else {
    stream << "sampler";
  }

  switch (type) {
    case FLOAT_BUFFER:
    case INT_BUFFER:
    case UINT_BUFFER:
      stream << "Buffer";
      break;
    case FLOAT_1D:
    case FLOAT_1D_ARRAY:
    case INT_1D:
    case INT_1D_ARRAY:
    case UINT_1D:
    case UINT_1D_ARRAY:
      stream << "1D";
      break;
    case FLOAT_2D:
    case FLOAT_2D_ARRAY:
    case INT_2D:
    case INT_2D_ARRAY:
    case UINT_2D:
    case UINT_2D_ARRAY:
    case SHADOW_2D:
    case SHADOW_2D_ARRAY:
      stream << "2D";
      break;
    case FLOAT_3D:
    case INT_3D:
    case UINT_3D:
      stream << "3D";
      break;
    case FLOAT_CUBE:
    case FLOAT_CUBE_ARRAY:
    case INT_CUBE:
    case INT_CUBE_ARRAY:
    case UINT_CUBE:
    case UINT_CUBE_ARRAY:
    case SHADOW_CUBE:
    case SHADOW_CUBE_ARRAY:
      stream << "Cube";
      break;
    default:
      break;
  }

  switch (type) {
    case FLOAT_1D_ARRAY:
    case FLOAT_2D_ARRAY:
    case FLOAT_CUBE_ARRAY:
    case INT_1D_ARRAY:
    case INT_2D_ARRAY:
    case INT_CUBE_ARRAY:
    case UINT_1D_ARRAY:
    case UINT_2D_ARRAY:
    case UINT_CUBE_ARRAY:
    case SHADOW_2D_ARRAY:
    case SHADOW_CUBE_ARRAY:
      stream << "Array";
      break;
    default:
      break;
  }

  switch (type) {
    case SHADOW_2D:
    case SHADOW_2D_ARRAY:
    case SHADOW_CUBE:
    case SHADOW_CUBE_ARRAY:
      stream << "Shadow";
      break;
    default:
      break;
  }
}

enum SamplerType {
  DUMMY_IMAGE_TYPE = 0,
};

static inline std::ostream &operator<<(std::ostream &stream, const ImageType &type)
{
  print_image_type(stream, type, true);
  return stream;
}

static inline std::ostream &operator<<(std::ostream &stream, const SamplerType &type)
{
  print_image_type(stream, *(const ImageType *)&type, false);
  return stream;
}

static inline std::ostream &operator<<(std::ostream &stream, const Qualifier &qualifiers)
{
  if (qualifiers & RESTRICT) {
    stream << "restrict";
  }
  if (qualifiers & READ_ONLY) {
    stream << "readonly";
  }
  if (qualifiers & WRITE_ONLY) {
    stream << "writeonly";
  }
  return stream;
}

static inline void print_resource(std::ostream &stream, const ShaderCreateInfo::Resource &res)
{
  stream << "layout(";

  switch (res.bind_type) {
    case ShaderCreateInfo::Resource::BindType::IMAGE:
      stream << res.image.format << ", ";
      break;
    case ShaderCreateInfo::Resource::BindType::UNIFORM_BUFFER:
      stream << "std140, ";
      break;
    case ShaderCreateInfo::Resource::BindType::STORAGE_BUFFER:
      stream << "std430, ";
      break;
    default:
      break;
  }

  stream << "binding = " << res.slot << ") ";

  switch (res.bind_type) {
    case ShaderCreateInfo::Resource::BindType::SAMPLER:
      stream << "uniform " << res.sampler.type << " " << res.sampler.name << ";\n";
      break;
    case ShaderCreateInfo::Resource::BindType::IMAGE:
      stream << "uniform " << res.image.qualifiers << " " << res.image.type << " "
             << res.image.name << ";\n";
      break;
    case ShaderCreateInfo::Resource::BindType::UNIFORM_BUFFER:
      stream << "uniform " << res.uniformbuf.name << "{ " << res.uniformbuf.type_name << " "
             << res.uniformbuf.name << "; };\n";
      break;
    case ShaderCreateInfo::Resource::BindType::STORAGE_BUFFER:
      stream << "buffer " << res.storagebuf.qualifiers << " " << res.storagebuf.name << "{ "
             << res.storagebuf.type_name << " " << res.storagebuf.name << "; };\n";
      break;
  }
}

std::string GLShader::resources_declare(const ShaderCreateInfo &info) const
{
  std::stringstream ss;

  for (const ShaderCreateInfo::Resource &res : info.batch_resources_) {
    print_resource(ss, res);
  }
  for (const ShaderCreateInfo::Resource &res : info.pass_resources_) {
    print_resource(ss, res);
  }
  for (const ShaderCreateInfo::PushConst &uniform : info.push_constants_) {
    ss << "uniform " << uniform.type << " " << uniform.name << ";\n";
  }
  std::cout << ss.str();
  return ss.str();
}

std::string GLShader::vertex_interface_declare(const ShaderCreateInfo &info) const
{
  std::stringstream ss;
  for (const ShaderCreateInfo::VertIn &attr : info.vertex_inputs_) {
#if 1 /* If using layout. */
    ss << "layout(location = " << attr.index << ") ";
#endif
    ss << "in " << attr.type << " " << attr.name << ";\n";
  }

  for (const StageInterfaceInfo *iface : info.vertex_out_interfaces_) {
    ss << "out " << iface->name << "{" << std::endl;
    for (const StageInterfaceInfo::InOut &inout : iface->inouts) {
      ss << "  " << inout.interp << " " << inout.type << " " << inout.name << ";\n";
    }
    ss << "} " << iface->instance_name << ";" << std::endl;
  }
  std::cout << ss.str();
  return ss.str();
}

std::string GLShader::fragment_interface_declare(const ShaderCreateInfo &info) const
{
  std::stringstream ss;

  const Vector<StageInterfaceInfo *> &in_interfaces = (info.geometry_source_.is_empty()) ?
                                                          info.vertex_out_interfaces_ :
                                                          info.geometry_out_interfaces_;

  for (const StageInterfaceInfo *iface : in_interfaces) {
    ss << "in " << iface->name << "{" << std::endl;
    for (const StageInterfaceInfo::InOut &inout : iface->inouts) {
      ss << "  " << inout.interp << " " << inout.type << " " << inout.name << ";\n";
    }
    ss << "} " << iface->instance_name << ";" << std::endl;
  }

  for (const ShaderCreateInfo::VertIn &attr : info.vertex_inputs_) {
#if 1 /* If using layout. */
    ss << "layout(location = " << attr.index << ") ";
#endif
    ss << "in " << attr.type << " " << attr.name << ";\n";
  }
  std::cout << ss.str();
  return ss.str();
}

std::string GLShader::geometry_interface_declare(const ShaderCreateInfo &info) const
{
  std::stringstream ss;
  for (const ShaderCreateInfo::FragOut &output : info.fragment_outputs_) {
    ss << "layout(location = " << output.index;
    switch (output.blend) {
      case SRC_0:
        ss << ", index = 0";
        break;
      case SRC_1:
        ss << ", index = 1";
        break;
      default:
        break;
    }
    ss << ") ";
    ss << "out " << output.type << " " << output.name << ";\n";
  }

  for (const StageInterfaceInfo *iface : info.vertex_out_interfaces_) {
    ss << "in " << iface->name << "{" << std::endl;
    for (const StageInterfaceInfo::InOut &inout : iface->inouts) {
      ss << "  " << inout.interp << " " << inout.type << " " << inout.name << ";\n";
    }
    ss << "} " << iface->instance_name << "[];" << std::endl;
  }

  for (const StageInterfaceInfo *iface : info.geometry_out_interfaces_) {
    ss << "out " << iface->name << "{" << std::endl;
    for (const StageInterfaceInfo::InOut &inout : iface->inouts) {
      ss << "  " << inout.interp << " " << inout.type << " " << inout.name << ";\n";
    }
    ss << "} " << iface->instance_name << ";" << std::endl;
  }
  return ss.str();
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Shader stage creation
 * \{ */

static char *glsl_patch_default_get()
{
  /** Used for shader patching. Init once. */
  static char patch[512] = "\0";
  if (patch[0] != '\0') {
    return patch;
  }

  size_t slen = 0;
  /* Version need to go first. */
  STR_CONCAT(patch, slen, "#version 330\n");

  /* Enable extensions for features that are not part of our base GLSL version
   * don't use an extension for something already available! */
  if (GLContext::texture_gather_support) {
    STR_CONCAT(patch, slen, "#extension GL_ARB_texture_gather: enable\n");
    /* Some drivers don't agree on GLEW_ARB_texture_gather and the actual support in the
     * shader so double check the preprocessor define (see T56544). */
    STR_CONCAT(patch, slen, "#ifdef GL_ARB_texture_gather\n");
    STR_CONCAT(patch, slen, "#  define GPU_ARB_texture_gather\n");
    STR_CONCAT(patch, slen, "#endif\n");
  }
  if (GLContext::shader_draw_parameters_support) {
    STR_CONCAT(patch, slen, "#extension GL_ARB_shader_draw_parameters : enable\n");
    STR_CONCAT(patch, slen, "#define GPU_ARB_shader_draw_parameters\n");
  }
  if (GLContext::texture_cube_map_array_support) {
    STR_CONCAT(patch, slen, "#extension GL_ARB_texture_cube_map_array : enable\n");
    STR_CONCAT(patch, slen, "#define GPU_ARB_texture_cube_map_array\n");
  }

  /* Derivative sign can change depending on implementation. */
  STR_CONCATF(patch, slen, "#define DFDX_SIGN %1.1f\n", GLContext::derivative_signs[0]);
  STR_CONCATF(patch, slen, "#define DFDY_SIGN %1.1f\n", GLContext::derivative_signs[1]);

  BLI_assert(slen < sizeof(patch));
  return patch;
}

static char *glsl_patch_compute_get()
{
  /** Used for shader patching. Init once. */
  static char patch[512] = "\0";
  if (patch[0] != '\0') {
    return patch;
  }

  size_t slen = 0;
  /* Version need to go first. */
  STR_CONCAT(patch, slen, "#version 430\n");
  STR_CONCAT(patch, slen, "#extension GL_ARB_compute_shader :enable\n");
  BLI_assert(slen < sizeof(patch));
  return patch;
}

char *GLShader::glsl_patch_get(GLenum gl_stage)
{
  if (gl_stage == GL_COMPUTE_SHADER) {
    return glsl_patch_compute_get();
  }
  return glsl_patch_default_get();
}

GLuint GLShader::create_shader_stage(GLenum gl_stage, MutableSpan<const char *> sources)
{
  GLuint shader = glCreateShader(gl_stage);
  if (shader == 0) {
    fprintf(stderr, "GLShader: Error: Could not create shader object.");
    return 0;
  }

  /* Patch the shader code using the first source slot. */
  sources[0] = glsl_patch_get(gl_stage);

  glShaderSource(shader, sources.size(), sources.data(), nullptr);
  glCompileShader(shader);

  GLint status;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
  if (!status || (G.debug & G_DEBUG_GPU)) {
    char log[5000] = "";
    glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
    if (log[0] != '\0') {
      GLLogParser parser;
      switch (gl_stage) {
        case GL_VERTEX_SHADER:
          this->print_log(sources, log, "VertShader", !status, &parser);
          break;
        case GL_GEOMETRY_SHADER:
          this->print_log(sources, log, "GeomShader", !status, &parser);
          break;
        case GL_FRAGMENT_SHADER:
          this->print_log(sources, log, "FragShader", !status, &parser);
          break;
        case GL_COMPUTE_SHADER:
          this->print_log(sources, log, "ComputeShader", !status, &parser);
          break;
      }
    }
  }
  if (!status) {
    glDeleteShader(shader);
    compilation_failed_ = true;
    return 0;
  }

  debug::object_label(gl_stage, shader, name);

  glAttachShader(shader_program_, shader);
  return shader;
}

void GLShader::vertex_shader_from_glsl(MutableSpan<const char *> sources)
{
  vert_shader_ = this->create_shader_stage(GL_VERTEX_SHADER, sources);
}

void GLShader::geometry_shader_from_glsl(MutableSpan<const char *> sources)
{
  geom_shader_ = this->create_shader_stage(GL_GEOMETRY_SHADER, sources);
}

void GLShader::fragment_shader_from_glsl(MutableSpan<const char *> sources)
{
  frag_shader_ = this->create_shader_stage(GL_FRAGMENT_SHADER, sources);
}

void GLShader::compute_shader_from_glsl(MutableSpan<const char *> sources)
{
  compute_shader_ = this->create_shader_stage(GL_COMPUTE_SHADER, sources);
}

bool GLShader::finalize()
{
  if (compilation_failed_) {
    return false;
  }

  glLinkProgram(shader_program_);

  GLint status;
  glGetProgramiv(shader_program_, GL_LINK_STATUS, &status);
  if (!status) {
    char log[5000];
    glGetProgramInfoLog(shader_program_, sizeof(log), nullptr, log);
    Span<const char *> sources;
    GLLogParser parser;
    this->print_log(sources, log, "Linking", true, &parser);
    return false;
  }

  interface = new GLShaderInterface(shader_program_);

  return true;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Binding
 * \{ */

void GLShader::bind()
{
  BLI_assert(shader_program_ != 0);
  glUseProgram(shader_program_);
}

void GLShader::unbind()
{
#ifndef NDEBUG
  glUseProgram(0);
#endif
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Transform feedback
 *
 * TODO(fclem): Should be replaced by compute shaders.
 * \{ */

void GLShader::transform_feedback_names_set(Span<const char *> name_list,
                                            const eGPUShaderTFBType geom_type)
{
  glTransformFeedbackVaryings(
      shader_program_, name_list.size(), name_list.data(), GL_INTERLEAVED_ATTRIBS);
  transform_feedback_type_ = geom_type;
}

bool GLShader::transform_feedback_enable(GPUVertBuf *buf_)
{
  if (transform_feedback_type_ == GPU_SHADER_TFB_NONE) {
    return false;
  }

  GLVertBuf *buf = static_cast<GLVertBuf *>(unwrap(buf_));

  BLI_assert(buf->vbo_id_ != 0);

  glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, buf->vbo_id_);

  switch (transform_feedback_type_) {
    case GPU_SHADER_TFB_POINTS:
      glBeginTransformFeedback(GL_POINTS);
      break;
    case GPU_SHADER_TFB_LINES:
      glBeginTransformFeedback(GL_LINES);
      break;
    case GPU_SHADER_TFB_TRIANGLES:
      glBeginTransformFeedback(GL_TRIANGLES);
      break;
    default:
      return false;
  }
  return true;
}

void GLShader::transform_feedback_disable()
{
  glEndTransformFeedback();
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Uniforms setters
 * \{ */

void GLShader::uniform_float(int location, int comp_len, int array_size, const float *data)
{
  switch (comp_len) {
    case 1:
      glUniform1fv(location, array_size, data);
      break;
    case 2:
      glUniform2fv(location, array_size, data);
      break;
    case 3:
      glUniform3fv(location, array_size, data);
      break;
    case 4:
      glUniform4fv(location, array_size, data);
      break;
    case 9:
      glUniformMatrix3fv(location, array_size, 0, data);
      break;
    case 16:
      glUniformMatrix4fv(location, array_size, 0, data);
      break;
    default:
      BLI_assert(0);
      break;
  }
}

void GLShader::uniform_int(int location, int comp_len, int array_size, const int *data)
{
  switch (comp_len) {
    case 1:
      glUniform1iv(location, array_size, data);
      break;
    case 2:
      glUniform2iv(location, array_size, data);
      break;
    case 3:
      glUniform3iv(location, array_size, data);
      break;
    case 4:
      glUniform4iv(location, array_size, data);
      break;
    default:
      BLI_assert(0);
      break;
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name GPUVertFormat from Shader
 * \{ */

static uint calc_component_size(const GLenum gl_type)
{
  switch (gl_type) {
    case GL_FLOAT_VEC2:
    case GL_INT_VEC2:
    case GL_UNSIGNED_INT_VEC2:
      return 2;
    case GL_FLOAT_VEC3:
    case GL_INT_VEC3:
    case GL_UNSIGNED_INT_VEC3:
      return 3;
    case GL_FLOAT_VEC4:
    case GL_FLOAT_MAT2:
    case GL_INT_VEC4:
    case GL_UNSIGNED_INT_VEC4:
      return 4;
    case GL_FLOAT_MAT3:
      return 9;
    case GL_FLOAT_MAT4:
      return 16;
    case GL_FLOAT_MAT2x3:
    case GL_FLOAT_MAT3x2:
      return 6;
    case GL_FLOAT_MAT2x4:
    case GL_FLOAT_MAT4x2:
      return 8;
    case GL_FLOAT_MAT3x4:
    case GL_FLOAT_MAT4x3:
      return 12;
    default:
      return 1;
  }
}

static void get_fetch_mode_and_comp_type(int gl_type,
                                         GPUVertCompType *r_comp_type,
                                         GPUVertFetchMode *r_fetch_mode)
{
  switch (gl_type) {
    case GL_FLOAT:
    case GL_FLOAT_VEC2:
    case GL_FLOAT_VEC3:
    case GL_FLOAT_VEC4:
    case GL_FLOAT_MAT2:
    case GL_FLOAT_MAT3:
    case GL_FLOAT_MAT4:
    case GL_FLOAT_MAT2x3:
    case GL_FLOAT_MAT2x4:
    case GL_FLOAT_MAT3x2:
    case GL_FLOAT_MAT3x4:
    case GL_FLOAT_MAT4x2:
    case GL_FLOAT_MAT4x3:
      *r_comp_type = GPU_COMP_F32;
      *r_fetch_mode = GPU_FETCH_FLOAT;
      break;
    case GL_INT:
    case GL_INT_VEC2:
    case GL_INT_VEC3:
    case GL_INT_VEC4:
      *r_comp_type = GPU_COMP_I32;
      *r_fetch_mode = GPU_FETCH_INT;
      break;
    case GL_UNSIGNED_INT:
    case GL_UNSIGNED_INT_VEC2:
    case GL_UNSIGNED_INT_VEC3:
    case GL_UNSIGNED_INT_VEC4:
      *r_comp_type = GPU_COMP_U32;
      *r_fetch_mode = GPU_FETCH_INT;
      break;
    default:
      BLI_assert(0);
  }
}

void GLShader::vertformat_from_shader(GPUVertFormat *format) const
{
  GPU_vertformat_clear(format);

  GLint attr_len;
  glGetProgramiv(shader_program_, GL_ACTIVE_ATTRIBUTES, &attr_len);

  for (int i = 0; i < attr_len; i++) {
    char name[256];
    GLenum gl_type;
    GLint size;
    glGetActiveAttrib(shader_program_, i, sizeof(name), nullptr, &size, &gl_type, name);

    /* Ignore OpenGL names like `gl_BaseInstanceARB`, `gl_InstanceID` and `gl_VertexID`. */
    if (glGetAttribLocation(shader_program_, name) == -1) {
      continue;
    }

    GPUVertCompType comp_type;
    GPUVertFetchMode fetch_mode;
    get_fetch_mode_and_comp_type(gl_type, &comp_type, &fetch_mode);

    int comp_len = calc_component_size(gl_type) * size;

    GPU_vertformat_attr_add(format, name, comp_type, comp_len, fetch_mode);
  }
}

int GLShader::program_handle_get() const
{
  return (int)this->shader_program_;
}

/** \} */
