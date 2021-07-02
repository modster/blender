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
 * Parse/convert GLSL source from Vulkan GLSL to OpenGL GLSL.
 */

#include "gl_shader_converter.hh"

#include <optional>

#include "CLG_log.h"

static CLG_LogRef LOG = {"gpu.gl.shader.converter"};

namespace blender::gpu {

static bool is_error_state(GLShaderConverterState state)
{
  return !ELEM(state, GLShaderConverterState::OkChanged, GLShaderConverterState::OkUnchanged);
}

struct GLSLPatchResult {
  std::optional<std::string> patched_glsl;
  GLShaderConverterState state = GLShaderConverterState::OkUnchanged;

  void merge(const GLSLPatchResult &other, const StringRef unchanged_result)
  {
    switch (other.state) {
      case GLShaderConverterState::OkUnchanged:
        patched_glsl = unchanged_result;
        break;
      case GLShaderConverterState::OkChanged:
        patched_glsl = other.patched_glsl;
        if (state == GLShaderConverterState::OkUnchanged) {
          state = GLShaderConverterState::OkChanged;
        }
        break;
      case GLShaderConverterState::MismatchedPushConstantNames:
        state = other.state;
        break;
    }
  }
};

class GLSLPatch {
 private:
  static constexpr StringRef WHITESPACES = " \t\n\v\f\r";
  static constexpr StringRef VALID_NAME_CHARS =
      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01232456789_";

 public:
  virtual GLSLPatchResult patch(PatchContext &context, StringRef source) = 0;

 protected:
  static StringRef skip_whitespace(StringRef ref)
  {
    size_t skip = ref.find_first_not_of(WHITESPACES);
    if (skip == blender::StringRef::not_found) {
      return ref;
    }
    return ref.drop_prefix(skip);
  }

  static StringRef extract_name(StringRef src)
  {
    StringRef result = src;
    size_t skip = result.find_first_not_of(VALID_NAME_CHARS);
    BLI_assert(skip != StringRef::not_found);
    return result.substr(0, skip);
  }
};

class PatchPushConstants : public GLSLPatch {
  static constexpr StringRef LAYOUT_PUSH_CONSTANTS = "layout(push_constant)";
  static constexpr StringRef LAYOUT_STD140 = "layout(std140)";

 public:
  GLSLPatchResult patch(PatchContext &context, StringRef source) override
  {
    GLSLPatchResult result;

    size_t pos = source.find(LAYOUT_PUSH_CONSTANTS);
    if (pos == StringRef::not_found) {
      result.state = GLShaderConverterState::OkUnchanged;
      return result;
    }

    std::stringstream patched_glsl;
    patched_glsl << source.substr(0, pos);
    patched_glsl << LAYOUT_STD140;
    patched_glsl << source.substr(pos + LAYOUT_PUSH_CONSTANTS.size());

    StringRef name = source.substr(pos + LAYOUT_PUSH_CONSTANTS.size());
    name = skip_whitespace(name);
    name = name.drop_known_prefix("uniform");
    name = skip_whitespace(name);
    name = extract_name(name);

    if (context.push_constants.name.empty()) {
      context.push_constants.name = name;
    }
    else if (context.push_constants.name != name) {
      CLOG_ERROR(&LOG,
                 "Detected different push_constants binding names ('%s' and '%s'). push_constants "
                 "binding names must be identical across all stages.",
                 context.push_constants.name.c_str(),
                 std::string(name).c_str());
      result.state = GLShaderConverterState::MismatchedPushConstantNames;
      return result;
    }

    std::string patched_glsl_str = patched_glsl.str();
    result.state = GLShaderConverterState::OkChanged;
    GLSLPatchResult recursive_result = patch(context, patched_glsl_str);
    result.merge(recursive_result, patched_glsl_str);
    return result;
  };
};

class GLSLPatcher : public GLSLPatch {
 private:
  static void patch(PatchContext &context,
                    GLSLPatch &patch,
                    StringRef source,
                    GLSLPatchResult &r_result)
  {
    /* Do not patch when result is in error so the error state won't be rewritten. */
    if (is_error_state(r_result.state)) {
      return;
    }

    GLSLPatchResult patch_result = patch.patch(context, source);
    r_result.merge(patch_result, source);
  }

 public:
  GLSLPatchResult patch(PatchContext &context, StringRef source) override
  {
    GLSLPatchResult result;
    PatchPushConstants push_constants;
    patch(context, push_constants, source, result);
    return result;
  }
};

void GLShaderConverter::patch(MutableSpan<const char *> sources)
{
  for (int i = 0; i < sources.size(); i++) {
    GLSLPatcher patcher;
    const char *source = sources[i];
    GLSLPatchResult patch_result = patcher.patch(context_, source);
    if (is_error_state(patch_result.state)) {
      state = patch_result.state;
      return;
    }
    if (patch_result.state == GLShaderConverterState::OkChanged) {
      BLI_assert(patch_result.patched_glsl);
      patched_sources_.append(*patch_result.patched_glsl);
      sources[i] = patched_sources_.last().c_str();
    }
  }
}

bool GLShaderConverter::has_errors() const
{
  return is_error_state(state);
}

void GLShaderConverter::free()
{
  patched_sources_.clear();
}

}  // namespace blender::gpu
