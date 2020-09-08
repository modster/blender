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
 * The Original Code is Copyright (C) 2005 Blender Foundation.
 * All rights reserved.
 */

/** \file
 * \ingroup gpu
 *
 * Implement our own subset of KHR_debug extension.
 * We just wrap some functions
 */

#include "BLI_utildefines.h"

#include "glew-mx.h"

#include "gl_debug.hh"

namespace blender::gpu::debug {

#define _VA_ARG_LIST1(t) t
#define _VA_ARG_LIST2(t, a) t a
#define _VA_ARG_LIST4(t, a, ...) _VA_ARG_LIST2(t, a), _VA_ARG_LIST2(__VA_ARGS__)
#define _VA_ARG_LIST6(t, a, ...) _VA_ARG_LIST2(t, a), _VA_ARG_LIST4(__VA_ARGS__)
#define _VA_ARG_LIST8(t, a, ...) _VA_ARG_LIST2(t, a), _VA_ARG_LIST6(__VA_ARGS__)
#define _VA_ARG_LIST10(t, a, ...) _VA_ARG_LIST2(t, a), _VA_ARG_LIST8(__VA_ARGS__)
#define _VA_ARG_LIST12(t, a, ...) _VA_ARG_LIST2(t, a), _VA_ARG_LIST10(__VA_ARGS__)
#define _VA_ARG_LIST14(t, a, ...) _VA_ARG_LIST2(t, a), _VA_ARG_LIST12(__VA_ARGS__)
#define _VA_ARG_LIST16(t, a, ...) _VA_ARG_LIST2(t, a), _VA_ARG_LIST14(__VA_ARGS__)
#define _VA_ARG_LIST18(t, a, ...) _VA_ARG_LIST2(t, a), _VA_ARG_LIST16(__VA_ARGS__)
#define _VA_ARG_LIST20(t, a, ...) _VA_ARG_LIST2(t, a), _VA_ARG_LIST18(__VA_ARGS__)
#define ARG_LIST(...) VA_NARGS_CALL_OVERLOAD(_VA_ARG_LIST, __VA_ARGS__)

#define _VA_ARG_LIST_CALL1(t)
#define _VA_ARG_LIST_CALL2(t, a) a
#define _VA_ARG_LIST_CALL4(t, a, ...) _VA_ARG_LIST_CALL2(t, a), _VA_ARG_LIST_CALL2(__VA_ARGS__)
#define _VA_ARG_LIST_CALL6(t, a, ...) _VA_ARG_LIST_CALL2(t, a), _VA_ARG_LIST_CALL4(__VA_ARGS__)
#define _VA_ARG_LIST_CALL8(t, a, ...) _VA_ARG_LIST_CALL2(t, a), _VA_ARG_LIST_CALL6(__VA_ARGS__)
#define _VA_ARG_LIST_CALL10(t, a, ...) _VA_ARG_LIST_CALL2(t, a), _VA_ARG_LIST_CALL8(__VA_ARGS__)
#define _VA_ARG_LIST_CALL12(t, a, ...) _VA_ARG_LIST_CALL2(t, a), _VA_ARG_LIST_CALL10(__VA_ARGS__)
#define _VA_ARG_LIST_CALL14(t, a, ...) _VA_ARG_LIST_CALL2(t, a), _VA_ARG_LIST_CALL12(__VA_ARGS__)
#define _VA_ARG_LIST_CALL16(t, a, ...) _VA_ARG_LIST_CALL2(t, a), _VA_ARG_LIST_CALL14(__VA_ARGS__)
#define _VA_ARG_LIST_CALL18(t, a, ...) _VA_ARG_LIST_CALL2(t, a), _VA_ARG_LIST_CALL16(__VA_ARGS__)
#define _VA_ARG_LIST_CALL20(t, a, ...) _VA_ARG_LIST_CALL2(t, a), _VA_ARG_LIST_CALL18(__VA_ARGS__)
#define ARG_LIST_CALL(...) VA_NARGS_CALL_OVERLOAD(_VA_ARG_LIST_CALL, __VA_ARGS__)

#define DEBUG_FUNC_DECLARE_VAL(pfn, rtn_type, fn, ...) \
  pfn real_##fn; \
  static rtn_type GLAPIENTRY debug_##fn(ARG_LIST(__VA_ARGS__)) \
  { \
    check_gl_error("generated before " #fn); \
    rtn_type ret = real_##fn(ARG_LIST_CALL(__VA_ARGS__)); \
    check_gl_error("" #fn); \
    return ret; \
  }

#define DEBUG_FUNC_DECLARE(pfn, fn, ...) \
  pfn real_##fn; \
  static void GLAPIENTRY debug_##fn(ARG_LIST(__VA_ARGS__)) \
  { \
    check_gl_error("generated before " #fn); \
    real_##fn(ARG_LIST_CALL(__VA_ARGS__)); \
    check_gl_error("" #fn); \
  }

#define DEBUG_FUNC_DUMMY(pfn, fn, ...) \
  pfn real_##fn; \
  static void GLAPIENTRY debug_##fn(ARG_LIST(__VA_ARGS__)) \
  { \
    UNUSED_VARS(ARG_LIST_CALL(__VA_ARGS__)); \
  }

/* List of wrapped functions. We dont have to support all of them.
 * Some functions might be declared as extern in GLEW. We cannot override them in this case.
 * Keep the list in alphabetical order. */

/* Avoid very long declarations. */
/* clang-format off */
DEBUG_FUNC_DECLARE_VAL(PFNGLMAPBUFFERRANGEPROC, void *, glMapBufferRange, GLenum, target, GLintptr, offset, GLsizeiptr, length, GLbitfield, access);
DEBUG_FUNC_DECLARE_VAL(PFNGLUNMAPBUFFERPROC, GLboolean, glUnmapBuffer, GLenum, target);
DEBUG_FUNC_DECLARE(PFNGLBEGINQUERYPROC, glBeginQuery, GLenum, target, GLuint, id);
DEBUG_FUNC_DECLARE(PFNGLBEGINTRANSFORMFEEDBACKPROC, glBeginTransformFeedback, GLenum, primitiveMode);
DEBUG_FUNC_DECLARE(PFNGLBINDBUFFERBASEPROC, glBindBufferBase, GLenum, target, GLuint, index, GLuint, buffer);
DEBUG_FUNC_DECLARE(PFNGLBINDBUFFERPROC, glBindBuffer, GLenum, target, GLuint, buffer);
DEBUG_FUNC_DECLARE(PFNGLBINDFRAMEBUFFERPROC, glBindFramebuffer, GLenum, target, GLuint, framebuffer);
DEBUG_FUNC_DECLARE(PFNGLBINDSAMPLERPROC, glBindSampler, GLuint, unit, GLuint, sampler);
DEBUG_FUNC_DECLARE(PFNGLBINDVERTEXARRAYPROC, glBindVertexArray, GLuint, array);
DEBUG_FUNC_DECLARE(PFNGLBLITFRAMEBUFFERPROC, glBlitFramebuffer, GLint, srcX0, GLint, srcY0, GLint, srcX1, GLint, srcY1, GLint, dstX0, GLint, dstY0, GLint, dstX1, GLint, dstY1, GLbitfield, mask, GLenum, filter);
DEBUG_FUNC_DECLARE(PFNGLBUFFERDATAPROC, glBufferData, GLenum, target, GLsizeiptr, size, const void *, data, GLenum, usage);
DEBUG_FUNC_DECLARE(PFNGLBUFFERSUBDATAPROC, glBufferSubData, GLenum, target, GLintptr, offset, GLsizeiptr, size, const void *, data);
DEBUG_FUNC_DECLARE(PFNGLDELETEBUFFERSPROC, glDeleteBuffers, GLsizei, n, const GLuint *, buffers);
DEBUG_FUNC_DECLARE(PFNGLDELETEFRAMEBUFFERSPROC, glDeleteFramebuffers, GLsizei, n, const GLuint*, framebuffers);
DEBUG_FUNC_DECLARE(PFNGLDELETEPROGRAMPROC, glDeleteProgram, GLuint, program);
DEBUG_FUNC_DECLARE(PFNGLDELETEQUERIESPROC, glDeleteQueries, GLsizei, n, const GLuint *, ids);
DEBUG_FUNC_DECLARE(PFNGLDELETESAMPLERSPROC, glDeleteSamplers, GLsizei, count, const GLuint *, samplers);
DEBUG_FUNC_DECLARE(PFNGLDELETESHADERPROC, glDeleteShader, GLuint, shader);
DEBUG_FUNC_DECLARE(PFNGLDELETEVERTEXARRAYSPROC, glDeleteVertexArrays, GLsizei, n, const GLuint *, arrays);
DEBUG_FUNC_DECLARE(PFNGLDRAWARRAYSINSTANCEDBASEINSTANCEPROC, glDrawArraysInstancedBaseInstance, GLenum, mode, GLint, first, GLsizei, count, GLsizei, primcount, GLuint, baseinstance);
DEBUG_FUNC_DECLARE(PFNGLDRAWARRAYSINSTANCEDPROC, glDrawArraysInstanced, GLenum, mode, GLint, first, GLsizei, count, GLsizei, primcount);
DEBUG_FUNC_DECLARE(PFNGLDRAWBUFFERSPROC, glDrawBuffers, GLsizei, n, const GLenum*, bufs);
DEBUG_FUNC_DECLARE(PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXBASEINSTANCEPROC, glDrawElementsInstancedBaseVertexBaseInstance, GLenum, mode, GLsizei, count, GLenum, type, const void *, indices, GLsizei, primcount, GLint, basevertex, GLuint, baseinstance);
DEBUG_FUNC_DECLARE(PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXPROC, glDrawElementsInstancedBaseVertex, GLenum, mode, GLsizei, count, GLenum, type, const void *, indices, GLsizei, instancecount, GLint, basevertex);
DEBUG_FUNC_DECLARE(PFNGLENDQUERYPROC, glEndQuery, GLenum, target);
DEBUG_FUNC_DECLARE(PFNGLENDTRANSFORMFEEDBACKPROC, glEndTransformFeedback, void);
DEBUG_FUNC_DECLARE(PFNGLFRAMEBUFFERTEXTURE2DPROC, glFramebufferTexture2D, GLenum, target, GLenum, attachment, GLenum, textarget, GLuint, texture, GLint, level);
DEBUG_FUNC_DECLARE(PFNGLFRAMEBUFFERTEXTURELAYERPROC, glFramebufferTextureLayer, GLenum, target, GLenum, attachment, GLuint, texture, GLint, level, GLint, layer);
DEBUG_FUNC_DECLARE(PFNGLFRAMEBUFFERTEXTUREPROC, glFramebufferTexture, GLenum, target, GLenum, attachment, GLuint, texture, GLint, level);
DEBUG_FUNC_DECLARE(PFNGLGENBUFFERSPROC, glGenBuffers, GLsizei, n, GLuint *, buffers);
DEBUG_FUNC_DECLARE(PFNGLGENERATEMIPMAPPROC, glGenerateMipmap, GLenum, target);
DEBUG_FUNC_DECLARE(PFNGLGENERATETEXTUREMIPMAPPROC, glGenerateTextureMipmap, GLuint, texture);
DEBUG_FUNC_DECLARE(PFNGLGENFRAMEBUFFERSPROC, glGenFramebuffers, GLsizei, n, GLuint *, framebuffers);
DEBUG_FUNC_DECLARE(PFNGLGENQUERIESPROC, glGenQueries, GLsizei, n, GLuint *, ids);
DEBUG_FUNC_DECLARE(PFNGLGENSAMPLERSPROC, glGenSamplers, GLsizei, n, GLuint *, samplers);
DEBUG_FUNC_DECLARE(PFNGLGENVERTEXARRAYSPROC, glGenVertexArrays, GLsizei, n, GLuint *, arrays);
DEBUG_FUNC_DECLARE(PFNGLLINKPROGRAMPROC, glLinkProgram, GLuint, program);
DEBUG_FUNC_DECLARE(PFNGLTEXTUREBUFFERPROC, glTextureBuffer, GLuint, texture, GLenum, internalformat, GLuint, buffer);
DEBUG_FUNC_DECLARE(PFNGLUSEPROGRAMPROC, glUseProgram, GLuint, program);
DEBUG_FUNC_DUMMY(PFNGLOBJECTLABELPROC, glObjectLabel, GLenum, identifier, GLuint, name, GLsizei, length, const GLchar *, label);
/* clang-format on */

#undef DEBUG_FUNC_DECLARE

/* On some systems,  */
void init_debug_layer(void)
{
#define DEBUG_WRAP(function) \
  do { \
    real_##function = function; \
    function = &debug_##function; \
  } while (0)

  DEBUG_WRAP(glBeginQuery);
  DEBUG_WRAP(glBeginTransformFeedback);
  DEBUG_WRAP(glBindBuffer);
  DEBUG_WRAP(glBindBufferBase);
  DEBUG_WRAP(glBindFramebuffer);
  DEBUG_WRAP(glBindSampler);
  DEBUG_WRAP(glBindVertexArray);
  DEBUG_WRAP(glBlitFramebuffer);
  DEBUG_WRAP(glBufferData);
  DEBUG_WRAP(glBufferSubData);
  DEBUG_WRAP(glDeleteBuffers);
  DEBUG_WRAP(glDeleteFramebuffers);
  DEBUG_WRAP(glDeleteProgram);
  DEBUG_WRAP(glDeleteQueries);
  DEBUG_WRAP(glDeleteSamplers);
  DEBUG_WRAP(glDeleteShader);
  DEBUG_WRAP(glDeleteVertexArrays);
  DEBUG_WRAP(glDrawArraysInstanced);
  DEBUG_WRAP(glDrawArraysInstancedBaseInstance);
  DEBUG_WRAP(glDrawBuffers);
  DEBUG_WRAP(glDrawElementsInstancedBaseVertex);
  DEBUG_WRAP(glDrawElementsInstancedBaseVertexBaseInstance);
  DEBUG_WRAP(glEndQuery);
  DEBUG_WRAP(glEndTransformFeedback);
  DEBUG_WRAP(glFramebufferTexture);
  DEBUG_WRAP(glFramebufferTexture2D);
  DEBUG_WRAP(glFramebufferTextureLayer);
  DEBUG_WRAP(glGenBuffers);
  DEBUG_WRAP(glGenerateMipmap);
  DEBUG_WRAP(glGenerateTextureMipmap);
  DEBUG_WRAP(glGenFramebuffers);
  DEBUG_WRAP(glGenQueries);
  DEBUG_WRAP(glGenSamplers);
  DEBUG_WRAP(glGenVertexArrays);
  DEBUG_WRAP(glLinkProgram);
  DEBUG_WRAP(glMapBufferRange);
  DEBUG_WRAP(glObjectLabel);
  DEBUG_WRAP(glTextureBuffer);
  DEBUG_WRAP(glUnmapBuffer);
  DEBUG_WRAP(glUseProgram);

#undef DEBUG_WRAP
}

}  // namespace blender::gpu::debug