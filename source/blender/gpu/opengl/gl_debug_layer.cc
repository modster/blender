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

/* Manual line breaks for readability. */
/* clang-format off */
#define _VA_ARG_LIST1(t) t
#define _VA_ARG_LIST2(t, a) t a
#define _VA_ARG_LIST4(t, a, b, c) \
  _VA_ARG_LIST2(t, a), _VA_ARG_LIST2(b, c)
#define _VA_ARG_LIST6(t, a, b, c, d, e) \
  _VA_ARG_LIST2(t, a), _VA_ARG_LIST4(b, c, d, e)
#define _VA_ARG_LIST8(t, a, b, c, d, e, f, g) \
  _VA_ARG_LIST2(t, a), _VA_ARG_LIST6(b, c, d, e, f, g)
#define _VA_ARG_LIST10(t, a, b, c, d, e, f, g, h, i) \
  _VA_ARG_LIST2(t, a), _VA_ARG_LIST8(b, c, d, e, f, g, h, i)
#define _VA_ARG_LIST12(t, a, b, c, d, e, f, g, h, i, j, k) \
  _VA_ARG_LIST2(t, a), _VA_ARG_LIST10(b, c, d, e, f, g, h, i, j, k)
#define _VA_ARG_LIST14(t, a, b, c, d, e, f, g, h, i, j, k, l, m) \
  _VA_ARG_LIST2(t, a), _VA_ARG_LIST12(b, c, d, e, f, g, h, i, j, k, l, m)
#define _VA_ARG_LIST16(t, a, b, c, d, e, f, g, h, i, j, k, l, m, o, p) \
  _VA_ARG_LIST2(t, a), _VA_ARG_LIST14(b, c, d, e, f, g, h, i, j, k, l, m, o, p)
#define _VA_ARG_LIST18(t, a, b, c, d, e, f, g, h, i, j, k, l, m, o, p, q, r) \
  _VA_ARG_LIST2(t, a), _VA_ARG_LIST16(b, c, d, e, f, g, h, i, j, k, l, m, o, p, q, r)
#define _VA_ARG_LIST20(t, a, b, c, d, e, f, g, h, i, j, k, l, m, o, p, q, r, s, u) \
  _VA_ARG_LIST2(t, a), _VA_ARG_LIST18(b, c, d, e, f, g, h, i, j, k, l, m, o, p, q, r, s, u)
#define ARG_LIST(...) VA_NARGS_CALL_OVERLOAD(_VA_ARG_LIST, __VA_ARGS__)

#define _VA_ARG_LIST_CALL1(t)
#define _VA_ARG_LIST_CALL2(t, a) a
#define _VA_ARG_LIST_CALL4(t, a, b, c) \
  _VA_ARG_LIST_CALL2(t, a), _VA_ARG_LIST_CALL2(b, c)
#define _VA_ARG_LIST_CALL6(t, a, b, c, d, e) \
  _VA_ARG_LIST_CALL2(t, a), _VA_ARG_LIST_CALL4(b, c, d, e)
#define _VA_ARG_LIST_CALL8(t, a, b, c, d, e, f, g) \
  _VA_ARG_LIST_CALL2(t, a), _VA_ARG_LIST_CALL6(b, c, d, e, f, g)
#define _VA_ARG_LIST_CALL10(t, a, b, c, d, e, f, g, h, i) \
  _VA_ARG_LIST_CALL2(t, a), _VA_ARG_LIST_CALL8(b, c, d, e, f, g, h, i)
#define _VA_ARG_LIST_CALL12(t, a, b, c, d, e, f, g, h, i, j, k) \
  _VA_ARG_LIST_CALL2(t, a), _VA_ARG_LIST_CALL10(b, c, d, e, f, g, h, i, j, k)
#define _VA_ARG_LIST_CALL14(t, a, b, c, d, e, f, g, h, i, j, k, l, m) \
  _VA_ARG_LIST_CALL2(t, a), _VA_ARG_LIST_CALL12(b, c, d, e, f, g, h, i, j, k, l, m)
#define _VA_ARG_LIST_CALL16(t, a, b, c, d, e, f, g, h, i, j, k, l, m, o, p) \
  _VA_ARG_LIST_CALL2(t, a), _VA_ARG_LIST_CALL14(b, c, d, e, f, g, h, i, j, k, l, m, o, p)
#define _VA_ARG_LIST_CALL18(t, a, b, c, d, e, f, g, h, i, j, k, l, m, o, p, q, r) \
  _VA_ARG_LIST_CALL2(t, a), _VA_ARG_LIST_CALL16(b, c, d, e, f, g, h, i, j, k, l, m, o, p, q, r)
#define _VA_ARG_LIST_CALL20(t, a, b, c, d, e, f, g, h, i, j, k, l, m, o, p, q, r, s, u) \
  _VA_ARG_LIST_CALL2(t, a), _VA_ARG_LIST_CALL18(b, c, d, e, f, g, h, i, j, k, l, m, o, p, q, r, s, u)
#define ARG_LIST_CALL(...) VA_NARGS_CALL_OVERLOAD(_VA_ARG_LIST_CALL, __VA_ARGS__)
/* clang-format on */

typedef void *GPUvoidptr;

#define GPUvoidptr_set void *ret =
#define GPUvoidptr_ret return ret

#define GLboolean_set GLboolean ret =
#define GLboolean_ret return ret

#define void_set
#define void_ret

#define DEBUG_FUNC_DECLARE(pfn, rtn_type, fn, ...) \
  pfn real_##fn; \
  static rtn_type GLAPIENTRY debug_##fn(ARG_LIST(__VA_ARGS__)) \
  { \
    check_gl_error("generated before " #fn); \
    rtn_type##_set real_##fn(ARG_LIST_CALL(__VA_ARGS__)); \
    check_gl_error("" #fn); \
    rtn_type##_ret; \
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
DEBUG_FUNC_DECLARE(PFNGLBEGINQUERYPROC, void, glBeginQuery, GLenum, target, GLuint, id);
DEBUG_FUNC_DECLARE(PFNGLBEGINTRANSFORMFEEDBACKPROC, void, glBeginTransformFeedback, GLenum, primitiveMode);
DEBUG_FUNC_DECLARE(PFNGLBINDBUFFERBASEPROC, void, glBindBufferBase, GLenum, target, GLuint, index, GLuint, buffer);
DEBUG_FUNC_DECLARE(PFNGLBINDBUFFERPROC, void, glBindBuffer, GLenum, target, GLuint, buffer);
DEBUG_FUNC_DECLARE(PFNGLBINDFRAMEBUFFERPROC, void, glBindFramebuffer, GLenum, target, GLuint, framebuffer);
DEBUG_FUNC_DECLARE(PFNGLBINDSAMPLERPROC, void, glBindSampler, GLuint, unit, GLuint, sampler);
DEBUG_FUNC_DECLARE(PFNGLBINDVERTEXARRAYPROC, void, glBindVertexArray, GLuint, array);
DEBUG_FUNC_DECLARE(PFNGLBLITFRAMEBUFFERPROC, void, glBlitFramebuffer, GLint, srcX0, GLint, srcY0, GLint, srcX1, GLint, srcY1, GLint, dstX0, GLint, dstY0, GLint, dstX1, GLint, dstY1, GLbitfield, mask, GLenum, filter);
DEBUG_FUNC_DECLARE(PFNGLBUFFERDATAPROC, void, glBufferData, GLenum, target, GLsizeiptr, size, const void *, data, GLenum, usage);
DEBUG_FUNC_DECLARE(PFNGLBUFFERSUBDATAPROC, void, glBufferSubData, GLenum, target, GLintptr, offset, GLsizeiptr, size, const void *, data);
DEBUG_FUNC_DECLARE(PFNGLDELETEBUFFERSPROC, void, glDeleteBuffers, GLsizei, n, const GLuint *, buffers);
DEBUG_FUNC_DECLARE(PFNGLDELETEFRAMEBUFFERSPROC, void, glDeleteFramebuffers, GLsizei, n, const GLuint*, framebuffers);
DEBUG_FUNC_DECLARE(PFNGLDELETEPROGRAMPROC, void, glDeleteProgram, GLuint, program);
DEBUG_FUNC_DECLARE(PFNGLDELETEQUERIESPROC, void, glDeleteQueries, GLsizei, n, const GLuint *, ids);
DEBUG_FUNC_DECLARE(PFNGLDELETESAMPLERSPROC, void, glDeleteSamplers, GLsizei, count, const GLuint *, samplers);
DEBUG_FUNC_DECLARE(PFNGLDELETESHADERPROC, void, glDeleteShader, GLuint, shader);
DEBUG_FUNC_DECLARE(PFNGLDELETEVERTEXARRAYSPROC, void, glDeleteVertexArrays, GLsizei, n, const GLuint *, arrays);
DEBUG_FUNC_DECLARE(PFNGLDRAWARRAYSINSTANCEDBASEINSTANCEPROC, void, glDrawArraysInstancedBaseInstance, GLenum, mode, GLint, first, GLsizei, count, GLsizei, primcount, GLuint, baseinstance);
DEBUG_FUNC_DECLARE(PFNGLDRAWARRAYSINSTANCEDPROC, void, glDrawArraysInstanced, GLenum, mode, GLint, first, GLsizei, count, GLsizei, primcount);
DEBUG_FUNC_DECLARE(PFNGLDRAWBUFFERSPROC, void, glDrawBuffers, GLsizei, n, const GLenum*, bufs);
DEBUG_FUNC_DECLARE(PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXBASEINSTANCEPROC, void, glDrawElementsInstancedBaseVertexBaseInstance, GLenum, mode, GLsizei, count, GLenum, type, const void *, indices, GLsizei, primcount, GLint, basevertex, GLuint, baseinstance);
DEBUG_FUNC_DECLARE(PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXPROC, void, glDrawElementsInstancedBaseVertex, GLenum, mode, GLsizei, count, GLenum, type, const void *, indices, GLsizei, instancecount, GLint, basevertex);
DEBUG_FUNC_DECLARE(PFNGLENDQUERYPROC, void, glEndQuery, GLenum, target);
DEBUG_FUNC_DECLARE(PFNGLENDTRANSFORMFEEDBACKPROC, void, glEndTransformFeedback, void);
DEBUG_FUNC_DECLARE(PFNGLFRAMEBUFFERTEXTURE2DPROC, void, glFramebufferTexture2D, GLenum, target, GLenum, attachment, GLenum, textarget, GLuint, texture, GLint, level);
DEBUG_FUNC_DECLARE(PFNGLFRAMEBUFFERTEXTURELAYERPROC, void, glFramebufferTextureLayer, GLenum, target, GLenum, attachment, GLuint, texture, GLint, level, GLint, layer);
DEBUG_FUNC_DECLARE(PFNGLFRAMEBUFFERTEXTUREPROC, void, glFramebufferTexture, GLenum, target, GLenum, attachment, GLuint, texture, GLint, level);
DEBUG_FUNC_DECLARE(PFNGLGENBUFFERSPROC, void, glGenBuffers, GLsizei, n, GLuint *, buffers);
DEBUG_FUNC_DECLARE(PFNGLGENERATEMIPMAPPROC, void, glGenerateMipmap, GLenum, target);
DEBUG_FUNC_DECLARE(PFNGLGENERATETEXTUREMIPMAPPROC, void, glGenerateTextureMipmap, GLuint, texture);
DEBUG_FUNC_DECLARE(PFNGLGENFRAMEBUFFERSPROC, void, glGenFramebuffers, GLsizei, n, GLuint *, framebuffers);
DEBUG_FUNC_DECLARE(PFNGLGENQUERIESPROC, void, glGenQueries, GLsizei, n, GLuint *, ids);
DEBUG_FUNC_DECLARE(PFNGLGENSAMPLERSPROC, void, glGenSamplers, GLsizei, n, GLuint *, samplers);
DEBUG_FUNC_DECLARE(PFNGLGENVERTEXARRAYSPROC, void, glGenVertexArrays, GLsizei, n, GLuint *, arrays);
DEBUG_FUNC_DECLARE(PFNGLLINKPROGRAMPROC, void, glLinkProgram, GLuint, program);
DEBUG_FUNC_DECLARE(PFNGLMAPBUFFERRANGEPROC, GPUvoidptr, glMapBufferRange, GLenum, target, GLintptr, offset, GLsizeiptr, length, GLbitfield, access);
DEBUG_FUNC_DECLARE(PFNGLTEXTUREBUFFERPROC, void, glTextureBuffer, GLuint, texture, GLenum, internalformat, GLuint, buffer);
DEBUG_FUNC_DECLARE(PFNGLUNMAPBUFFERPROC, GLboolean, glUnmapBuffer, GLenum, target);
DEBUG_FUNC_DECLARE(PFNGLUSEPROGRAMPROC, void, glUseProgram, GLuint, program);
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