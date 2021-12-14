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
 * Descriptior type used to define shader structure, resources and interfaces.
 *
 * Some rule of thumb:
 * - Do not include anything else than this file in each descriptor file.
 */

#pragma once

#include "BLI_string_ref.hh"
#include "BLI_vector.hh"
#include "GPU_texture.h"

namespace blender::gpu::shader {

#ifndef GPU_SHADER_CREATE_INFO
/* Helps intelisense / auto-completion. */
#  define GPU_SHADER_INTERFACE_INFO(_interface, _inst_name) \
    StageInterfaceInfo _interface(#_interface, _inst_name); \
    _interface
#  define GPU_SHADER_CREATE_INFO(_descriptor) \
    ShaderCreateInfo _descriptor(#_descriptor); \
    _descriptor
#endif

enum Type {
  FLOAT = 0,
  VEC2,
  VEC3,
  VEC4,
  MAT4,
  UINT,
  UVEC2,
  UVEC3,
  UVEC4,
  INT,
  IVEC2,
  IVEC3,
  IVEC4,
  BOOL,
};

/* Samplers & images. */
enum ImageType {
  FLOAT_BUFFER = 0,
  FLOAT_1D,
  FLOAT_1D_ARRAY,
  FLOAT_2D,
  FLOAT_2D_ARRAY,
  FLOAT_3D,
  INT_BUFFER,
  INT_1D,
  INT_1D_ARRAY,
  INT_2D,
  INT_2D_ARRAY,
  INT_3D,
  UINT_BUFFER,
  UINT_1D,
  UINT_1D_ARRAY,
  UINT_2D,
  UINT_2D_ARRAY,
  UINT_3D,
  SHADOW_2D,
  SHADOW_2D_ARRAY,
};

/* Storage qualifiers. */
enum Qualifier {
  RESTRICT = (1 << 0),
  READ_ONLY = (1 << 1),
  WRITE_ONLY = (1 << 2),
};

enum Frequency {
  BATCH = 0,
  PASS,
};

/* Dual Source Blending Index. */
enum DualBlend {
  NONE = 0,
  SRC_0,
  SRC_1,
};

/* Interpolation qualifiers. */
enum Interpolation {
  SMOOTH = 0,
  FLAT,
  NO_PERSPECTIVE,
};

struct StageInterfaceInfo {
 private:
  struct InOut {
    Interpolation interp;
    Type type;
    StringRefNull name;
  };

  StringRefNull name_;
  /** Name of the instance of the block (used to access). Can be empty "". */
  StringRefNull instance_name_;
  /** List of all members of the interface. */
  Vector<InOut> inouts;

 public:
  StageInterfaceInfo(const char *name, const char *instance_name)
      : name_(name), instance_name_(instance_name){};
  ~StageInterfaceInfo(){};

  using Self = StageInterfaceInfo;

  Self &smooth(Type type, StringRefNull _name)
  {
    inouts.append({Interpolation::SMOOTH, type, _name});
    return *(Self *)this;
  }

  Self &flat(Type type, StringRefNull _name)
  {
    inouts.append({Interpolation::FLAT, type, _name});
    return *(Self *)this;
  }

  Self &no_perspective(Type type, StringRefNull _name)
  {
    inouts.append({Interpolation::NO_PERSPECTIVE, type, _name});
    return *(Self *)this;
  }
};

/**
 * @brief Describe inputs & outputs, stage interfaces, resources and sources of a shader.
 *        If all data is correctly provided, this is all that is needed to create and compile
 *        a GPUShader.
 *
 * IMPORTANT: All strings are references only. Make sure all the strings used by a
 *            ShaderCreateInfo are not freed until it is consumed or deleted.
 */
struct ShaderCreateInfo {
 private:
  /** Shader name for debugging. */
  StringRefNull name_;
  /** True if the shader is static and can be precompiled at compile time. */
  bool do_static_compilation_ = false;
  /** Only for compute shaders. */
  int local_group_size_[3] = {0, 0, 0};

  struct VertIn {
    int index;
    Type type;
    StringRefNull name;
  };
  Vector<VertIn> vertex_inputs_;

  struct FragOut {
    int index;
    Type type;
    DualBlend blend;
    StringRefNull name;
  };
  Vector<FragOut> fragment_outputs_;

  struct Sampler {
    ImageType type;
    eGPUSamplerState sampler;
    StringRefNull name;
  };

  struct Image {
    eGPUTextureFormat format;
    ImageType type;
    Qualifier qualifiers;
    StringRefNull name;
  };

  struct UniformBuf {
    StringRefNull struct_name;
    StringRefNull name;
  };

  struct StorageBuf {
    Qualifier qualifiers;
    StringRefNull struct_name;
    StringRefNull name;
  };

  struct Resource {
    enum BindType {
      UNIFORM_BUFFER = 0,
      STORAGE_BUFFER,
      SAMPLER,
      IMAGE,
    };

    BindType bind_type;
    int slot;
    union {
      Sampler sampler;
      Image image;
      UniformBuf uniformbuf;
      StorageBuf storagebuf;
    };

    Resource(BindType type, int _slot) : bind_type(type), slot(_slot){};
  };
  /**
   * Resources are grouped by frequency of change.
   * Pass resources are meants to be valid for the whole pass.
   * Batch resources can be changed in a more granular manner (per object/material).
   * Mis-usage will only produce suboptimal performance.
   */
  Vector<Resource> pass_resources_, batch_resources_;

  Vector<StageInterfaceInfo *> vertex_out_interfaces_;
  Vector<StageInterfaceInfo *> geometry_out_interfaces_;

  /* Push constants needs are the same as vertex input. */
  using PushConst = VertIn;

  Vector<PushConst> push_constants_;

  StringRefNull vertex_source_, geometry_source_, fragment_source_, compute_source_;

  Vector<std::array<StringRefNull, 2>> defines_;
  /**
   * Name of other descriptors to recursively merge with this one.
   * No data slot must overlap otherwise we throw an error.
   */
  Vector<StringRefNull> additional_infos_;

 public:
  ShaderCreateInfo(const char *name) : name_(name){};
  ~ShaderCreateInfo(){};

  using Self = ShaderCreateInfo;

  /* -------------------------------------------------------------------- */
  /** \name Shaders in/outs (fixed function pipeline config)
   * \{ */

  Self &vertex_in(int slot, Type type, StringRefNull name)
  {
    vertex_inputs_.append({slot, type, name});
    return *(Self *)this;
  }

  Self &vertex_out(StageInterfaceInfo &interface)
  {
    vertex_out_interfaces_.append(&interface);
    return *(Self *)this;
  }

  /* Only needed if geometry shader is enabled. */
  Self &geometry_out(StageInterfaceInfo &interface)
  {
    geometry_out_interfaces_.append(&interface);
    return *(Self *)this;
  }

  Self &fragment_out(int slot, Type type, StringRefNull name, DualBlend blend = NONE)
  {
    fragment_outputs_.append({slot, type, blend, name});
    return *(Self *)this;
  }

  /** \} */

  /* -------------------------------------------------------------------- */
  /** \name Resources bindings points
   * \{ */

  Self &uniform_buf(int slot, StringRefNull struct_name, StringRefNull name, Frequency freq = PASS)
  {
    Resource res(Resource::BindType::UNIFORM_BUFFER, slot);
    res.uniformbuf.name = name;
    res.uniformbuf.struct_name = struct_name;
    ((freq == PASS) ? pass_resources_ : batch_resources_).append(res);
    return *(Self *)this;
  }

  Self &storage_buf(int slot,
                    Qualifier qualifiers,
                    StringRefNull struct_name,
                    StringRefNull name,
                    Frequency freq = PASS)
  {
    Resource res(Resource::BindType::STORAGE_BUFFER, slot);
    res.storagebuf.qualifiers = qualifiers;
    res.storagebuf.struct_name = struct_name;
    res.storagebuf.name = name;
    ((freq == PASS) ? pass_resources_ : batch_resources_).append(res);
    return *(Self *)this;
  }

  Self &image(int slot,
              eGPUTextureFormat format,
              Qualifier qualifiers,
              ImageType type,
              StringRefNull name,
              Frequency freq = PASS)
  {
    Resource res(Resource::BindType::IMAGE, slot);
    res.image.format = format;
    res.image.qualifiers = qualifiers;
    res.image.type = type;
    res.image.name = name;
    ((freq == PASS) ? pass_resources_ : batch_resources_).append(res);
    return *(Self *)this;
  }

  Self &sampler(int slot,
                ImageType type,
                StringRefNull name,
                Frequency freq = PASS,
                eGPUSamplerState sampler = (eGPUSamplerState)-1)
  {
    Resource res(Resource::BindType::SAMPLER, slot);
    res.sampler.type = type;
    res.sampler.name = name;
    res.sampler.sampler = sampler;
    ((freq == PASS) ? pass_resources_ : batch_resources_).append(res);
    return *(Self *)this;
  }

  /** \} */

  /* -------------------------------------------------------------------- */
  /** \name Shader Source
   * \{ */

  Self &vertex_source(StringRefNull filename)
  {
    vertex_source_ = filename;
    return *(Self *)this;
  }

  Self &geometry_source(StringRefNull filename)
  {
    geometry_source_ = filename;
    return *(Self *)this;
  }

  Self &fragment_source(StringRefNull filename)
  {
    fragment_source_ = filename;
    return *(Self *)this;
  }

  Self &compute_source(StringRefNull filename)
  {
    compute_source_ = filename;
    return *(Self *)this;
  }

  /** \} */

  /* -------------------------------------------------------------------- */
  /** \name Push constants
   *
   * Data managed by GPUShader. Can be set through uniform functions. Must be less than 128bytes.
   * One slot represents 4bytes. Each element needs to have enough empty space left after it.
   * example:
   * [0] = PUSH_CONSTANT(MAT4, "ModelMatrix"),
   * ---- 16 slots occupied by ModelMatrix ----
   * [16] = PUSH_CONSTANT(VEC4, "color"),
   * ---- 4 slots occupied by color ----
   * [20] = PUSH_CONSTANT(BOOL, "srgbToggle"),
   * \{ */

  Self &push_constant(int slot, Type type, StringRefNull name)
  {
    push_constants_.append({slot, type, name});
    return *(Self *)this;
  }

  /** \} */

  /* -------------------------------------------------------------------- */
  /** \name Compute shaders Local Group Size
   * \{ */

  Self &local_group_size(int x, int y = 1, int z = 1)
  {
    local_group_size_[0] = x;
    local_group_size_[1] = y;
    local_group_size_[2] = z;
    return *(Self *)this;
  }

  /** \} */

  /* -------------------------------------------------------------------- */
  /** \name Defines
   * \{ */

  Self &define(StringRefNull name, StringRefNull value = "")
  {
    defines_.append({name, value});
    return *(Self *)this;
  }

  /** \} */

  /* -------------------------------------------------------------------- */
  /** \name Defines
   * \{ */

  Self &do_static_compilation(bool value)
  {
    do_static_compilation_ = value;
    return *(Self *)this;
  }

  /** \} */

  /* -------------------------------------------------------------------- */
  /** \name Additional Create Info
   *
   * Used to share parts of the infos that are common to many shaders.
   * \{ */

  Self &additional_info(StringRefNull descriptor_name0,
                        StringRefNull descriptor_name1 = "",
                        StringRefNull descriptor_name2 = "",
                        StringRefNull descriptor_name3 = "",
                        StringRefNull descriptor_name4 = "")
  {
    additional_infos_.append(descriptor_name0);
    if (!descriptor_name1.is_empty()) {
      additional_infos_.append(descriptor_name1);
    }
    if (!descriptor_name2.is_empty()) {
      additional_infos_.append(descriptor_name2);
    }
    if (!descriptor_name3.is_empty()) {
      additional_infos_.append(descriptor_name3);
    }
    if (!descriptor_name4.is_empty()) {
      additional_infos_.append(descriptor_name4);
    }
    return *(Self *)this;
  }

  /** \} */
};

}  // namespace blender::gpu::shader
