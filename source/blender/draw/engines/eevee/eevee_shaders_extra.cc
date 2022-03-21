/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. */

/** \file
 * \ingroup EEVEE
 *
 * This file is only there to handle ShaderCreateInfos.
 */

#include "GPU_shader.h"

#include "BLI_string_ref.hh"

#include "gpu_shader_create_info.hh"

#include "eevee_private.h"

void eevee_shader_material_create_info_amend(GPUMaterial *gpumat,
                                             GPUCodegenOutput *codegen_,
                                             char *frag,
                                             char *vert,
                                             char *geom,
                                             char *defines)
{
  using namespace blender::gpu::shader;

  uint64_t options = GPU_material_uuid_get(gpumat);
  const bool is_background = (options & (VAR_WORLD_PROBE | VAR_WORLD_BACKGROUND)) != 0;

  GPUCodegenOutput &codegen = *codegen_;
  ShaderCreateInfo &info = *reinterpret_cast<ShaderCreateInfo *>(codegen.create_info);

  info.auto_resource_location(true);

  std::stringstream global_vars;

  std::stringstream attr_load;
  attr_load << "void attrib_load()\n";
  attr_load << "{\n";
  attr_load << ((codegen.attr_load) ? codegen.attr_load : "");
  attr_load << "}\n\n";

  std::stringstream vert_gen, frag_gen;

  if (is_background) {
    frag_gen << attr_load.str();
  }
  else {
    vert_gen << attr_load.str();
  }
  vert_gen << vert;
  info.vertex_source_generated = vert_gen.str();
  /* Everything is in generated source. */
  info.vertex_source("eevee_empty.glsl");

  {
    frag_gen << frag;
    frag_gen << "Closure nodetree_exec()\n";
    frag_gen << "{\n";
    if (GPU_material_is_volume_shader(gpumat)) {
      //   frag_gen << ((codegen.volume) ? codegen.volume : "return CLOSURE_DEFAULT;\n");
      frag_gen << "return CLOSURE_DEFAULT;\n";
    }
    else {
      //   frag_gen << ((codegen.surface) ? codegen.surface : "return CLOSURE_DEFAULT;\n");
      frag_gen << "return CLOSURE_DEFAULT;\n";
    }
    frag_gen << "}\n\n";

    info.fragment_source_generated = frag_gen.str();
    /* Everything is in generated source. */
    info.fragment_source("eevee_empty.glsl");
  }

  if (geom) {
    // info.geometry_source_generated = blender::StringRefNull(geom);
    /* Everything is in generated source. */
    info.geometry_source("eevee_empty.glsl");
  }

  if (defines) {
    info.typedef_source_generated += blender::StringRefNull(defines);
  }

  info.additional_info("draw_view");
}
