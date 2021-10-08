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
 * Convert material node-trees to GLSL.
 */

#include "MEM_guardedalloc.h"

#include "DNA_customdata_types.h"
#include "DNA_image_types.h"

#include "BLI_blenlib.h"
#include "BLI_dynstr.h"
#include "BLI_ghash.h"
#include "BLI_hash_mm2a.h"
#include "BLI_link_utils.h"
#include "BLI_threads.h"
#include "BLI_utildefines.h"

#include "PIL_time.h"

#include "BKE_material.h"
#include "BKE_world.h"

#include "GPU_capabilities.h"
#include "GPU_material.h"
#include "GPU_shader.h"
#include "GPU_uniform_buffer.h"
#include "GPU_vertex_format.h"

#include "BLI_sys_types.h" /* for intptr_t support */

#include "gpu_codegen.h"
#include "gpu_material_library.h"
#include "gpu_node_graph.h"

#include <stdarg.h>
#include <string.h>

#include <sstream>
#include <string>

extern "C" {
extern char datatoc_gpu_shader_codegen_lib_glsl[];
}
/* -------------------------------------------------------------------- */
/** \name GPUPass Cache
 *
 * Internal shader cache: This prevent the shader recompilation / stall when
 * using undo/redo AND also allows for GPUPass reuse if the Shader code is the
 * same for 2 different Materials. Unused GPUPasses are free by Garbage collection.
 */

/* Only use one linklist that contains the GPUPasses grouped by hash. */
static GPUPass *pass_cache = nullptr;
static SpinLock pass_cache_spin;

/* Search by hash only. Return first pass with the same hash.
 * There is hash collision if (pass->next && pass->next->hash == hash) */
static GPUPass *gpu_pass_cache_lookup(uint32_t hash)
{
  BLI_spin_lock(&pass_cache_spin);
  /* Could be optimized with a Lookup table. */
  for (GPUPass *pass = pass_cache; pass; pass = pass->next) {
    if (pass->hash == hash) {
      BLI_spin_unlock(&pass_cache_spin);
      return pass;
    }
  }
  BLI_spin_unlock(&pass_cache_spin);
  return nullptr;
}

static void gpu_pass_cache_insert_after(GPUPass *node, GPUPass *pass)
{
  BLI_spin_lock(&pass_cache_spin);
  if (node != nullptr) {
    /* Add after the first pass having the same hash. */
    pass->next = node->next;
    node->next = pass;
  }
  else {
    /* No other pass have same hash, just prepend to the list. */
    BLI_LINKS_PREPEND(pass_cache, pass);
  }
  BLI_spin_unlock(&pass_cache_spin);
}

/* Check all possible passes with the same hash. */
static GPUPass *gpu_pass_cache_resolve_collision(GPUPass *pass,
                                                 GPUShaderSource *source,
                                                 uint32_t hash)
{
  BLI_spin_lock(&pass_cache_spin);
  /* Collision, need to `strcmp` the whole shader. */
  for (; pass && (pass->hash == hash); pass = pass->next) {
    if ((source->defines != nullptr) && (!STREQ(pass->source.defines, source->defines))) {
      /* Pass */
    }
    else if ((source->geometry != nullptr) && (!STREQ(pass->source.geometry, source->geometry))) {
      /* Pass */
    }
    else if (STREQ(pass->source.fragment, source->fragment) &&
             STREQ(pass->source.vertex, source->vertex)) {
      BLI_spin_unlock(&pass_cache_spin);
      return pass;
    }
  }
  BLI_spin_unlock(&pass_cache_spin);
  return nullptr;
}

static bool gpu_pass_is_valid(GPUPass *pass)
{
  /* Shader is not null if compilation is successful. */
  return (pass->compiled == false || pass->shader != nullptr);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Type > string conversion
 * \{ */

static std::ostream &operator<<(std::ostream &stream, const GPUInput *input)
{
  switch (input->source) {
    case GPU_SOURCE_OUTPUT:
      return stream << "tmp" << input->id;
    case GPU_SOURCE_CONSTANT:
      return stream << "cons" << input->id;
    case GPU_SOURCE_UNIFORM:
      return stream << "unf" << input->id;
    case GPU_SOURCE_ATTR:
      return stream << "var" << input->attr->id;
    case GPU_SOURCE_UNIFORM_ATTR:
      return stream << "UNIFORM_ATTR_UBO.attr" << input->uniform_attr->id;
    case GPU_SOURCE_STRUCT:
      return stream << "strct" << input->id;
    case GPU_SOURCE_TEX:
      return stream << input->texture->sampler_name;
    case GPU_SOURCE_TEX_TILED_MAPPING:
      return stream << input->texture->tiled_mapping_name;
    case GPU_SOURCE_VOLUME_GRID:
      return stream << input->volume_grid->sampler_name;
    case GPU_SOURCE_VOLUME_GRID_TRANSFORM:
      return stream << input->volume_grid->transform_name;
    default:
      BLI_assert(0);
      return stream;
  }
}

static std::ostream &operator<<(std::ostream &stream, const GPUOutput *output)
{
  return stream << "tmp" << output->id;
}

static std::ostream &operator<<(std::ostream &stream, const eGPUType &type)
{
  return stream << gpu_data_type_to_string(type);
}

/* Trick type to change overload and keep a somewhat nice syntax. */
struct GPUConstant : public GPUInput {
};

/* Print data constructor (i.e: vec2(1.0f, 1.0f)). */
static std::ostream &operator<<(std::ostream &stream, const GPUConstant *input)
{
  stream << input->type << "(";
  for (int i = 0; i < input->type; i++) {
    char formated_float[32];
    /* Print with the maximum precision for single precision float using scientific notation.
     * See https://stackoverflow.com/questions/16839658/#answer-21162120 */
    SNPRINTF(formated_float, "%.9g", input->vec[i]);
    stream << formated_float;
    if (i < input->type - 1) {
      stream << ", ";
    }
  }
  stream << ")";
  return stream;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name GLSL code generation
 * \{ */

class GPUCodegen {
 public:
  GPUMaterial &mat;
  GPUNodeGraph &graph;
  GPUCodegenOutput output = {};

 private:
  uint32_t hash_ = 0;
  BLI_HashMurmur2A hm2a_;
  ListBase ubo_inputs_ = {nullptr, nullptr};

 public:
  GPUCodegen(GPUMaterial *mat_, GPUNodeGraph *graph_) : mat(*mat_), graph(*graph_)
  {
    BLI_hash_mm2a_init(&hm2a_, GPU_material_uuid_get(&mat));
  }

  ~GPUCodegen()
  {
    MEM_SAFE_FREE(output.attribs_declare);
    MEM_SAFE_FREE(output.attribs_interface);
    MEM_SAFE_FREE(output.attribs_load);
    MEM_SAFE_FREE(output.attribs_passthrough);
    MEM_SAFE_FREE(output.surface);
    MEM_SAFE_FREE(output.volume);
    MEM_SAFE_FREE(output.thickness);
    MEM_SAFE_FREE(output.displacement);
    MEM_SAFE_FREE(output.uniforms);
    MEM_SAFE_FREE(output.library);
    BLI_freelistN(&ubo_inputs_);
  };

  void generate_graphs();
  void generate_uniform_buffer();
  void generate_attribs();
  void generate_resources();
  void generate_library();

  uint32_t hash_get() const
  {
    return hash_;
  }

 private:
  void set_unique_ids();

  void node_serialize(std::stringstream &eval_ss, const GPUNode *node);
  char *graph_serialize(eGPUNodeTag tree_tag, GPUNodeLink *output_link);

  static char *extract_c_str(std::stringstream &stream)
  {
    auto string = stream.str();
    return BLI_strdup(string.c_str());
  }
};

static char attr_prefix_get(CustomDataType type)
{
  switch (type) {
    case CD_MTFACE:
      return 'u';
    case CD_TANGENT:
      return 't';
    case CD_MCOL:
      return 'c';
    case CD_PROP_COLOR:
      return 'c';
    case CD_AUTO_FROM_NAME:
      return 'a';
    case CD_HAIRLENGTH:
      return 'l';
    default:
      BLI_assert_msg(0, "GPUVertAttr Prefix type not found : This should not happen!");
      return '\0';
  }
}

void GPUCodegen::generate_attribs()
{
  if (BLI_listbase_is_empty(&graph.attributes)) {
    output.attribs_declare = nullptr;
    output.attribs_interface = nullptr;
    output.attribs_load = nullptr;
    output.attribs_passthrough = nullptr;
    return;
  }

  /* Input declaration, loading / assignment to interface and geometry shader passthrough. */
  std::stringstream decl_ss, iface_ss, load_ss, pass_ss;

  LISTBASE_FOREACH (GPUMaterialAttribute *, attr, &graph.attributes) {
    eGPUType type = attr->gputype;
    eGPUType in_type = attr->gputype;
    char name[GPU_MAX_SAFE_ATTR_NAME + 1] = "orco";
    if (attr->type == CD_ORCO) {
      /* OPTI : orco is computed from local positions, but only if no modifier is present. */
      GPU_material_flag_set(&mat, GPU_MATFLAG_OBJECT_INFO);
      /* Need vec4 to detect usage of default attribute. */
      in_type = GPU_VEC4;
    }
    else {
      name[0] = attr_prefix_get((CustomDataType)(attr->type));
      name[1] = '\0';
    }

    if (attr->name[0] != '\0') {
      /* XXX FIXME : see notes in mesh_render_data_create() */
      GPU_vertformat_safe_attr_name(attr->name, &name[1], GPU_MAX_SAFE_ATTR_NAME);
    }
    /* NOTE : Replicate changes to mesh_render_data_create() in draw_cache_impl_mesh.c */
    decl_ss << in_type << " " << name << ";\n";

    iface_ss << type << " var" << attr->id << ";\n";

    load_ss << "var" << attr->id;

    switch (attr->type) {
      case CD_ORCO:
        load_ss << " = attr_load_orco(" << name << ");\n";
        break;
      case CD_TANGENT:
        load_ss << " = attr_load_tangent(" << name << ");\n";
        break;
      case CD_MTFACE:
        load_ss << " = attr_load_uv(" << name << ");\n";
        break;
      case CD_MCOL:
        load_ss << " = attr_load_color(" << name << ");\n";
        break;
      default:
        load_ss << " = attr_load_" << type << "(" << name << ");\n";
        break;
    }
    pass_ss << "attr_out.var" << attr->id << " = attr_in[vert_id].var" << attr->id << ";\n";
  }

  output.attribs_declare = extract_c_str(decl_ss);
  output.attribs_interface = extract_c_str(iface_ss);
  output.attribs_load = extract_c_str(load_ss);
  output.attribs_passthrough = extract_c_str(pass_ss);
}

void GPUCodegen::generate_resources()
{
  std::stringstream ss;

  /* Textures. */
  LISTBASE_FOREACH (GPUMaterialTexture *, tex, &graph.textures) {
    if (tex->colorband) {
      ss << "uniform sampler1DArray " << tex->sampler_name << ";\n";
    }
    else if (tex->tiled_mapping_name[0] != '\0') {
      ss << "uniform sampler2DArray " << tex->sampler_name << ";\n";
      ss << "uniform sampler1DArray " << tex->tiled_mapping_name << ";\n";
    }
    else {
      ss << "uniform sampler2D " << tex->sampler_name << ";\n";
    }
  }
  /* Volume Grids. */
  LISTBASE_FOREACH (GPUMaterialVolumeGrid *, grid, &graph.volume_grids) {
    ss << "uniform sampler3D " << grid->sampler_name << ";\n";
    /* TODO(fclem) global uniform. To put in a UBO. */
    ss << "uniform mat4 " << grid->transform_name << " = mat4(0.0);\n";
  }

  if (!BLI_listbase_is_empty(&ubo_inputs_)) {
    /* NOTE: generate_uniform_buffer() should have sorted the inputs before this. */
    ss << "layout(std140) uniform nodeTree {\n";
    LISTBASE_FOREACH (LinkData *, link, &ubo_inputs_) {
      GPUInput *input = (GPUInput *)(link->data);
      ss << input->type << " " << input << ";\n";
    }
    ss << "};\n\n";
  }

  if (!BLI_listbase_is_empty(&graph.uniform_attrs.list)) {
    GPU_material_flag_set(&mat, GPU_MATFLAG_UNIFORMS_ATTRIB);

    ss << "struct UniformAttributes {\n";
    LISTBASE_FOREACH (GPUUniformAttr *, attr, &graph.uniform_attrs.list) {
      ss << "vec4 attr" << attr->id << ";\n";
    }
    ss << "};\n\n";
  }

  output.uniforms = extract_c_str(ss);
}

void GPUCodegen::generate_library()
{
  output.library = gpu_material_library_generate_code(graph.used_libraries);
}

void GPUCodegen::node_serialize(std::stringstream &eval_ss, const GPUNode *node)
{
  /* Declare constants. */
  LISTBASE_FOREACH (GPUInput *, input, &node->inputs) {
    switch (input->source) {
      case GPU_SOURCE_STRUCT:
        eval_ss << input->type << " " << input << " = CLOSURE_DEFAULT;\n";
        break;
      case GPU_SOURCE_CONSTANT:
        eval_ss << input->type << " " << input << " = " << (GPUConstant *)input << ";\n";
        break;
      default:
        break;
    }
  }
  /* Declare temporary variables for node output storage. */
  LISTBASE_FOREACH (GPUOutput *, output, &node->outputs) {
    eval_ss << output->type << " " << output << ";\n";
  }

  /* Function call. */
  eval_ss << node->name << "(";
  /* Input arguments. */
  LISTBASE_FOREACH (GPUInput *, input, &node->inputs) {
    switch (input->source) {
      case GPU_SOURCE_OUTPUT:
      case GPU_SOURCE_ATTR: {
        /* These inputs can have non matching types. Do conversion. */
        eGPUType to = input->type;
        eGPUType from = (input->source == GPU_SOURCE_ATTR) ? input->attr->gputype :
                                                             input->link->output->type;
        if (from != to) {
          /* Use defines declared inside codegen_lib (i.e: vec4_from_float). */
          eval_ss << to << "_from_" << from << "(";
        }

        if (input->source == GPU_SOURCE_ATTR) {
          eval_ss << input;
        }
        else {
          eval_ss << input->link->output;
        }

        if (from != to) {
          eval_ss << ")";
        }
        break;
      }
      default:
        eval_ss << input;
        break;
    }
    eval_ss << ", ";
  }
  /* Output arguments. */
  LISTBASE_FOREACH (GPUOutput *, output, &node->outputs) {
    eval_ss << output;
    if (output->next) {
      eval_ss << ", ";
    }
  }
  eval_ss << ");\n\n";
}

char *GPUCodegen::graph_serialize(eGPUNodeTag tree_tag, GPUNodeLink *output_link)
{
  if (output_link == nullptr) {
    return nullptr;
  }

  std::stringstream eval_ss;
  /* Render engine implement this function if needed. */
  eval_ss << "ntree_eval_init();\n\n";
  /* NOTE: The node order is already top to bottom (or left to right in node editor)
   * because of the evaluation order inside ntreeExecGPUNodes(). */
  LISTBASE_FOREACH (GPUNode *, node, &graph.nodes) {
    if ((node->tag & tree_tag) == 0 || (node->tag & GPU_NODE_TAG_EVAL) != 0) {
      continue;
    }
    node_serialize(eval_ss, node);
  }
  /* Render engine implement this function if needed. */
  eval_ss << "ntree_eval_weights();\n\n";
  /* Output eval function at the end. */
  LISTBASE_FOREACH (GPUNode *, node, &graph.nodes) {
    if ((node->tag & tree_tag) == 0 || (node->tag & GPU_NODE_TAG_EVAL) == 0) {
      continue;
    }
    node_serialize(eval_ss, node);
  }
  eval_ss << "return " << output_link->output << ";\n";

  char *eval_c_str = extract_c_str(eval_ss);
  BLI_hash_mm2a_add(&hm2a_, (uchar *)eval_c_str, eval_ss.str().size());
  return eval_c_str;
}

void GPUCodegen::generate_uniform_buffer()
{
  /* Extract uniform inputs. */
  LISTBASE_FOREACH (GPUNode *, node, &graph.nodes) {
    LISTBASE_FOREACH (GPUInput *, input, &node->inputs) {
      if (input->source == GPU_SOURCE_UNIFORM && !input->link) {
        /* We handle the UBO uniforms separately. */
        BLI_addtail(&ubo_inputs_, BLI_genericNodeN(input));
      }
    }
  }
  if (!BLI_listbase_is_empty(&ubo_inputs_)) {
    /* This sorts the inputs based on size. */
    GPU_material_uniform_buffer_create(&mat, &ubo_inputs_);
  }
}

/* Sets id for unique names for all inputs, resources and temp variables. */
void GPUCodegen::set_unique_ids()
{
  int id = 1;
  LISTBASE_FOREACH (GPUNode *, node, &graph.nodes) {
    LISTBASE_FOREACH (GPUInput *, input, &node->inputs) {
      input->id = id++;
    }
    LISTBASE_FOREACH (GPUOutput *, output, &node->outputs) {
      output->id = id++;
    }
  }
}

void GPUCodegen::generate_graphs()
{
  set_unique_ids();

  output.surface = graph_serialize(GPU_NODE_TAG_SURFACE, graph.outlink_surface);
  output.volume = graph_serialize(GPU_NODE_TAG_VOLUME, graph.outlink_volume);
  output.displacement = graph_serialize(GPU_NODE_TAG_DISPLACEMENT, graph.outlink_displacement);
  output.thickness = graph_serialize(GPU_NODE_TAG_THICKNESS, graph.outlink_thickness);

  LISTBASE_FOREACH (GPUMaterialAttribute *, attr, &graph.attributes) {
    BLI_hash_mm2a_add(&hm2a_, (uchar *)attr->name, strlen(attr->name));
  }

  hash_ = BLI_hash_mm2a_end(&hm2a_);
}

GPUPass *GPU_generate_pass(GPUMaterial *material,
                           GPUNodeGraph *graph,
                           GPUCodegenCallbackFn finalize_source_cb,
                           void *thunk)
{
  /* Prune the unused nodes and extract attributes before compiling so the
   * generated VBOs are ready to accept the future shader. */
  gpu_node_graph_prune_unused(graph);
  gpu_node_graph_finalize_uniform_attrs(graph);

  GPUCodegen codegen(material, graph);
  codegen.generate_graphs();
  codegen.generate_uniform_buffer();

  /* Cache lookup: Reuse shaders already compiled. */
  GPUPass *pass_hash = gpu_pass_cache_lookup(codegen.hash_get());

  /* FIXME(fclem): This is broken. Since we only check for the hash and not the full source
   * there is no way to have a collision currently. Some advocated to only use a bigger hash. */
  if (pass_hash && (pass_hash->next == nullptr || pass_hash->next->hash != codegen.hash_get())) {
    if (!gpu_pass_is_valid(pass_hash)) {
      /* Shader has already been created but failed to compile. */
      return nullptr;
    }
    /* No collision, just return the pass. */
    pass_hash->refcount += 1;
    return pass_hash;
  }

  /* Either the shader is not compiled or there is a hash collision...
   * continue generating the shader strings. */
  codegen.generate_attribs();
  codegen.generate_resources();
  codegen.generate_library();

  /* Make engine add its own code and implement the generated functions. */
  GPUShaderSource source = finalize_source_cb(thunk, material, &codegen.output);

  GPUPass *pass = nullptr;
  if (pass_hash) {
    /* Cache lookup: Reuse shaders already compiled. */
    pass = gpu_pass_cache_resolve_collision(pass_hash, &source, codegen.hash_get());
  }

  if (pass) {
    MEM_SAFE_FREE(source.vertex);
    MEM_SAFE_FREE(source.geometry);
    MEM_SAFE_FREE(source.fragment);
    MEM_SAFE_FREE(source.defines);
    /* Cache hit. Reuse the same GPUPass and GPUShader. */
    if (!gpu_pass_is_valid(pass)) {
      /* Shader has already been created but failed to compile. */
      return nullptr;
    }
    pass->refcount += 1;
  }
  else {
    /* We still create a pass even if shader compilation
     * fails to avoid trying to compile again and again. */
    pass = (GPUPass *)MEM_callocN(sizeof(GPUPass), "GPUPass");
    pass->shader = nullptr;
    pass->refcount = 1;
    pass->hash = codegen.hash_get();
    pass->source = source;
    pass->compiled = false;

    gpu_pass_cache_insert_after(pass_hash, pass);
  }
  return pass;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Compilation
 * \{ */

static int count_active_texture_sampler(GPUShader *shader, char *source)
{
  const char *code = source;

  /* Remember this is per stage. */
  GSet *sampler_ids = BLI_gset_int_new(__func__);
  int num_samplers = 0;

  while ((code = strstr(code, "uniform "))) {
    /* Move past "uniform". */
    code += 7;
    /* Skip following spaces. */
    while (*code == ' ') {
      code++;
    }
    /* Skip "i" from potential isamplers. */
    if (*code == 'i') {
      code++;
    }
    /* Skip following spaces. */
    if (BLI_str_startswith(code, "sampler")) {
      /* Move past "uniform". */
      code += 7;
      /* Skip sampler type suffix. */
      while (!ELEM(*code, ' ', '\0')) {
        code++;
      }
      /* Skip following spaces. */
      while (*code == ' ') {
        code++;
      }

      if (*code != '\0') {
        char sampler_name[64];
        code = gpu_str_skip_token(code, sampler_name, sizeof(sampler_name));
        int id = GPU_shader_get_uniform(shader, sampler_name);

        if (id == -1) {
          continue;
        }
        /* Catch duplicates. */
        if (BLI_gset_add(sampler_ids, POINTER_FROM_INT(id))) {
          num_samplers++;
        }
      }
    }
  }

  BLI_gset_free(sampler_ids, nullptr);

  return num_samplers;
}

static bool gpu_pass_shader_validate(GPUPass *pass, GPUShader *shader)
{
  if (shader == nullptr) {
    return false;
  }

  /* NOTE: The only drawback of this method is that it will count a sampler
   * used in the fragment shader and only declared (but not used) in the vertex
   * shader as used by both. But this corner case is not happening for now. */
  int vert_samplers_len = count_active_texture_sampler(shader, pass->source.vertex);
  int frag_samplers_len = count_active_texture_sampler(shader, pass->source.fragment);

  int total_samplers_len = vert_samplers_len + frag_samplers_len;

  /* Validate against opengl limit. */
  if ((frag_samplers_len > GPU_max_textures_frag()) ||
      (vert_samplers_len > GPU_max_textures_vert())) {
    return false;
  }

  if (pass->source.geometry) {
    int geom_samplers_len = count_active_texture_sampler(shader, pass->source.geometry);
    total_samplers_len += geom_samplers_len;
    if (geom_samplers_len > GPU_max_textures_geom()) {
      return false;
    }
  }

  return (total_samplers_len <= GPU_max_textures());
}

bool GPU_pass_compile(GPUPass *pass, const char *shname)
{
  bool success = true;
  if (!pass->compiled) {
    GPUShader *shader = GPU_shader_create(pass->source.vertex,
                                          pass->source.fragment,
                                          pass->source.geometry,
                                          nullptr,
                                          pass->source.defines,
                                          shname);

    /* NOTE: Some drivers / gpu allows more active samplers than the opengl limit.
     * We need to make sure to count active samplers to avoid undefined behavior. */
    if (!gpu_pass_shader_validate(pass, shader)) {
      success = false;
      if (shader != nullptr) {
        fprintf(stderr, "GPUShader: error: too many samplers in shader.\n");
        GPU_shader_free(shader);
        shader = nullptr;
      }
    }
    pass->shader = shader;
    pass->compiled = true;
  }
  return success;
}

GPUShader *GPU_pass_shader_get(GPUPass *pass)
{
  return pass->shader;
}

void GPU_pass_release(GPUPass *pass)
{
  BLI_assert(pass->refcount > 0);
  pass->refcount--;
}

static void gpu_pass_free(GPUPass *pass)
{
  BLI_assert(pass->refcount == 0);
  if (pass->shader) {
    GPU_shader_free(pass->shader);
  }
  MEM_SAFE_FREE(pass->source.vertex);
  MEM_SAFE_FREE(pass->source.fragment);
  MEM_SAFE_FREE(pass->source.geometry);
  MEM_SAFE_FREE(pass->source.defines);
  MEM_freeN(pass);
}

void GPU_pass_cache_garbage_collect(void)
{
  static int lasttime = 0;
  const int shadercollectrate = 60; /* hardcoded for now. */
  int ctime = (int)PIL_check_seconds_timer();

  if (ctime < shadercollectrate + lasttime) {
    return;
  }

  lasttime = ctime;

  BLI_spin_lock(&pass_cache_spin);
  GPUPass *next, **prev_pass = &pass_cache;
  for (GPUPass *pass = pass_cache; pass; pass = next) {
    next = pass->next;
    if (pass->refcount == 0) {
      /* Remove from list */
      *prev_pass = next;
      gpu_pass_free(pass);
    }
    else {
      prev_pass = &pass->next;
    }
  }
  BLI_spin_unlock(&pass_cache_spin);
}

void GPU_pass_cache_init(void)
{
  BLI_spin_init(&pass_cache_spin);
}

void GPU_pass_cache_free(void)
{
  BLI_spin_lock(&pass_cache_spin);
  while (pass_cache) {
    GPUPass *next = pass_cache->next;
    gpu_pass_free(pass_cache);
    pass_cache = next;
  }
  BLI_spin_unlock(&pass_cache_spin);

  BLI_spin_end(&pass_cache_spin);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Module
 * \{ */

void gpu_codegen_init(void)
{
}

void gpu_codegen_exit(void)
{
  BKE_world_defaults_free_gpu();
  BKE_material_defaults_free_gpu();
  GPU_shader_free_builtin_shaders();
}

/** \} */