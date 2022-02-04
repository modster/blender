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
 * Copyright 2021, Blender Foundation.
 */

/** \file
 * \ingroup draw_engine
 *
 * Engine processing the render buffer using GLSL to apply the scene compositing node tree.
 */

#include "DRW_render.h"

#include "BLI_map.hh"
#include "BLI_string_ref.hh"
#include "BLI_utildefines.h"
#include "BLI_vector.hh"
#include "BLI_vector_set.hh"

#include "DNA_node_types.h"
#include "DNA_scene_types.h"

#include "GPU_texture.h"

#include "IMB_colormanagement.h"

#include "NOD_compositor_execute.hh"
#include "NOD_derived_node_tree.hh"

#include "compositor_shader.hh"

namespace blender::compositor {

using nodes::CompositorContext;
using namespace nodes::derived_node_tree_types;

class DRWCompositorContext : public CompositorContext {
 private:
  /* The node currently being executed. */
  DNode node_;
  /* A map associating output sockets with the textures storing their contents. The map only stores
   * the textures that were already computed by a dependency node and are still needed by one or
   * more dependent nodes, so the node currently executing can get its inputs and outputs from this
   * member. See get_input_texture and get_output_texture. */
  const Map<DSocket, GPUTexture *> &allocated_textures_;

 public:
  DRWCompositorContext(DNode node, const Map<DSocket, GPUTexture *> &allocated_textures)
      : node_(node), allocated_textures_(allocated_textures)
  {
  }

  const GPUTexture *get_input_texture(StringRef identifier) override
  {
    /* Find the output socket connected to the input with the given input identifier and return its
     * allocated texture. If the input is not linked, return nullptr. */
    GPUTexture *texture = nullptr;
    node_.input_by_identifier(identifier).foreach_origin_socket([&](const DSocket origin) {
      texture = allocated_textures_.lookup(origin);
    });
    return texture;
  }

  const GPUTexture *get_output_texture(StringRef identifier) override
  {
    return allocated_textures_.lookup(node_.output_by_identifier(identifier));
  }

  const GPUTexture *get_viewport_texture() override
  {
    return DRW_viewport_texture_list_get()->color;
  }

  const GPUTexture *get_pass_texture(int view_layer, eScenePassType pass_type) override
  {
    return DRW_render_pass_find(DRW_context_state_get()->scene, view_layer, pass_type)->pass_tx;
  }

  const bNode &node() override
  {
    return *node_->bnode();
  }
};

class Compiler {
 public:
 private:
  /* The derived and reference node trees repressing the compositor setup. */
  NodeTreeRefMap tree_ref_map_;
  DerivedNodeTree tree_;
  /* The output node whose result should be computed and drawn. */
  DNode output_node_;
  /* Stores a heuristic estimation of the number of needed intermediate buffers
   * to compute every node and all of its dependencies. */
  Map<DNode, int> needed_buffers_;
  /* An ordered set of nodes defining the schedule of node execution. */
  VectorSet<DNode> node_schedule_;

 public:
  Compiler(bNodeTree *scene_node_tree) : tree_(*scene_node_tree, tree_ref_map_){};

  void compile()
  {
    compute_output_node();
    compute_needed_buffers(output_node_);
    compute_schedule(output_node_);
  }

  void dump_schedule()
  {
    for (const DNode &node : node_schedule_) {
      std::cout << node->name() << std::endl;
    }
  }

 private:
  /* Computes the output node whose result should be computed and drawn. The output node is the
   * node marked as NODE_DO_OUTPUT. If multiple types of output nodes are marked, then the
   * preference will be CMP_NODE_COMPOSITE > CMP_NODE_VIEWER > CMP_NODE_SPLITVIEWER. */
  void compute_output_node()
  {
    const NodeTreeRef &root_tree = tree_.root_context().tree();
    for (const NodeRef *node : root_tree.nodes_by_type("CompositorNodeComposite")) {
      if (node->bnode()->flag & NODE_DO_OUTPUT) {
        output_node_ = DNode(&tree_.root_context(), node);
        return;
      }
    }
    for (const NodeRef *node : root_tree.nodes_by_type("CompositorNodeViewer")) {
      if (node->bnode()->flag & NODE_DO_OUTPUT) {
        output_node_ = DNode(&tree_.root_context(), node);
        return;
      }
    }
    for (const NodeRef *node : root_tree.nodes_by_type("CompositorNodeSplitViewer")) {
      if (node->bnode()->flag & NODE_DO_OUTPUT) {
        output_node_ = DNode(&tree_.root_context(), node);
        return;
      }
    }
  }

  /* Computes a heuristic estimation of the number of needed intermediate buffers to compute this
   * node and all of its dependencies. The method recursively computes the needed buffers for all
   * node dependencies and stores them in the needed_buffers_ map. So the root/output node can be
   * provided to compute the needed buffers for all nodes.
   *
   * Consider a node that takes n number of buffers as an input from a number of node dependencies,
   * which we shall call the input nodes. The node also computes and outputs m number of buffers.
   * In order for the node to compute its output, a number of intermediate buffers will be needed.
   * Since the node takes n buffers and outputs m buffers, then the number of buffers directly
   * needed by the node is (n + m). But each of the input buffers are computed by a node that, in
   * turn, needs a number of buffers to compute its output. So the total number of buffers needed
   * to compute the output of the node is max(n + m, d) where d is the number of buffers needed by
   * the input node that needs the largest number of buffers. We only consider the input node that
   * needs the largest number of buffers, because those buffers can be reused by any input node
   * that needs a lesser number of buffers.
   *
   * If the node tree was, in fact, a tree, then this would be an accurate computation. However,
   * the node tree is in fact a graph that allows output sharing, so the computation in this case
   * is merely a heuristic estimation that works well in most cases. */
  int compute_needed_buffers(DNode node)
  {
    /* Compute the number of buffers that the node takes as an input as well as the number of
     * buffers needed to compute the most demanding dependency node. */
    int input_buffers = 0;
    int buffers_needed_by_dependencies = 0;
    for (const InputSocketRef *input_ref : node->inputs()) {
      const DInputSocket input{node.context(), input_ref};
      /* Only consider inputs that are linked, that is, those that take a buffer. */
      input.foreach_origin_socket([&](const DSocket origin) {
        input_buffers++;
        /* The origin node was already computed before, so skip it. */
        if (needed_buffers_.contains(origin.node())) {
          return;
        }
        /* Recursively compute the number of buffers needed to compute this dependency node. */
        const int buffers_needed_by_origin = compute_needed_buffers(origin.node());
        if (buffers_needed_by_origin > buffers_needed_by_dependencies) {
          buffers_needed_by_dependencies = buffers_needed_by_origin;
        }
      });
    }

    /* Compute the number of buffers that will be computed/output by this node. */
    int output_buffers = 0;
    for (const OutputSocketRef *output : node->outputs()) {
      if (output->logically_linked_sockets().size() != 0) {
        output_buffers++;
      }
    }

    /* Compute the heuristic estimation of the number of needed intermediate buffers to compute
     * this node and all of its dependencies. */
    const int total_buffers = MAX2(input_buffers + output_buffers, buffers_needed_by_dependencies);
    needed_buffers_.add_new(node, total_buffers);
    return total_buffers;
  }

  /* Computes the most optimal execution schedule of the nodes and stores the schedule in
   * node_schedule_. This is essentially a post-order depth first traversal of the node tree from
   * the output node to the leaf input nodes, with informed order of traversal of children.
   *
   * There are multiple different possible orders of evaluating a node graph, each of which needs
   * to allocate a number of intermediate buffers to store its intermediate results. It follows
   * that we need to find the evaluation order which uses the least amount of intermediate buffers.
   * For instance, consider a node that takes two input buffers A and B. Each of those buffers is
   * computed through a number of nodes constituting a sub-graph whose root is the node that
   * outputs that buffer. Suppose the number of intermediate buffers needed to compute A and B are
   * N(A) and N(B) respectively and N(A) > N(B). Then evaluating the sub-graph computing A would be
   * a better option than that of B, because had B was computed first, its outputs will need to be
   * stored in extra buffers in addition to the buffers needed by A.
   *
   * This is a heuristic generalization of the Sethi–Ullman algorithm, a generalization that
   * doesn't always guarantee an optimal evaluation order, as the optimal evaluation order is very
   * difficult to compute, however, this method works well in most cases. */
  void compute_schedule(DNode node)
  {
    /* Compute the nodes directly connected to the node inputs sorted by their needed buffers such
     * that the node with the highest number of needed buffers comes first. */
    Vector<DNode> sorted_origin_nodes;
    for (const InputSocketRef *input_ref : node->inputs()) {
      const DInputSocket input{node.context(), input_ref};
      input.foreach_origin_socket([&](const DSocket origin) {
        /* The origin node was added before or was already schedule, so skip it. The number of
         * origin nodes is very small, so linear search is okay.
         */
        if (sorted_origin_nodes.contains(origin.node()) ||
            node_schedule_.contains(origin.node())) {
          return;
        }
        /* Sort on insertion, the number of origin nodes is very small, so this is okay. */
        int insertion_position = 0;
        for (int i = 0; i < sorted_origin_nodes.size(); i++) {
          if (needed_buffers_.lookup(origin.node()) >
              needed_buffers_.lookup(sorted_origin_nodes[i])) {
            break;
          }
          insertion_position++;
        }
        sorted_origin_nodes.insert(insertion_position, origin.node());
      });
    }

    /* Recursively schedule origin nodes. Nodes with higher number of needed intermediate buffers
     * are scheduled first. */
    for (const DNode &origin_node : sorted_origin_nodes) {
      compute_schedule(origin_node);
    }

    node_schedule_.add_new(node);
  }
};

/* Keep in sync with CompositorData in compositor_lib.glsl. */
typedef struct CompositorData {
  float luminance_coefficients[3];
  float frame_number;
} CompositorData;

BLI_STATIC_ASSERT_ALIGN(CompositorData, 16)

static ShaderModule *g_shader_module = nullptr;

class Instance {
 public:
  ShaderModule &shaders;

 private:
  /** TODO(fclem) multipass. */
  DRWPass *pass_;
  /** A UBO storing CompositorData. */
  GPUUniformBuf *ubo_;
  GPUMaterial *gpumat_;
  /** Temp buffers to hold intermediate results or the input color. */
  GPUTexture *tmp_buffer_ = nullptr;
  GPUFrameBuffer *tmp_fb_ = nullptr;

  bool enabled_;

 public:
  Instance(ShaderModule &shader_module) : shaders(shader_module)
  {
    ubo_ = GPU_uniformbuf_create_ex(sizeof(CompositorData), &ubo_, "CompositorData");
  };

  ~Instance()
  {
    GPU_uniformbuf_free(ubo_);
    GPU_FRAMEBUFFER_FREE_SAFE(tmp_fb_);
  }

  void init()
  {
    const DRWContextState *ctx_state = DRW_context_state_get();
    Scene *scene = ctx_state->scene;
    enabled_ = scene->use_nodes && scene->nodetree;

    if (!enabled_) {
      return;
    }

    gpumat_ = shaders.material_get(scene);
    enabled_ = GPU_material_status(gpumat_) == GPU_MAT_SUCCESS;

    if (!enabled_) {
      return;
    }

    /* Create temp double buffer to render to or copy source to. */
    /* TODO(fclem) with multipass compositing we might need more than one temp buffer. */
    DrawEngineType *owner = (DrawEngineType *)g_shader_module;
    eGPUTextureFormat format = GPU_texture_format(DRW_viewport_texture_list_get()->color);
    tmp_buffer_ = DRW_texture_pool_query_fullscreen(format, owner);

    GPU_framebuffer_ensure_config(&tmp_fb_,
                                  {
                                      GPU_ATTACHMENT_NONE,
                                      GPU_ATTACHMENT_TEXTURE(tmp_buffer_),
                                  });
  }

  void sync()
  {
    if (!enabled_) {
      return;
    }

    pass_ = DRW_pass_create("Compositing", DRW_STATE_WRITE_COLOR);
    DRWShadingGroup *grp = DRW_shgroup_material_create(gpumat_, pass_);

    sync_compositor_ubo(grp);

    ListBase rpasses = GPU_material_render_passes(gpumat_);
    LISTBASE_FOREACH (GPUMaterialRenderPass *, gpu_rp, &rpasses) {
      DRWRenderPass *drw_rp = DRW_render_pass_find(
          gpu_rp->scene, gpu_rp->viewlayer, gpu_rp->pass_type);
      if (drw_rp) {
        DRW_shgroup_uniform_texture_ex(
            grp, gpu_rp->sampler_name, drw_rp->pass_tx, gpu_rp->sampler_state);
      }
    }

    DRW_shgroup_call_procedural_triangles(grp, nullptr, 1);
  }

  void draw()
  {
    if (!enabled_) {
      return;
    }

    DefaultTextureList *dtxl = DRW_viewport_texture_list_get();

    /* Reset default view. */
    DRW_view_set_active(nullptr);

    GPU_framebuffer_bind(tmp_fb_);
    DRW_draw_pass(pass_);

    /* TODO(fclem) only copy if we need to. Only possible in multipass.
     * This is because dtxl->color can also be an input to the compositor. */
    GPU_texture_copy(dtxl->color, tmp_buffer_);
  }

 private:
  void sync_compositor_ubo(DRWShadingGroup *shading_group)
  {
    CompositorData compositor_data;
    IMB_colormanagement_get_luminance_coefficients(compositor_data.luminance_coefficients);
    compositor_data.frame_number = (float)DRW_context_state_get()->scene->r.cfra;

    GPU_uniformbuf_update(ubo_, &compositor_data);
    DRW_shgroup_uniform_block(shading_group, "compositor_block", ubo_);
  }
};

}  // namespace blender::compositor

/* -------------------------------------------------------------------- */
/** \name C interface
 * \{ */

using namespace blender::compositor;

typedef struct COMPOSITOR_Data {
  DrawEngineType *engine_type;
  DRWViewportEmptyList *fbl;
  DRWViewportEmptyList *txl;
  DRWViewportEmptyList *psl;
  DRWViewportEmptyList *stl;
  Instance *instance_data;
} COMPOSITOR_Data;

static void compositor_engine_init(void *vedata)
{
  COMPOSITOR_Data *ved = (COMPOSITOR_Data *)vedata;

  if (g_shader_module == nullptr) {
    /* TODO(fclem) threadsafety. */
    g_shader_module = new ShaderModule();
  }

  if (ved->instance_data == nullptr) {
    ved->instance_data = new Instance(*g_shader_module);
  }

  ved->instance_data->init();
}

static void compositor_engine_free(void)
{
  delete g_shader_module;
  g_shader_module = nullptr;
}

static void compositor_instance_free(void *instance_data_)
{
  Instance *instance_data = reinterpret_cast<Instance *>(instance_data_);
  delete instance_data;
}

static void compositor_cache_init(void *vedata)
{
  COMPOSITOR_Data *ved = (COMPOSITOR_Data *)vedata;
  ved->instance_data->sync();
}

static void compositor_draw(void *vedata)
{
  COMPOSITOR_Data *ved = (COMPOSITOR_Data *)vedata;
  ved->instance_data->draw();
}

/** \} */

extern "C" {

static const DrawEngineDataSize compositor_data_size = DRW_VIEWPORT_DATA_SIZE(COMPOSITOR_Data);

DrawEngineType draw_engine_compositor_type = {
    nullptr,
    nullptr,
    N_("Compositor"),
    &compositor_data_size,
    &compositor_engine_init,
    &compositor_engine_free,
    &compositor_instance_free,
    &compositor_cache_init,
    nullptr,
    nullptr,
    &compositor_draw,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};
}
