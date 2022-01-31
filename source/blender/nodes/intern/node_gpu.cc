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
 * The Original Code is Copyright (C) 2007 Blender Foundation.
 * All rights reserved.
 */

/** \file
 * \ingroup nodes
 */

#include "BLI_listbase.h"
#include "BLI_math_vector.h"

#include "DNA_node_types.h"

#include "GPU_material.h"

#include "node_exec.h"

#include "NOD_gpu.h"

static void nodestack_get_vec(float *in, short type_in, bNodeStack *ns)
{
  const float *from = ns->vec;

  if (type_in == SOCK_FLOAT) {
    if (ns->sockettype == SOCK_FLOAT) {
      *in = *from;
    }
    else {
      *in = (from[0] + from[1] + from[2]) / 3.0f;
    }
  }
  else if (type_in == SOCK_VECTOR) {
    if (ns->sockettype == SOCK_FLOAT) {
      in[0] = from[0];
      in[1] = from[0];
      in[2] = from[0];
    }
    else {
      copy_v3_v3(in, from);
    }
  }
  else { /* type_in==SOCK_RGBA */
    if (ns->sockettype == SOCK_RGBA) {
      copy_v4_v4(in, from);
    }
    else if (ns->sockettype == SOCK_FLOAT) {
      in[0] = from[0];
      in[1] = from[0];
      in[2] = from[0];
      in[3] = 1.0f;
    }
    else {
      copy_v3_v3(in, from);
      in[3] = 1.0f;
    }
  }
}

void node_gpu_stack_from_data(struct GPUNodeStack *gs, int type, bNodeStack *ns)
{
  memset(gs, 0, sizeof(*gs));

  if (ns == nullptr) {
    /* node_get_stack() will generate nullptr bNodeStack pointers
     * for unknown/unsupported types of sockets. */
    zero_v4(gs->vec);
    gs->link = nullptr;
    gs->type = GPU_NONE;
    gs->hasinput = false;
    gs->hasoutput = false;
    gs->sockettype = type;
  }
  else {
    nodestack_get_vec(gs->vec, type, ns);
    gs->link = (GPUNodeLink *)ns->data;

    if (type == SOCK_FLOAT) {
      gs->type = GPU_FLOAT;
    }
    else if (type == SOCK_INT) {
      gs->type = GPU_FLOAT; /* HACK: Support as float. */
    }
    else if (type == SOCK_VECTOR) {
      gs->type = GPU_VEC3;
    }
    else if (type == SOCK_RGBA) {
      gs->type = GPU_VEC4;
    }
    else if (type == SOCK_SHADER) {
      gs->type = GPU_CLOSURE;
    }
    else {
      gs->type = GPU_NONE;
    }

    gs->hasinput = ns->hasinput && ns->data;
    /* XXX Commented out the ns->data check here, as it seems it's not always set,
     *     even though there *is* a valid connection/output... But that might need
     *     further investigation.
     */
    gs->hasoutput = ns->hasoutput /*&& ns->data*/;
    gs->sockettype = ns->sockettype;
  }
}

void node_data_from_gpu_stack(bNodeStack *ns, GPUNodeStack *gs)
{
  copy_v4_v4(ns->vec, gs->vec);
  ns->data = gs->link;
  ns->sockettype = gs->sockettype;
}

static void gpu_stack_from_data_list(GPUNodeStack *gs, ListBase *sockets, bNodeStack **ns)
{
  int i;
  LISTBASE_FOREACH_INDEX (bNodeSocket *, socket, sockets, i) {
    node_gpu_stack_from_data(&gs[i], socket->type, ns[i]);
  }

  gs[i].end = true;
}

static void data_from_gpu_stack_list(ListBase *sockets, bNodeStack **ns, GPUNodeStack *gs)
{
  int i;
  LISTBASE_FOREACH_INDEX (bNodeSocket *, socket, sockets, i) {
    node_data_from_gpu_stack(ns[i], &gs[i]);
  }
}

void ntreeExecGPUNodes(bNodeTreeExec *exec, GPUMaterial *mat, bNode *output_node)
{
  bNodeExec *nodeexec;
  bNode *node;
  int n;
  bNodeStack *stack;
  bNodeStack *nsin[MAX_SOCKET];  /* arbitrary... watch this */
  bNodeStack *nsout[MAX_SOCKET]; /* arbitrary... watch this */
  GPUNodeStack gpuin[MAX_SOCKET + 1], gpuout[MAX_SOCKET + 1];
  bool do_it;

  stack = exec->stack;

  for (n = 0, nodeexec = exec->nodeexec; n < exec->totnodes; n++, nodeexec++) {
    node = nodeexec->node;

    do_it = false;
    /* for groups, only execute outputs for edited group */
    if (node->typeinfo->nclass == NODE_CLASS_OUTPUT) {
      if ((output_node != nullptr) && (node == output_node)) {
        do_it = true;
      }
    }
    else {
      do_it = true;
    }

    if (do_it) {
      if (node->typeinfo->gpu_fn) {
        node_get_stack(node, stack, nsin, nsout);
        gpu_stack_from_data_list(gpuin, &node->inputs, nsin);
        gpu_stack_from_data_list(gpuout, &node->outputs, nsout);
        if (node->typeinfo->gpu_fn(mat, node, &nodeexec->data, gpuin, gpuout)) {
          data_from_gpu_stack_list(&node->outputs, nsout, gpuout);
        }
      }
    }
  }
}
