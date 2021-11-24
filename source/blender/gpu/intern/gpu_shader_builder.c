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
 * Compile time automation of shader compilation and validation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gpu_shader_dependency_private.h"

#include "gpu_shader_descriptor.h"
#undef GPU_STAGE_INTERFACE_CREATE
#undef GPU_SHADER_DESCRIPTOR

/* Return 0 on success. (Recursive). */
static bool descriptor_flatten(const GPUShaderDescriptor *input, GPUShaderDescriptor *output)
{
  int errors = 0;

  if (output->name == NULL) {
    output->name = input->name;
  }

  if (input->do_static_compilation) {
    /* If one descriptor is valid, final one is too. */
    output->do_static_compilation = true;
  }

  if (input->vertex_source) {
    if (output->vertex_source) {
      printf("Error: %s.vertex_source cannot be overriden by %s.\n", output->name, input->name);
    }
    else {
      output->vertex_source = input->vertex_source;
    }
  }
  if (input->geometry_source) {
    if (output->geometry_source) {
      printf("Error: %s.geometry_source cannot be overriden by %s.\n", output->name, input->name);
    }
    else {
      output->geometry_source = input->geometry_source;
    }
  }
  if (input->fragment_source) {
    if (output->fragment_source) {
      printf("Error: %s.fragment_source cannot be overriden by %s.\n", output->name, input->name);
    }
    else {
      output->fragment_source = input->fragment_source;
    }
  }
  if (input->compute_source) {
    if (output->compute_source) {
      printf("Error: %s.compute_source cannot be overriden by %s.\n", output->name, input->name);
    }
    else {
      output->compute_source = input->compute_source;
    }
  }

  for (int i = 0; i < ARRAY_SIZE(input->defines); i++) {
    if (input->defines[i]) {
      if (output->defines[i]) {
        printf("Error: %s.defines[%d] cannot be overriden by %s.\n", output->name, i, input->name);
      }
      else {
        output->defines[i] = input->defines[i];
      }
    }
  }

  for (int i = 0; i < ARRAY_SIZE(input->vertex_inputs); i++) {
    if (input->vertex_inputs[i].type == UNUSED) {
      continue;
    }
    else if (output->vertex_inputs[i].type != UNUSED) {
      printf("Error: vertex_inputs[%d] is already used by \"%s\".\n",
             i,
             output->vertex_inputs[i].name);
      errors++;
    }
    else {
      output->vertex_inputs[i] = input->vertex_inputs[i];
    }
  }

  for (int i = 0; i < ARRAY_SIZE(input->fragment_outputs); i++) {
    if (input->fragment_outputs[i].type == UNUSED) {
      continue;
    }
    else if (output->fragment_outputs[i].type != UNUSED) {
      printf("Error: fragment_outputs[%d] is already used by \"%s\".\n",
             i,
             output->fragment_outputs[i].name);
      errors++;
    }
    else {
      output->fragment_outputs[i] = input->fragment_outputs[i];
    }
  }

  for (int i = 0; i < ARRAY_SIZE(input->vertex_out_interfaces); i++) {
    if (input->vertex_out_interfaces[i].inouts_len == 0) {
      continue;
    }
    else if (output->vertex_out_interfaces[i].inouts_len != 0) {
      printf("Error: vertex_out_interfaces[%d] is already used by \"%s\".\n",
             i,
             output->vertex_out_interfaces[i].name);
      errors++;
    }
    else {
      output->vertex_out_interfaces[i] = input->vertex_out_interfaces[i];
    }
  }

  for (int i = 0; i < ARRAY_SIZE(input->geometry_out_interfaces); i++) {
    if (input->geometry_out_interfaces[i].inouts_len == 0) {
      continue;
    }
    else if (output->geometry_out_interfaces[i].inouts_len != 0) {
      printf("Error: geometry_out_interfaces[%d] is already used by \"%s\".\n",
             i,
             output->geometry_out_interfaces[i].name);
      errors++;
    }
    else {
      output->geometry_out_interfaces[i] = input->geometry_out_interfaces[i];
    }
  }

  for (int i = 0; i < ARRAY_SIZE(input->resources); i++) {
    for (int j = 0; j < ARRAY_SIZE(input->resources[i]); j++) {
      if (input->resources[i][j].bind_type == UNUSED) {
        continue;
      }
      else if (output->resources[i][j].bind_type != UNUSED) {
        printf("Error: resources[%d][%d] is already used by \"%s\".\n",
               i,
               j,
               output->resources[i][j].name);
        errors++;
      }
      else {
        output->resources[i][j] = input->resources[i][j];
      }
    }
  }

  for (int i = 0; i < ARRAY_SIZE(input->push_constants); i++) {
    if (input->push_constants[i].type == UNUSED) {
      continue;
    }
    else if (output->push_constants[i].type != UNUSED) {
      printf("Error: push_constants[%d] is already used by \"%s\".\n",
             i,
             output->push_constants[i].name);
      errors++;
    }
    else {
      output->push_constants[i] = input->push_constants[i];
    }
  }

  if (errors) {
    printf("Error: Cannot merge %s into %s.\n", input->name, output->name);
    return errors;
  }

  for (int i = 0; i < ARRAY_SIZE(input->additional_descriptors); i++) {
    if (input->additional_descriptors[i]) {
      errors += descriptor_flatten(input->additional_descriptors[i], output);
    }
  }
  return errors;
}

#define ENUM_TO_STR(_enum) [_enum] = #_enum

static const char *inout_type_to_str(eGPUInOutType type)
{
  const char *strings[] = {
      ENUM_TO_STR(UNUSED),         ENUM_TO_STR(BOOL),          ENUM_TO_STR(FLOAT),
      ENUM_TO_STR(VEC2),           ENUM_TO_STR(VEC3),          ENUM_TO_STR(VEC4),
      ENUM_TO_STR(UINT),           ENUM_TO_STR(UVEC2),         ENUM_TO_STR(UVEC3),
      ENUM_TO_STR(UVEC4),          ENUM_TO_STR(INT),           ENUM_TO_STR(IVEC2),
      ENUM_TO_STR(IVEC3),          ENUM_TO_STR(IVEC4),         ENUM_TO_STR(FLOAT_1D),
      ENUM_TO_STR(FLOAT_1D_ARRAY), ENUM_TO_STR(FLOAT_2D),      ENUM_TO_STR(FLOAT_2D_ARRAY),
      ENUM_TO_STR(FLOAT_3D),       ENUM_TO_STR(INT_1D),        ENUM_TO_STR(INT_1D_ARRAY),
      ENUM_TO_STR(INT_2D),         ENUM_TO_STR(INT_2D_ARRAY),  ENUM_TO_STR(INT_3D),
      ENUM_TO_STR(UINT_1D),        ENUM_TO_STR(UINT_1D_ARRAY), ENUM_TO_STR(UINT_2D),
      ENUM_TO_STR(UINT_2D_ARRAY),  ENUM_TO_STR(UINT_3D),       ENUM_TO_STR(STRUCT),
  };
  return strings[type];
}

#undef ENUM_TO_STR

/* Write newlines as separated lines. */
static void write_str(FILE *fp,
                      const char *indent,
                      const char *iface_type,
                      const GPUShaderDescriptor *desc,
                      const char *const *src)
{
  (void)desc;
  (void)iface_type;
  const char *str = *src;
  int n_char = strcspn(str, "\n");
  do {
    if (str != *src) {
      fprintf(fp, "\n%s", indent);
    }
    fprintf(fp, "\"%.*s\\n\"", n_char, str);
    str += n_char + 1;
    n_char = strcspn(str, "\n");
  } while (n_char > 1);
}

static void write_inout(FILE *fp,
                        const char *indent,
                        const char *iface_type,
                        const GPUShaderDescriptor *desc,
                        const GPUInOut *inout)
{
  (void)indent;
  (void)desc;
  (void)iface_type;
  fprintf(fp, "{%s, \"%s\", %u}", inout_type_to_str(inout->type), inout->name, (uint)inout->qual);
}

static void write_interface_declaration(FILE *fp,
                                        const char *indent,
                                        const char *iface_type,
                                        const GPUShaderDescriptor *desc,
                                        const GPUInterfaceBlockDescription *iface)
{
  (void)indent;
  fprintf(fp, "GPUInOut %s_%s_%s[] = {\n", desc->name, iface_type, iface->name);
  for (int i = 0; i < iface->inouts_len; i++) {
    fprintf(fp, "  ");
    write_inout(fp, indent, iface_type, desc, &iface->inouts[i]);
    fprintf(fp, ",\n");
  }
  fprintf(fp, "};\n");
}

static void write_interface(FILE *fp,
                            const char *indent,
                            const char *iface_type,
                            const GPUShaderDescriptor *desc,
                            const GPUInterfaceBlockDescription *iface)
{
  (void)indent;
  if (iface->inouts_len > 0) {
    fprintf(
        fp, "STAGE_INTERFACE(\"%s\", %s_%s_%s)", iface->name, desc->name, iface_type, iface->name);
  }
  else {
    fprintf(fp, "{NULL, 0, NULL}");
  }
}

#define write_array(fp, desc, array, array_name, _used, write_function, iface_type, indent) \
  { \
    int used_slots = 0; \
    for (int i = 0; i < ARRAY_SIZE(array); i++) { \
      used_slots += (array[i] _used != UNUSED); \
    } \
    if (used_slots) { \
      if (#array_name[0] != '\0') { \
        fprintf(fp, "%s  " #array_name " = ", indent); \
      } \
      else { \
        fprintf(fp, "%s  ", indent); \
      } \
      fprintf(fp, "{\n"); \
      for (int i = 0; i < ARRAY_SIZE(array); i++) { \
        if (array[i] _used != UNUSED) { \
          fprintf(fp, "%s    [%d] = ", indent, i); \
          write_function(fp, indent, iface_type, desc, &array[i]); \
          fprintf(fp, ",\n"); \
        } \
      } \
      fprintf(fp, "%s  },\n", indent); \
    } \
  }

static void write_resource_bind(FILE *fp,
                                const char *indent,
                                const char *iface_type,
                                const GPUShaderDescriptor *desc,
                                const GPUResourceBind *bind)
{
  (void)indent;
  (void)desc;
  (void)iface_type;
  fprintf(fp,
          "{%u, %u, %u, %u, \"%s\", \"%s\"}",
          (uint)bind->bind_type,
          (uint)bind->type,
          (uint)bind->qual,
          (uint)bind->sampler,
          bind->type_name ? bind->type_name : "",
          bind->name);
}

static void write_descriptor(FILE *fp, const GPUShaderDescriptor *desc)
{
  fprintf(fp, "extern const GPUShaderDescriptor %s;\n", desc->name);

  for (int i = 0; i < ARRAY_SIZE(desc->vertex_out_interfaces); i++) {
    if (desc->vertex_out_interfaces[i].inouts_len != 0) {
      write_interface_declaration(fp, "", "vert", desc, &desc->vertex_out_interfaces[i]);
    }
  }
  for (int i = 0; i < ARRAY_SIZE(desc->geometry_out_interfaces); i++) {
    if (desc->geometry_out_interfaces[i].inouts_len != 0) {
      write_interface_declaration(fp, "", "geom", desc, &desc->geometry_out_interfaces[i]);
    }
  }

  fprintf(fp, "const GPUShaderDescriptor %s = {\n", desc->name);
  write_array(fp, desc, desc->vertex_inputs, .vertex_inputs, .type, write_inout, "", "");
  write_array(fp, desc, desc->fragment_outputs, .fragment_outputs, .type, write_inout, "", "");
  write_array(fp,
              desc,
              desc->vertex_out_interfaces,
              .vertex_out_interfaces,
              .inouts_len,
              write_interface,
              "vert",
              "");
  write_array(fp,
              desc,
              desc->geometry_out_interfaces,
              .geometry_out_interfaces,
              .inouts_len,
              write_interface,
              "frag",
              "");
  {
    int used_slots_1 = 0;
    for (int i = 0; i < ARRAY_SIZE(desc->resources); i++) {
      for (int j = 0; j < ARRAY_SIZE(desc->resources[i]); j++) {
        used_slots_1 += (desc->resources[i][j].type != UNUSED);
      }
    }
    if (used_slots_1) {
      fprintf(fp, "  .resources = {\n");
      write_array(
          fp, desc, desc->resources[0], [DESCRIPTOR_SET_0], .type, write_resource_bind, "", "  ");
      write_array(
          fp, desc, desc->resources[1], [DESCRIPTOR_SET_1], .type, write_resource_bind, "", "  ");
      fprintf(fp, "  },\n");
    }
  }
  write_array(fp, desc, desc->push_constants, .push_constants, .type, write_inout, "", "");
  write_array(fp, desc, desc->defines, .defines, , write_str, "", "");

  if (desc->vertex_source) {
    fprintf(fp, "  .vertex_source = \"%s\",\n", desc->vertex_source);
  }
  if (desc->geometry_source) {
    fprintf(fp, "  .geometry_source = \"%s\",\n", desc->geometry_source);
  }
  if (desc->fragment_source) {
    fprintf(fp, "  .fragment_source = \"%s\",\n", desc->fragment_source);
  }
  if (desc->compute_source) {
    fprintf(fp, "  .compute_source = \"%s\",\n", desc->compute_source);
  }

  fprintf(fp, "};\n\n");
}

int main(int argc, char const *argv[])
{
  if (argc < 1) {
    printf("Usage: shader_builder <data_file_to>\n");
    exit(1);
  }

  size_t descriptors_len = 0;
/* Count number of descriptors. */
#define GPU_STAGE_INTERFACE_CREATE(_interface, ...)
#define GPU_SHADER_DESCRIPTOR(_descriptor, ...) descriptors_len++;
#include "gpu_shader_descriptor_list.h"
#undef GPU_STAGE_INTERFACE_CREATE
#undef GPU_SHADER_DESCRIPTOR

  GPUShaderDescriptor **descriptors = calloc(descriptors_len, sizeof(void *));

  size_t index = 0;
/* Declare everything first to be able to avoid dependency for references. */
#define GPU_STAGE_INTERFACE_CREATE(_interface, ...) GPUInOut _interface[] = __VA_ARGS__;
#define GPU_SHADER_DESCRIPTOR(_descriptor, ...) \
  GPUShaderDescriptor _descriptor; \
  descriptors[index++] = &_descriptor;
#include "gpu_shader_descriptor_list.h"
#undef GPU_STAGE_INTERFACE_CREATE
#undef GPU_SHADER_DESCRIPTOR

/* Set values. */
#define GPU_STAGE_INTERFACE_CREATE(_interface, ...)
#define GPU_SHADER_DESCRIPTOR(_descriptor, ...) \
  _descriptor = (GPUShaderDescriptor)__VA_ARGS__; \
  _descriptor.name = #_descriptor;
#include "gpu_shader_descriptor_list.h"
#undef GPU_STAGE_INTERFACE_CREATE
#undef GPU_SHADER_DESCRIPTOR

  FILE *fp = fopen(argv[1], "w");

  fprintf(fp, "#include \"gpu_shader_descriptor.h\"\n");

#if 0 /* TEST */
  gpu_shader_dependency_init();
#endif

  int result = 0;
  for (size_t i = 0; i < descriptors_len; i++) {
    const GPUShaderDescriptor *descriptor = descriptors[i];

    GPUShaderDescriptor flattened_descriptor = {0};

    int errors = descriptor_flatten(descriptor, &flattened_descriptor);
    if (errors != 0) {
      result = 1;
      continue;
    }

    descriptor = &flattened_descriptor;

    /* TODO(fclem): Validate push constant alignment & overlapping. */

    if (descriptor->do_static_compilation) {
#if 0 /* TODO */
      descriptor->vulkan_spirv = vk_shader_compile(descriptor);
      if (descriptor->vulkan_spirv == nullptr) {
        result = 1;
      }
#endif
#if 0 /* TODO */
      if (gl_shader_validate(descriptor) == false) {
        result = 1;
      }
#endif
#if 0 /* TEST */
      if (descriptor->vertex_source) {
        char *src = gpu_shader_dependency_get_resolved_source(descriptor->vertex_source);

        FILE *test_fp = fopen(descriptor->vertex_source, "w");
        fprintf(test_fp, "%s", src);
        fclose(test_fp);

        free(src);
      }
#endif
    }
    write_descriptor(fp, descriptor);
  }

#if 0 /* TEST */
  gpu_shader_dependency_exit();
#endif

  fclose(fp);

  free(descriptors);

  return result;
}
