/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2021-2022 Intel Corporation */

#ifdef WITH_ONEAPI

// clang-format off
#  include "kernel.h"
#  include <iostream>
#  include <map>
#  include <set>

#  include <level_zero/ze_api.h>
#  include <ext/oneapi/backend/level_zero.hpp>

#  include "kernel/device/oneapi/compat.h"
#  include "kernel/device/oneapi/device_id.h"
#  include "kernel/device/oneapi/globals.h"
#  include "kernel/device/oneapi/kernel_templates.h"

#  include "kernel/device/gpu/kernel.h"
// clang-format on

static OneAPIErrorCallback s_error_cb = nullptr;
static void *s_error_user_ptr = nullptr;

static std::vector<sycl::device> oneapi_available_devices(
    std::map<std::string, std::string> *proper_names_map = nullptr);

void oneapi_set_error_cb(OneAPIErrorCallback cb, void *user_ptr)
{
  s_error_cb = cb;
  s_error_user_ptr = user_ptr;
}

bool is_gpu(sycl::queue *queue);

bool is_gpu(sycl::queue *queue)
{
  sycl::info::device_type device_type =
      queue->get_device().get_info<sycl::info::device::device_type>();
  return device_type == sycl::info::device_type::gpu;
}

bool is_cpu(sycl::queue *queue);

bool is_cpu(sycl::queue *queue)
{
  sycl::info::device_type device_type =
      queue->get_device().get_info<sycl::info::device::device_type>();
  return device_type == sycl::info::device_type::cpu;
}

void check_usm(SyclQueue *queue_, const void *usm_ptr, bool allow_host = false);

void check_usm(SyclQueue *queue_, const void *usm_ptr, bool allow_host)
{
#  ifdef _DEBUG
  sycl::queue *queue = reinterpret_cast<sycl::queue *>(queue_);
  sycl::usm::alloc usm_type = get_pointer_type(usm_ptr, queue->get_context());
  (void)usm_type;
  assert(
      usm_type == sycl::usm::alloc::device ||
      ((queue->is_host() || is_cpu(queue) || allow_host) && usm_type == sycl::usm::alloc::host));
#  endif
}

bool oneapi_create_queue(SyclQueue *&external_queue, int device_index)
{
  bool finished_correct = true;
  try {
    std::vector<sycl::device> devices = oneapi_available_devices();
    if (device_index < 0 || device_index >= devices.size()) {
      return false;
    }
    sycl::queue *created_queue = new sycl::queue(devices[device_index],
                                                 sycl::property::queue::in_order());
    external_queue = reinterpret_cast<SyclQueue *>(created_queue);
  }
  catch (sycl::exception const &e) {
    finished_correct = false;
    if (s_error_cb) {
      s_error_cb(e.what(), s_error_user_ptr);
    }
  }
  return finished_correct;
}

void oneapi_free_queue(SyclQueue *queue_)
{
  assert(queue_);
  sycl::queue *queue = reinterpret_cast<sycl::queue *>(queue_);
  delete queue;
}

void *oneapi_usm_aligned_alloc_host(SyclQueue *queue_, size_t memory_size, size_t alignment)
{
  assert(queue_);
  sycl::queue *queue = reinterpret_cast<sycl::queue *>(queue_);
  return sycl::aligned_alloc_host(alignment, memory_size, *queue);
}

void *oneapi_usm_alloc_device(SyclQueue *queue_, size_t memory_size)
{
  assert(queue_);
  sycl::queue *queue = reinterpret_cast<sycl::queue *>(queue_);
  return sycl::malloc_device(memory_size, *queue);
}

void oneapi_usm_free(SyclQueue *queue_, void *usm_ptr)
{
  assert(queue_);
  sycl::queue *queue = reinterpret_cast<sycl::queue *>(queue_);
  check_usm(queue_, usm_ptr, true);
  sycl::free(usm_ptr, *queue);
}

bool oneapi_usm_memcpy(SyclQueue *queue_, void *dest, void *src, size_t num_bytes)
{
  assert(queue_);
  sycl::queue *queue = reinterpret_cast<sycl::queue *>(queue_);
  check_usm(queue_, dest, true);
  check_usm(queue_, src, true);
  try {
    sycl::event mem_event = queue->memcpy(dest, src, num_bytes);
    mem_event.wait_and_throw();
    return true;
  }
  catch (sycl::exception const &e) {
    if (s_error_cb) {
      s_error_cb(e.what(), s_error_user_ptr);
    }
    return false;
  }
}

bool oneapi_usm_memset(SyclQueue *queue_, void *usm_ptr, unsigned char value, size_t num_bytes)
{
  assert(queue_);
  sycl::queue *queue = reinterpret_cast<sycl::queue *>(queue_);
  check_usm(queue_, usm_ptr, true);
  try {
    sycl::event mem_event = queue->memset(usm_ptr, value, num_bytes);
    mem_event.wait_and_throw();
    return true;
  }
  catch (sycl::exception const &e) {
    if (s_error_cb) {
      s_error_cb(e.what(), s_error_user_ptr);
    }
    return false;
  }
}

bool oneapi_queue_synchronize(SyclQueue *queue_)
{
  assert(queue_);
  sycl::queue *queue = reinterpret_cast<sycl::queue *>(queue_);
  try {
    queue->wait_and_throw();
    return true;
  }
  catch (sycl::exception const &e) {
    if (s_error_cb) {
      s_error_cb(e.what(), s_error_user_ptr);
    }
    return false;
  }
}

bool oneapi_trigger_runtime_compilation(SyclQueue *queue_)
{
  // NOTE(sirgienko) Execution of this simple kernel will check basic functionality and
  // also trigger runtime compilation of all existing oneAPI kernels
  assert(queue_);
  sycl::queue *queue = reinterpret_cast<sycl::queue *>(queue_);
  size_t N = 8;
  sycl::buffer<float, 1> A(N);
  sycl::buffer<float, 1> B(N);

  {
    sycl::host_accessor A_host_acc(A, sycl::write_only);
    for (size_t i = (size_t)0; i < N; i++)
      A_host_acc[i] = rand() % 32;
  }

  try {
    queue->submit([&](sycl::handler &cgh) {
      sycl::accessor A_acc(A, cgh, sycl::read_only, sycl::no_init);
      sycl::accessor B_acc(B, cgh, sycl::write_only, sycl::no_init);

      cgh.parallel_for(N, [=](sycl::id<1> idx) { B_acc[idx] = A_acc[idx] + idx.get(0); });
    });
    queue->wait_and_throw();

    sycl::host_accessor A_host_acc(A, sycl::read_only);
    sycl::host_accessor B_host_acc(B, sycl::read_only);

    for (size_t i = (size_t)0; i < N; i++) {
      float result = A_host_acc[i] + B_host_acc[i];
      (void)result;
    }
  }
  catch (sycl::exception const &e) {
    if (s_error_cb) {
      s_error_cb(e.what(), s_error_user_ptr);
    }
    return false;
  }

  return true;
}

bool oneapi_kernel_globals_size(SyclQueue *queue_, size_t &kernel_global_size)
{
  kernel_global_size = sizeof(KernelGlobalsGPU);

  return true;
}

void oneapi_set_global_memory(SyclQueue *queue_,
                              void *kernel_globals,
                              const char *memory_name,
                              void *memory_device_pointer)
{
  assert(queue_);
  assert(kernel_globals);
  assert(memory_name);
  assert(memory_device_pointer);
  KernelGlobalsGPU *globals = (KernelGlobalsGPU *)kernel_globals;
  check_usm(queue_, memory_device_pointer);
  check_usm(queue_, kernel_globals, true);

  std::string matched_name(memory_name);

// This macros will change global ptr of KernelGlobals via name matching
#  define KERNEL_TEX(type, name) \
    else if (#name == matched_name) \
    { \
      globals->name = (type *)memory_device_pointer; \
      return; \
    }
  if (false) {
  }
  else if ("__integrator_state" == matched_name) {
    globals->integrator_state = (IntegratorStateGPU *)memory_device_pointer;
    return;
  }
  KERNEL_TEX(KernelData, __data)
#  include "kernel/textures.h"
  else
  {
    std::cerr << "Can't found global/constant memory with name \"" << matched_name << "\"!"
              << std::endl;
    assert(false);
  }
#  undef KERNEL_TEX
}

size_t oneapi_kernel_prefered_local_size(SyclQueue *queue_,
                                         const DeviceKernel kernel,
                                         const size_t kernel_global_size)
{
  assert(queue_);
  sycl::queue *queue = reinterpret_cast<sycl::queue *>(queue_);
  (void)kernel_global_size;
  const static size_t prefered_work_group_size_intersect_shading = 32;
  const static size_t prefered_work_group_size_technical = 1024;

  size_t prefered_work_group_size = 0;
  switch (kernel) {
    case DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA:
    case DEVICE_KERNEL_INTEGRATOR_INIT_FROM_BAKE:
    case DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST:
    case DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW:
    case DEVICE_KERNEL_INTEGRATOR_INTERSECT_SUBSURFACE:
    case DEVICE_KERNEL_INTEGRATOR_INTERSECT_VOLUME_STACK:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_RAYTRACE:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW:
      prefered_work_group_size = prefered_work_group_size_intersect_shading;
      break;

    case DEVICE_KERNEL_INTEGRATOR_QUEUED_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_QUEUED_SHADOW_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_ACTIVE_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_TERMINATED_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_SORTED_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_COMPACT_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_COMPACT_STATES:
    case DEVICE_KERNEL_INTEGRATOR_TERMINATED_SHADOW_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_COMPACT_SHADOW_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_COMPACT_SHADOW_STATES:
    case DEVICE_KERNEL_INTEGRATOR_RESET:
    case DEVICE_KERNEL_INTEGRATOR_SHADOW_CATCHER_COUNT_POSSIBLE_SPLITS:
      prefered_work_group_size = prefered_work_group_size_technical;
      break;

    default:
      prefered_work_group_size = 512;
  }

  const size_t limit_work_group_size =
      queue->get_device().get_info<sycl::info::device::max_work_group_size>();
  return std::min(limit_work_group_size, prefered_work_group_size);
}

bool oneapi_enqueue_kernel(KernelContext *kernel_context,
                           int kernel,
                           size_t global_size,
                           void **args)
{
  bool success = true;
  ::DeviceKernel device_kernel = (::DeviceKernel)kernel;
  KernelGlobalsGPU *kg = (KernelGlobalsGPU *)kernel_context->kernel_globals;
  sycl::queue *queue = reinterpret_cast<sycl::queue *>(kernel_context->queue);
  assert(queue);
  if (!queue) {
    return false;
  }

  size_t local_size = oneapi_kernel_prefered_local_size(
      kernel_context->queue, device_kernel, global_size);
  assert(global_size % local_size == 0);

  if (device_kernel == DEVICE_KERNEL_INTEGRATOR_ACTIVE_PATHS_ARRAY ||
      device_kernel == DEVICE_KERNEL_INTEGRATOR_QUEUED_PATHS_ARRAY ||
      device_kernel == DEVICE_KERNEL_INTEGRATOR_QUEUED_SHADOW_PATHS_ARRAY ||
      device_kernel == DEVICE_KERNEL_INTEGRATOR_TERMINATED_PATHS_ARRAY ||
      device_kernel == DEVICE_KERNEL_INTEGRATOR_TERMINATED_SHADOW_PATHS_ARRAY ||
      device_kernel == DEVICE_KERNEL_INTEGRATOR_COMPACT_PATHS_ARRAY ||
      device_kernel == DEVICE_KERNEL_INTEGRATOR_COMPACT_SHADOW_PATHS_ARRAY) {
    int num_states = *((int *)(args[0]));
    size_t groups_count = num_states / local_size + (num_states % local_size == 0 ? 0 : 1);
    /* NOTE(sirgienko) Because for now non-uniform workgroups don't work on most of
       oneAPI devices,here ise xtending of work size to match uniform requirements  */
    global_size = groups_count * local_size;

    if (device_kernel == DEVICE_KERNEL_INTEGRATOR_ACTIVE_PATHS_ARRAY) {
      local_size = GPU_PARALLEL_ACTIVE_INDEX_DEFAULT_BLOCK_SIZE;
    }
#  ifdef WITH_ONEAPI_SYCL_HOST_ENABLED
    if (queue->get_device().is_host()) {
      global_size = 1;
      local_size = 1;
    }
#  endif
  }

  /* Let the compiler throw an error if there are any kernels missing in this implementation. */
#  if defined(_WIN32)
#    pragma warning(error : 4062)
#  elif defined(__GNUC__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic error "-Wswitch"
#  endif

  try {
    queue->submit([&](sycl::handler &cgh) {
      switch (device_kernel) {
        case DEVICE_KERNEL_INTEGRATOR_RESET: {
          oneapi_call(kg, cgh, global_size, local_size, args, oneapi_kernel_integrator_reset);
          break;
        }
        case DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA: {
          oneapi_call(
              kg, cgh, global_size, local_size, args, oneapi_kernel_integrator_init_from_camera);
          break;
        }
        case DEVICE_KERNEL_INTEGRATOR_INIT_FROM_BAKE: {
          oneapi_call(
              kg, cgh, global_size, local_size, args, oneapi_kernel_integrator_init_from_bake);
          break;
        }
        case DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST: {
          oneapi_call(
              kg, cgh, global_size, local_size, args, oneapi_kernel_integrator_intersect_closest);
          break;
        }
        case DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW: {
          oneapi_call(
              kg, cgh, global_size, local_size, args, oneapi_kernel_integrator_intersect_shadow);
          break;
        }
        case DEVICE_KERNEL_INTEGRATOR_INTERSECT_SUBSURFACE: {
          oneapi_call(kg,
                      cgh,
                      global_size,
                      local_size,
                      args,
                      oneapi_kernel_integrator_intersect_subsurface);
          break;
        }
        case DEVICE_KERNEL_INTEGRATOR_INTERSECT_VOLUME_STACK: {
          oneapi_call(kg,
                      cgh,
                      global_size,
                      local_size,
                      args,
                      oneapi_kernel_integrator_intersect_volume_stack);
          break;
        }
        case DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND: {
          oneapi_call(
              kg, cgh, global_size, local_size, args, oneapi_kernel_integrator_shade_background);
          break;
        }
        case DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT: {
          oneapi_call(
              kg, cgh, global_size, local_size, args, oneapi_kernel_integrator_shade_light);
          break;
        }
        case DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW: {
          oneapi_call(
              kg, cgh, global_size, local_size, args, oneapi_kernel_integrator_shade_shadow);
          break;
        }
        case DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE: {
          oneapi_call(
              kg, cgh, global_size, local_size, args, oneapi_kernel_integrator_shade_surface);
          break;
        }
        case DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_RAYTRACE: {
          oneapi_call(kg,
                      cgh,
                      global_size,
                      local_size,
                      args,
                      oneapi_kernel_integrator_shade_surface_raytrace);
          break;
        }
        case DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME: {
          oneapi_call(
              kg, cgh, global_size, local_size, args, oneapi_kernel_integrator_shade_volume);
          break;
        }
        case DEVICE_KERNEL_INTEGRATOR_QUEUED_PATHS_ARRAY: {
          oneapi_call(
              kg, cgh, global_size, local_size, args, oneapi_kernel_integrator_queued_paths_array);
          break;
        }
        case DEVICE_KERNEL_INTEGRATOR_QUEUED_SHADOW_PATHS_ARRAY: {
          oneapi_call(kg,
                      cgh,
                      global_size,
                      local_size,
                      args,
                      oneapi_kernel_integrator_queued_shadow_paths_array);
          break;
        }
        case DEVICE_KERNEL_INTEGRATOR_ACTIVE_PATHS_ARRAY: {
          oneapi_call(
              kg, cgh, global_size, local_size, args, oneapi_kernel_integrator_active_paths_array);
          break;
        }
        case DEVICE_KERNEL_INTEGRATOR_TERMINATED_PATHS_ARRAY: {
          oneapi_call(kg,
                      cgh,
                      global_size,
                      local_size,
                      args,
                      oneapi_kernel_integrator_terminated_paths_array);
          break;
        }
        case DEVICE_KERNEL_INTEGRATOR_TERMINATED_SHADOW_PATHS_ARRAY: {
          oneapi_call(kg,
                      cgh,
                      global_size,
                      local_size,
                      args,
                      oneapi_kernel_integrator_terminated_shadow_paths_array);
          break;
        }
        case DEVICE_KERNEL_INTEGRATOR_SORTED_PATHS_ARRAY: {
          oneapi_call(
              kg, cgh, global_size, local_size, args, oneapi_kernel_integrator_sorted_paths_array);
          break;
        }
        case DEVICE_KERNEL_INTEGRATOR_COMPACT_PATHS_ARRAY: {
          oneapi_call(kg,
                      cgh,
                      global_size,
                      local_size,
                      args,
                      oneapi_kernel_integrator_compact_paths_array);
          break;
        }
        case DEVICE_KERNEL_INTEGRATOR_COMPACT_SHADOW_PATHS_ARRAY: {
          oneapi_call(kg,
                      cgh,
                      global_size,
                      local_size,
                      args,
                      oneapi_kernel_integrator_compact_shadow_paths_array);
          break;
        }
        case DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_CHECK: {
          oneapi_call(kg,
                      cgh,
                      global_size,
                      local_size,
                      args,
                      oneapi_kernel_adaptive_sampling_convergence_check);
          break;
        }
        case DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_FILTER_X: {
          oneapi_call(
              kg, cgh, global_size, local_size, args, oneapi_kernel_adaptive_sampling_filter_x);
          break;
        }
        case DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_FILTER_Y: {
          oneapi_call(
              kg, cgh, global_size, local_size, args, oneapi_kernel_adaptive_sampling_filter_y);
          break;
        }
        case DEVICE_KERNEL_SHADER_EVAL_DISPLACE: {
          oneapi_call(kg, cgh, global_size, local_size, args, oneapi_kernel_shader_eval_displace);
          break;
        }
        case DEVICE_KERNEL_SHADER_EVAL_BACKGROUND: {
          oneapi_call(
              kg, cgh, global_size, local_size, args, oneapi_kernel_shader_eval_background);
          break;
        }
        case DEVICE_KERNEL_SHADER_EVAL_CURVE_SHADOW_TRANSPARENCY: {
          oneapi_call(kg,
                      cgh,
                      global_size,
                      local_size,
                      args,
                      oneapi_kernel_shader_eval_curve_shadow_transparency);
          break;
        }
        case DEVICE_KERNEL_PREFIX_SUM: {
          oneapi_call(kg, cgh, global_size, local_size, args, oneapi_kernel_prefix_sum);
          break;
        }

        /* clang-format off */
    #  define DEVICE_KERNEL_FILM_CONVERT_PARTIAL(VARIANT, variant) \
    case DEVICE_KERNEL_FILM_CONVERT_##VARIANT: { \
      oneapi_call(kg, cgh, \
                            global_size, \
                            local_size, \
                            args, \
                            oneapi_kernel_film_convert_##variant); \
      break; \
     }

#  define DEVICE_KERNEL_FILM_CONVERT(variant, VARIANT) \
      DEVICE_KERNEL_FILM_CONVERT_PARTIAL(VARIANT, variant) \
      DEVICE_KERNEL_FILM_CONVERT_PARTIAL(VARIANT##_HALF_RGBA, variant##_half_rgba)

      DEVICE_KERNEL_FILM_CONVERT(depth, DEPTH);
      DEVICE_KERNEL_FILM_CONVERT(mist, MIST);
      DEVICE_KERNEL_FILM_CONVERT(sample_count, SAMPLE_COUNT);
      DEVICE_KERNEL_FILM_CONVERT(float, FLOAT);
      DEVICE_KERNEL_FILM_CONVERT(light_path, LIGHT_PATH);
      DEVICE_KERNEL_FILM_CONVERT(float3, FLOAT3);
      DEVICE_KERNEL_FILM_CONVERT(motion, MOTION);
      DEVICE_KERNEL_FILM_CONVERT(cryptomatte, CRYPTOMATTE);
      DEVICE_KERNEL_FILM_CONVERT(shadow_catcher, SHADOW_CATCHER);
      DEVICE_KERNEL_FILM_CONVERT(shadow_catcher_matte_with_shadow,
                                 SHADOW_CATCHER_MATTE_WITH_SHADOW);
      DEVICE_KERNEL_FILM_CONVERT(combined, COMBINED);
      DEVICE_KERNEL_FILM_CONVERT(float4, FLOAT4);

#  undef DEVICE_KERNEL_FILM_CONVERT
#  undef DEVICE_KERNEL_FILM_CONVERT_PARTIAL
          /* clang-format on */

        case DEVICE_KERNEL_FILTER_GUIDING_PREPROCESS: {
          oneapi_call(
              kg, cgh, global_size, local_size, args, oneapi_kernel_filter_guiding_preprocess);
          break;
        }
        case DEVICE_KERNEL_FILTER_GUIDING_SET_FAKE_ALBEDO: {
          oneapi_call(kg,
                      cgh,
                      global_size,
                      local_size,
                      args,
                      oneapi_kernel_filter_guiding_set_fake_albedo);
          break;
        }
        case DEVICE_KERNEL_FILTER_COLOR_PREPROCESS: {
          oneapi_call(
              kg, cgh, global_size, local_size, args, oneapi_kernel_filter_color_preprocess);
          break;
        }
        case DEVICE_KERNEL_FILTER_COLOR_POSTPROCESS: {
          oneapi_call(
              kg, cgh, global_size, local_size, args, oneapi_kernel_filter_color_postprocess);
          break;
        }
        case DEVICE_KERNEL_CRYPTOMATTE_POSTPROCESS: {
          oneapi_call(
              kg, cgh, global_size, local_size, args, oneapi_kernel_cryptomatte_postprocess);
          break;
        }
        case DEVICE_KERNEL_INTEGRATOR_COMPACT_STATES: {
          oneapi_call(
              kg, cgh, global_size, local_size, args, oneapi_kernel_integrator_compact_states);
          break;
        }
        case DEVICE_KERNEL_INTEGRATOR_COMPACT_SHADOW_STATES: {
          oneapi_call(kg,
                      cgh,
                      global_size,
                      local_size,
                      args,
                      oneapi_kernel_integrator_compact_shadow_states);
          break;
        }
        case DEVICE_KERNEL_INTEGRATOR_SHADOW_CATCHER_COUNT_POSSIBLE_SPLITS: {
          oneapi_call(kg,
                      cgh,
                      global_size,
                      local_size,
                      args,
                      oneapi_kernel_integrator_shadow_catcher_count_possible_splits);
          break;
        }
        /* Unsupported kernels */
        case DEVICE_KERNEL_NUM:
        case DEVICE_KERNEL_INTEGRATOR_MEGAKERNEL:
          assert(0);
          return false;
      }

      /* Unknown kernel. */
      assert(0);
      return false;
    });
    if (kernel_context->with_per_kernel_statistics) {
      queue->wait_and_throw();
    }
  }
  catch (sycl::exception const &e) {
    if (s_error_cb) {
      s_error_cb(e.what(), s_error_user_ptr);
      success = false;
    }
  }

#  if defined(_WIN32)
#    pragma warning(default : 4062)
#  elif defined(__GNUC__)
#    pragma GCC diagnostic pop
#  endif
  return success;
}

// TODO(sirgienko) Put proper linux driver version number here, when it will be known, which
// exactly driver will have lowerest support
static const int windows_lowest_supported_driver_build_version = 1011660;
static const int linux_lowest_supported_driver_build_version = 20066;

// Windows: os_type == 1
// Linux: os_type == 2
static int parse_driver_build_version(const sycl::device &device, int &os_type)
{
  const std::string &device_name = device.get_info<sycl::info::device::name>();
  const std::string &driver_version = device.get_info<sycl::info::device::driver_version>();

  // NOTE(sirgienko) Windows Intel GPUs driver version look like this:
  //   xx.x.101.1340
  // the 7 last digits are forming the version number.
  // For more details, see
  // https://www.intel.com/content/www/us/en/support/articles/000005654/graphics.html
  if (std::count(driver_version.begin(), driver_version.end(), '.') == 3) {
    os_type = 1;
  }
  // Linux versions examples: 22.01.022066, 22.01.018324
  // Last number is used as driver version number.
  else if (std::count(driver_version.begin(), driver_version.end(), '.') == 2) {
    os_type = 2;
  }
  else {
    std::cerr << "Unable to parse incorrect/unknown Intel GPU driver version \"" << driver_version
              << "\" - device \"" << device_name << "\" will be skipped" << std::endl;
    os_type = 0;
    return 0;
  }

  size_t second_dot_position = driver_version.find('.', driver_version.find('.', 0) + 1);
  // For easier parsing, let's consider third dot as an end string position for Linux version
  // string
  size_t third_dot_position = os_type == 1 ? driver_version.find('.', second_dot_position + 1) :
                                             driver_version.length();

  int driver_build_version = 0;
  try {
    const std::string &third_number_substr = driver_version.substr(
        second_dot_position + 1, third_dot_position - second_dot_position - 1);
    // Windows
    if (os_type == 1) {
      const std::string &forth_number_substr = driver_version.substr(
          third_dot_position + 1, driver_version.length() - third_dot_position - 1);
      if (third_number_substr.length() == 3 && forth_number_substr.length() == 4)
        driver_build_version = std::stoi(third_number_substr) * 10000 +
                               std::stoi(forth_number_substr);
      else
        std::cerr << "Unable to parse incorrect Intel GPU driver version \"" << driver_version
                  << "\" - " << third_number_substr << "." << forth_number_substr
                  << " does not match template xxx.xxxx, so device \"" << device_name
                  << "\" be skipped" << std::endl;
    }
    // Linux
    else if (os_type == 2) {
      if (third_number_substr.length() == 5 || third_number_substr.length() == 6)
        driver_build_version = std::stoi(third_number_substr);
      else
        std::cerr << "Unable to parse unknown Intel GPU driver version \"" << driver_version
                  << "\" - the version does not match template xx.xx.xxxxx, so device \""
                  << device_name << "\" will be skipped" << std::endl;
    }
  }
  catch (std::invalid_argument &e) {
    std::cerr << "Unable to parse unknown Intel GPU driver version \"" << driver_version
              << "\" - throws number parse exception \"" << e.what() << "\" so device \""
              << device_name << "\" will be skipped" << std::endl;
  }

  return driver_build_version;
}

static size_t make_device_uuid(const sycl::device &device)
{
  size_t max_compute_units = device.get_info<sycl::info::device::max_compute_units>();
  size_t max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();
  // Host device for some reason doesn't support this, so we set some default value
  size_t max_clock_frequency = device.is_host() ?
                                   0 :
                                   device.get_info<sycl::info::device::max_clock_frequency>();

  return max_compute_units << 20 | max_work_group_size << 10 | max_clock_frequency;
}

static std::vector<sycl::device> oneapi_available_devices(
    std::map<std::string, std::string> *proper_names_map)
{
  bool allow_all_devices = false;
  if (getenv("CYCLES_ONEAPI_ALL_DEVICES"))
    allow_all_devices = true;

  bool allow_host = false;
  // Host device useful only for debugging so we hide this device with
  // default build settings
#  ifdef WITH_ONEAPI_SYCL_HOST_ENABLED
  allow_host = true;
#  endif

  // There are no benefits to support OpenCL backend, while we target L0 backend.
  // Also, presenting two platforms in UI will give user an ability to choose one device for
  // rendering twice: via OpenCL and via Level0, which doesn't make any sense (and will
  // have obvious performance impact).
  const static std::string level_zero_platform_name = "Intel(R) Level-Zero";
  const static std::string opencl_platform_name = "Intel(R) OpenCL HD Graphics";
  const static std::set<std::string> supported_platforms = {level_zero_platform_name};
  const static bool is_level_zero_support_enabled = supported_platforms.find(
                                                        level_zero_platform_name) !=
                                                    supported_platforms.end();
  const std::vector<sycl::platform> &oneapi_platforms = sycl::platform::get_platforms();

  // NOTE(sirgienko) Right now only OpenCL devices report GPU system driver version
  // and proper device name. So we need to get driver info first by scanning for OpenCL platforms.
  // And then filter-out devices by this information (and compute proper names in the same time)

  // HW characteristics will be used to create an unique key - not best solution, but at least
  // working for all different devices.
  std::map<size_t, bool> device_filtering_list;
  std::map<size_t, std::string> device_name_list;
  for (const sycl::platform &platform : oneapi_platforms) {
    const std::string &platform_name = platform.get_info<sycl::info::platform::name>();

    if (platform_name == level_zero_platform_name)
      continue;

    bool is_enabled_host_device = allow_host && platform.is_host();
    const std::vector<sycl::device> &oneapi_devices = platform.get_devices();
    for (const sycl::device &device : oneapi_devices) {
      size_t uuid = make_device_uuid(device);

      if (device_filtering_list.find(uuid) == device_filtering_list.end())
        device_filtering_list[uuid] = false;

      if (allow_all_devices || is_enabled_host_device) {
        device_filtering_list[uuid] = true;
      }
      else {
        if (device.is_gpu() && platform_name == opencl_platform_name &&
            is_level_zero_support_enabled) {
          int os_type = 0;
          int driver_build_version = parse_driver_build_version(device, os_type);

          // Ignore GPUs with driver version older that first recommended
          bool is_windows = os_type == 1;
          if (is_windows && driver_build_version >= windows_lowest_supported_driver_build_version)
            device_filtering_list[uuid] = true;

          bool is_linux = os_type == 2;
          if (is_linux && driver_build_version >= linux_lowest_supported_driver_build_version)
            device_filtering_list[uuid] = true;
        }
        else {
          device_filtering_list[uuid] = false;
        }
      }

      device_name_list[uuid] = device.get_info<sycl::info::device::name>();
    }
  }

  // Clean this map before putting any new values into it
  if (proper_names_map)
    proper_names_map->clear();

  std::vector<sycl::device> available_devices;
  for (const sycl::platform &platform : oneapi_platforms) {
    const std::string &platform_name = platform.get_info<sycl::info::platform::name>();

    if (platform_name == opencl_platform_name)
      continue;

    const std::vector<sycl::device> &oneapi_devices = platform.get_devices();
    for (const sycl::device &device : oneapi_devices) {
      size_t uuid = make_device_uuid(device);

      bool is_filtered_out_additionaly = false;

      // For GPU we will additionally filter-out devices by device id
      if (device.is_gpu() && platform_name == level_zero_platform_name &&
          is_level_zero_support_enabled) {
        ze_device_handle_t ze_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
            device);
        ze_device_properties_t prop;
        zeDeviceGetProperties(ze_device, &prop);
        is_filtered_out_additionaly = oneapi_allowed_gpu_devices.find(prop.deviceId) ==
                                      oneapi_allowed_gpu_devices.end();
      }

      if (device_filtering_list[uuid] && !is_filtered_out_additionaly) {
        available_devices.push_back(device);

        if (proper_names_map) {
          const std::string &device_name = device.get_info<sycl::info::device::name>();
          if (device_name_list.find(uuid) != device_name_list.end()) {
            (*proper_names_map)[device_name] = device_name_list[uuid];
          }
          else {
            // If we don't have good name, then just use name, which reported by backend
            (*proper_names_map)[device_name] = device_name;
          }
        }
      }
    }
  }

  return available_devices;
}

char *oneapi_device_capabilities()
{
  std::stringstream capabilities;

  std::map<std::string, std::string> proper_names;
  const std::vector<sycl::device> &oneapi_devices = oneapi_available_devices(&proper_names);
  for (const sycl::device &device : oneapi_devices) {
    const std::string &name = proper_names[device.get_info<sycl::info::device::name>()];

    capabilities << std::string("\t") << name << "\n";
#  define WRITE_ATTR(attribute_name, attribute_variable) \
    capabilities << "\t\tsycl::info::device::" #attribute_name "\t\t\t" << attribute_variable \
                 << "\n";
#  define GET_NUM_ATTR(attribute) \
    { \
      size_t attribute = (size_t)device.get_info<sycl::info::device ::attribute>(); \
      capabilities << "\t\tsycl::info::device::" #attribute "\t\t\t" << attribute << "\n"; \
    }

    GET_NUM_ATTR(vendor_id)
    GET_NUM_ATTR(max_compute_units)
    GET_NUM_ATTR(max_work_item_dimensions)

    sycl::id<3> max_work_item_sizes = device.get_info<sycl::info::device::max_work_item_sizes>();
    WRITE_ATTR("max_work_item_sizes_dim0", ((size_t)max_work_item_sizes.get(0)))
    WRITE_ATTR("max_work_item_sizes_dim1", ((size_t)max_work_item_sizes.get(1)))
    WRITE_ATTR("max_work_item_sizes_dim2", ((size_t)max_work_item_sizes.get(2)))

    GET_NUM_ATTR(max_work_group_size)
    GET_NUM_ATTR(max_num_sub_groups)
    GET_NUM_ATTR(sub_group_independent_forward_progress)

    GET_NUM_ATTR(preferred_vector_width_char)
    GET_NUM_ATTR(preferred_vector_width_short)
    GET_NUM_ATTR(preferred_vector_width_int)
    GET_NUM_ATTR(preferred_vector_width_long)
    GET_NUM_ATTR(preferred_vector_width_float)
    GET_NUM_ATTR(preferred_vector_width_double)
    GET_NUM_ATTR(preferred_vector_width_half)

    GET_NUM_ATTR(native_vector_width_char)
    GET_NUM_ATTR(native_vector_width_short)
    GET_NUM_ATTR(native_vector_width_int)
    GET_NUM_ATTR(native_vector_width_long)
    GET_NUM_ATTR(native_vector_width_float)
    GET_NUM_ATTR(native_vector_width_double)
    GET_NUM_ATTR(native_vector_width_half)

    size_t max_clock_frequency =
        (size_t)(device.is_host() ? (size_t)0 :
                                    device.get_info<sycl::info::device::max_clock_frequency>());
    WRITE_ATTR("max_clock_frequency", max_clock_frequency)

    GET_NUM_ATTR(address_bits)
    GET_NUM_ATTR(max_mem_alloc_size)

    // NOTE(sirgienko) Implementation doesn't use image support as bindless images aren't supported
    // so we always return false, even if device supports HW texture usage acceleration
    bool image_support = false;
    WRITE_ATTR("image_support", (size_t)image_support)

    GET_NUM_ATTR(max_parameter_size)
    GET_NUM_ATTR(mem_base_addr_align)
    GET_NUM_ATTR(global_mem_size)
    GET_NUM_ATTR(local_mem_size)
    GET_NUM_ATTR(error_correction_support)
    GET_NUM_ATTR(profiling_timer_resolution)
    GET_NUM_ATTR(is_available)

#  undef GET_NUM_ATTR
#  undef WRITE_ATTR
    capabilities << "\n";
  }

  return ::strdup(capabilities.str().c_str());
}

void oneapi_free(void *p)
{
  if (p) {
    ::free(p);
  }
}

void oneapi_iterate_devices(OneAPIDeviceIteratorCallback cb, void *user_ptr)
{
  int num = 0;
  std::map<std::string, std::string> proper_names;
  std::vector<sycl::device> devices = oneapi_available_devices(&proper_names);
  for (sycl::device &device : devices) {
    const std::string &platform_name =
        device.get_platform().get_info<sycl::info::platform::name>();
    std::string name = proper_names[device.get_info<sycl::info::device::name>()];
    std::string id = "ONEAPI_" + platform_name + "_" + name;
    (cb)(id.c_str(), name.c_str(), num, user_ptr);
    num++;
  }
}

size_t oneapi_get_memcapacity(SyclQueue *queue)
{
  return reinterpret_cast<sycl::queue *>(queue)
      ->get_device()
      .get_info<sycl::info::device::global_mem_size>();
}

size_t oneapi_get_compute_units_amount(SyclQueue *queue)
{
  return reinterpret_cast<sycl::queue *>(queue)
      ->get_device()
      .get_info<sycl::info::device::max_compute_units>();
}

#endif /* WITH_ONEAPI */
