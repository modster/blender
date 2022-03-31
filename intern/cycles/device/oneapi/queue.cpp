/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2021-2022 Intel Corporation */

#ifdef WITH_ONEAPI

#  include "device/oneapi/queue.h"
#  include "device/oneapi/device_impl.h"
#  include "util/log.h"
#  include "util/time.h"
#  include <iomanip>
#  include <vector>

#  include "kernel/device/oneapi/kernel.h"

CCL_NAMESPACE_BEGIN

struct KernelExecutionInfo {
  double elapsed_summary = 0.0;
  int enqueue_count = 0;
};

struct OneapiKernelStats {
  OneapiKernelStats(){};
  ~OneapiKernelStats()
  {
  }

  void print_and_reset()
  {
    if (stats_data.size() != 0) {
      std::vector<std::pair<DeviceKernel, KernelExecutionInfo>> stats_sorted;
      for (const auto &stat : stats_data) {
        stats_sorted.push_back(stat);
      }

      sort(stats_sorted.begin(),
           stats_sorted.end(),
           [](const std::pair<DeviceKernel, KernelExecutionInfo> &a,
              const std::pair<DeviceKernel, KernelExecutionInfo> &b) {
             return a.second.elapsed_summary > b.second.elapsed_summary;
           });

      VLOG(1) << "oneAPI execution kernels statistics:";
      double summary = 0.0;
      for (const std::pair<DeviceKernel, KernelExecutionInfo> &iter : stats_sorted) {
        VLOG(1) << "  " << std::setfill(' ') << std::setw(10) << std::fixed << std::setprecision(5)
                << std::right << iter.second.elapsed_summary
                << "s: " << device_kernel_as_string(iter.first) << " ("
                << iter.second.enqueue_count << " runs)";
        summary += iter.second.elapsed_summary;
      }
      VLOG(1) << "Total measured kernel execution time: " << std::fixed << std::setprecision(5)
              << summary << "s";

      stats_data.clear();
      active_kernels.clear();
    }
  }

  void kernel_enqueued(DeviceKernel kernel)
  {
    assert(active_kernels.find(kernel) == active_kernels.end());
    active_kernels[kernel] = time_dt();
  }

  void kernel_finished(DeviceKernel kernel, unsigned int /*kernel_work_size*/)
  {
    assert(active_kernels.find(kernel) != active_kernels.end());
    double elapsed_time = time_dt() - active_kernels[kernel];
    active_kernels.erase(kernel);

    stats_data[kernel].elapsed_summary += elapsed_time;
    stats_data[kernel].enqueue_count += 1;
  }

  std::map<DeviceKernel, KernelExecutionInfo> stats_data;
  std::map<DeviceKernel, double> active_kernels;
};

static OneapiKernelStats global_kernel_stats;

/* OneapiDeviceQueue */

OneapiDeviceQueue::OneapiDeviceQueue(OneapiDevice *device)
    : DeviceQueue(device),
      oneapi_device(device),
      oneapi_dll(device->oneapi_dll_object()),
      kernel_context(nullptr)
{
  if (getenv("CYCLES_ONEAPI_KERNEL_STATS") && VLOG_IS_ON(1))
    with_kernel_statistics = true;
  else
    with_kernel_statistics = false;
}

OneapiDeviceQueue::~OneapiDeviceQueue()
{
  if (kernel_context)
    delete kernel_context;

  if (with_kernel_statistics)
    global_kernel_stats.print_and_reset();
}

int OneapiDeviceQueue::num_concurrent_states(const size_t state_size) const
{
  int num_states;

  const size_t compute_units =
      (oneapi_dll.oneapi_get_compute_units_amount)(oneapi_device->sycl_queue());
  if (compute_units >= 128) {
    // dGPU path, make sense to allocate more states, because it will be dedicated GPU memory
    int base = 1024 * 1024;
    // linear dependency (with coefficient less that 1) from amount of compute units
    num_states = (base * (compute_units / 128)) * 3 / 4;

    // Limit amount of integrator states by one quarter of device memory, because
    // other allocations will need some space as well
    size_t states_memory_size = num_states * state_size;
    size_t device_memory_amount = (oneapi_dll.oneapi_get_memcapacity)(oneapi_device->sycl_queue());
    if (states_memory_size >= device_memory_amount / 4) {
      num_states = device_memory_amount / 4 / state_size;
    }
  }
  else {
    // iGPU path - no really nead to allocate a lot of integrator states, because it is shared GPU
    // memory
    num_states = 1024 * 512;
  }

  VLOG(3) << "GPU queue concurrent states: " << num_states << ", using up to "
          << string_human_readable_size(num_states * state_size);

  return num_states;
}

int OneapiDeviceQueue::num_concurrent_busy_states() const
{
  const size_t compute_units =
      (oneapi_dll.oneapi_get_compute_units_amount)(oneapi_device->sycl_queue());
  if (compute_units >= 128) {
    return 1024 * 1024;
  }
  else {
    return 1024 * 512;
  }
}

void OneapiDeviceQueue::init_execution()
{
  oneapi_device->load_texture_info();

  SyclQueue *device_queue = oneapi_device->sycl_queue();
  void *kg_dptr = (void *)oneapi_device->kernel_globals_device_pointer();
  assert(device_queue);
  assert(kg_dptr);
  kernel_context = new KernelContext{device_queue, kg_dptr, with_kernel_statistics};

  debug_init_execution();
}

std::set<DeviceKernel> OneapiDeviceQueue::SUPPORTED_KERNELS = {
    DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA,
    DEVICE_KERNEL_INTEGRATOR_INIT_FROM_BAKE,
    DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST,
    DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW,
    DEVICE_KERNEL_INTEGRATOR_INTERSECT_SUBSURFACE,
    DEVICE_KERNEL_INTEGRATOR_INTERSECT_VOLUME_STACK,
    DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND,
    DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT,
    DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE,
    DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_RAYTRACE,
    DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME,
    DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW,

    DEVICE_KERNEL_INTEGRATOR_QUEUED_PATHS_ARRAY,
    DEVICE_KERNEL_INTEGRATOR_QUEUED_SHADOW_PATHS_ARRAY,
    DEVICE_KERNEL_INTEGRATOR_ACTIVE_PATHS_ARRAY,
    DEVICE_KERNEL_INTEGRATOR_TERMINATED_PATHS_ARRAY,
    DEVICE_KERNEL_INTEGRATOR_SORTED_PATHS_ARRAY,
    DEVICE_KERNEL_INTEGRATOR_COMPACT_PATHS_ARRAY,
    DEVICE_KERNEL_INTEGRATOR_COMPACT_STATES,
    DEVICE_KERNEL_INTEGRATOR_TERMINATED_SHADOW_PATHS_ARRAY,
    DEVICE_KERNEL_INTEGRATOR_COMPACT_SHADOW_PATHS_ARRAY,
    DEVICE_KERNEL_INTEGRATOR_COMPACT_SHADOW_STATES,
    DEVICE_KERNEL_INTEGRATOR_RESET,
    DEVICE_KERNEL_INTEGRATOR_SHADOW_CATCHER_COUNT_POSSIBLE_SPLITS,

    DEVICE_KERNEL_SHADER_EVAL_DISPLACE,
    DEVICE_KERNEL_SHADER_EVAL_BACKGROUND,
    DEVICE_KERNEL_SHADER_EVAL_CURVE_SHADOW_TRANSPARENCY,

#  define DECLARE_FILM_CONVERT_KERNEL(variant) \
    DEVICE_KERNEL_FILM_CONVERT_##variant, DEVICE_KERNEL_FILM_CONVERT_##variant##_HALF_RGBA

    DECLARE_FILM_CONVERT_KERNEL(DEPTH),
    DECLARE_FILM_CONVERT_KERNEL(MIST),
    DECLARE_FILM_CONVERT_KERNEL(SAMPLE_COUNT),
    DECLARE_FILM_CONVERT_KERNEL(FLOAT),
    DECLARE_FILM_CONVERT_KERNEL(LIGHT_PATH),
    DECLARE_FILM_CONVERT_KERNEL(FLOAT3),
    DECLARE_FILM_CONVERT_KERNEL(MOTION),
    DECLARE_FILM_CONVERT_KERNEL(CRYPTOMATTE),
    DECLARE_FILM_CONVERT_KERNEL(SHADOW_CATCHER),
    DECLARE_FILM_CONVERT_KERNEL(SHADOW_CATCHER_MATTE_WITH_SHADOW),
    DECLARE_FILM_CONVERT_KERNEL(COMBINED),
    DECLARE_FILM_CONVERT_KERNEL(FLOAT4),

#  undef DECLARE_FILM_CONVERT_KERNEL

    DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_CHECK,
    DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_FILTER_X,
    DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_FILTER_Y,

    DEVICE_KERNEL_FILTER_GUIDING_PREPROCESS,
    DEVICE_KERNEL_FILTER_GUIDING_SET_FAKE_ALBEDO,
    DEVICE_KERNEL_FILTER_COLOR_PREPROCESS,
    DEVICE_KERNEL_FILTER_COLOR_POSTPROCESS,

    DEVICE_KERNEL_CRYPTOMATTE_POSTPROCESS,

    DEVICE_KERNEL_PREFIX_SUM,
};

bool OneapiDeviceQueue::kernel_available(DeviceKernel kernel) const
{
  return SUPPORTED_KERNELS.find(kernel) != SUPPORTED_KERNELS.end();
}

bool OneapiDeviceQueue::enqueue(DeviceKernel kernel,
                                const int signed_kernel_work_size,
                                DeviceKernelArguments const &_args)
{
  if (oneapi_device->have_error()) {
    return false;
  }

  void **args = const_cast<void **>(_args.values);

  debug_enqueue(kernel, signed_kernel_work_size);
  assert(signed_kernel_work_size >= 0);
  size_t kernel_work_size = (size_t)signed_kernel_work_size;

  size_t kernel_local_size = (oneapi_dll.oneapi_kernel_prefered_local_size)(kernel_context->queue,
                                                                            (::DeviceKernel)kernel,
                                                                            kernel_work_size);
  size_t uniformed_kernel_work_size = round_up(kernel_work_size, kernel_local_size);

  assert(kernel_context);

  if (with_kernel_statistics)
    global_kernel_stats.kernel_enqueued(kernel);

  /* Call the oneAPI kernel DLL to launch the requested kernel. */
  bool is_finished_ok =
      (oneapi_dll.oneapi_enqueue_kernel)(kernel_context, kernel, uniformed_kernel_work_size, args);

  if (with_kernel_statistics)
    global_kernel_stats.kernel_finished(kernel, uniformed_kernel_work_size);

  if (is_finished_ok == false) {
    oneapi_device->set_error("oneAPI kernel \"" + std::string(device_kernel_as_string(kernel)) +
                             "\" execution error: got runtime exception \"" + oneapi_device->oneapi_error_message() +
                             "\"");
  }

  return is_finished_ok;
}

bool OneapiDeviceQueue::synchronize()
{
  if (oneapi_device->have_error()) {
    return false;
  }

  bool is_finished_ok = (oneapi_dll.oneapi_queue_synchronize)(oneapi_device->sycl_queue());
  if (is_finished_ok == false)
    oneapi_device->set_error("oneAPI unknown kernel execution error: got runtime exception \"" + oneapi_device->oneapi_error_message() + "\"");

  debug_synchronize();

  return !(oneapi_device->have_error());
}

void OneapiDeviceQueue::zero_to_device(device_memory &mem)
{
  oneapi_device->mem_zero(mem);
}

void OneapiDeviceQueue::copy_to_device(device_memory &mem)
{
  oneapi_device->mem_copy_to(mem);
}

void OneapiDeviceQueue::copy_from_device(device_memory &mem)
{
  oneapi_device->mem_copy_from(mem, 0, 1, 1, mem.memory_size());
}

CCL_NAMESPACE_END

#endif /* WITH_ONEAPI */
