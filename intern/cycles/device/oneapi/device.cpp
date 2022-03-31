/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2021-2022 Intel Corporation */

#include "device/oneapi/device.h"

#include "util/log.h"

#ifdef WITH_ONEAPI
#  include "device/device.h"
#  include "device/oneapi/device_impl.h"
#  include "device/oneapi/sycl.h"
#  include "device/oneapi/util.h"

#  include "util/string.h"

#  ifdef __linux__
#    include <dlfcn.h>
#  endif
#endif /* WITH_ONEAPI */

CCL_NAMESPACE_BEGIN

#ifdef WITH_ONEAPI
static oneAPIDLLInterface oneapi_dll;
#endif

#ifdef _WIN32
#  define LOAD_SYCL_SHARED_LIBRARY() (void *)(LoadLibrary(TEXT("sycl.dll")))
#  define LOAD_ONEAPI_SHARED_LIBRARY() (void *)(LoadLibrary(TEXT("cycles_kernel_oneapi.dll")))
#  define FREE_SHARED_LIBRARY(handle) FreeLibrary((HMODULE)handle)
#  define GET_SHARED_LIBRARY_SYMBOL(handle, name) GetProcAddress((HMODULE)handle, name)
#elif __linux__
#  define LOAD_SYCL_SHARED_LIBRARY() dlopen("libsycl.so", RTLD_LAZY)
#  define LOAD_ONEAPI_SHARED_LIBRARY() dlopen("libcycles_kernel_oneapi.so", RTLD_NOW)
#  define FREE_SHARED_LIBRARY(handle) dlclose(handle)
#  define GET_SHARED_LIBRARY_SYMBOL(handle, name) dlsym(handle, name)
#endif

bool device_oneapi_init()
{
#if !defined(WITH_ONEAPI)
  return false;
#else
  void *lib_handle = LOAD_SYCL_SHARED_LIBRARY();

  // If sycl shared library can't be found, then disable oneAPI rendering completely
  if (lib_handle == NULL) {
    VLOG(1) << "oneAPI shared library has not been found, so oneAPI rendering will be disabled";
    return false;
  }

  FREE_SHARED_LIBRARY(lib_handle);

  lib_handle = LOAD_ONEAPI_SHARED_LIBRARY();

  // This shouldn't happens, but still make sense to have a branch for this
  if (lib_handle == NULL) {
    LOG(ERROR) << "oneAPI kernel shared library cannot be loaded for some reason. This should not "
                  "happen, however, it occurs hence oneAPI rendering will be disabled";
    return false;
  }

#  define DLL_INTERFACE_CALL(function, return_type, ...) \
    (oneapi_dll.function) = reinterpret_cast<decltype(oneapi_dll.function)>( \
        GET_SHARED_LIBRARY_SYMBOL(lib_handle, #function)); \
    if (oneapi_dll.function == NULL) { \
      LOG(ERROR) << "oneAPI shared library function \"" << #function \
                 << "\" has not been loaded from kernel shared  - disable oneAPI " \
                    "library disable oneAPI implementation due to this"; \
      FREE_SHARED_LIBRARY(lib_handle); \
      return false; \
    }
#  include "kernel/device/oneapi/dll_interface_template.h"
#  undef DLL_INTERFACE_CALL

  VLOG(1) << "oneAPI kernel shared library has been loaded successfully";

  // We need to have this oneapi kernel shared library during all life-span of the Blender.
  // So it is not unloaded because of this.
  // FREE_SHARED_LIBRARY(lib_handle);

  // NOTE(sirgienko) we need to enable JIT cache from here and
  // right now this cache policy is controlled by env. variables
  VLOG(1) << "Enable oneAPI persistent cache via environment variables";
#  ifdef _WIN32
  (void)putenv("SYCL_CACHE_PERSISTENT=1");
  (void)putenv("SYCL_CACHE_THRESHOLD=0");
  VLOG(1) << "Enabled oneAPI JIT caching on Windows platform";
#  elif __linux__
  setenv("SYCL_CACHE_PERSISTENT", "1", true);
  setenv("SYCL_CACHE_THRESHOLD", "0", true);
  VLOG(1) << "Enabled oneAPI JIT caching on Linux platform ";
#  else
  VLOG(1) << "Could not enable oneAPI JIT caching because the OS (Operating System) platform is "
             "neither Linux nor Windows";
#  endif

  return true;
#endif
}

#if defined(_WIN32) || defined(__linux__)
#  undef LOAD_SYCL_SHARED_LIBRARY
#  undef LOAD_ONEAPI_SHARED_LIBRARY
#  undef FREE_SHARED_LIBRARY
#  undef GET_SHARED_LIBRARY_SYMBOL
#endif

Device *device_oneapi_create(const DeviceInfo &info, Stats &stats, Profiler &profiler)
{
#ifdef WITH_ONEAPI
  return new OneapiDevice(info, oneapi_dll, stats, profiler);
#else
  (void)info;
  (void)stats;
  (void)profiler;

  LOG(FATAL) << "Requested to create oneAPI device while not enabled for this build.";

  return nullptr;
#endif
}

#ifdef WITH_ONEAPI
static void device_iterator_cb(const char *id, const char *name, int num, void *user_ptr)
{
  vector<DeviceInfo> *devices = (vector<DeviceInfo> *)user_ptr;

  DeviceInfo info;

  info.type = DEVICE_ONEAPI;
  info.description = name;
  info.num = num;

  // NOTE(sirgienko) Should be unique at least on proper oneapi installation
  info.id = id;

  info.has_nanovdb = true;
  info.denoisers = 0;

  info.has_gpu_queue = true;

  // NOTE(sirgienko) oneAPI right now is focused on one device usage. In future it maybe will
  // change, but right now peer access from one device to another device is not supported
  info.has_peer_memory = false;

  // NOTE(sirgienko) Seems not possible to know from SYCL/oneAPI or Level0.
  info.display_device = false;

  devices->push_back(info);
  VLOG(1) << "Added device \"" << name << "\" with id \"" << info.id << "\".";
}
#endif

void device_oneapi_info(vector<DeviceInfo> &devices)
{
#ifdef WITH_ONEAPI
  (oneapi_dll.oneapi_iterate_devices)(device_iterator_cb, &devices);
#else  /* WITH_ONEAPI */
  (void)devices;
#endif /* WITH_ONEAPI */
}

string device_oneapi_capabilities()
{
  string capabilities;
#ifdef WITH_ONEAPI
  char *c_capabilities = (oneapi_dll.oneapi_device_capabilities)();
  if (c_capabilities) {
    capabilities = c_capabilities;
    (oneapi_dll.oneapi_free)(c_capabilities);
  }
#endif
  return capabilities;
}

CCL_NAMESPACE_END
