/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2021-2022 Intel Corporation */

#ifdef WITH_ONEAPI

#  include "device/device.h"
#  include "device/oneapi/device.h"
#  include "device/oneapi/queue.h"

#  include "util/map.h"

CCL_NAMESPACE_BEGIN

class DeviceQueue;

class OneapiDevice : public Device {
 private:
  SyclQueue *device_queue;

  using ConstMemMap = map<string, device_vector<uchar> *>;
  ConstMemMap m_const_mem_map;
  device_vector<TextureInfo> texture_info;
  bool need_texture_info;
  void *kg_memory;
  void *kg_memory_device;
  size_t kg_memory_size = (size_t)0;
  oneAPIDLLInterface oneapi_dll;
  std::string oneapi_error_string;

 public:
  virtual BVHLayoutMask get_bvh_layout_mask() const override;

  // No need to override
  // void set_error(const string &error) override;

  OneapiDevice(const DeviceInfo &info,
               oneAPIDLLInterface &oneapi_dll_object,
               Stats &stats,
               Profiler &profiler);

  virtual ~OneapiDevice();

  bool check_peer_access(Device *peer_device) override;

  bool load_kernels(const uint requested_features) override;

  string load_kernels_message(const uint requested_features) override;

  void load_texture_info();

  void generic_alloc(device_memory &mem);

  void generic_copy_to(device_memory &mem);

  void generic_free(device_memory &mem);

  SyclQueue *sycl_queue();

  string oneapi_error_message();

  oneAPIDLLInterface oneapi_dll_object();

  void *kernel_globals_device_pointer();

  void mem_alloc(device_memory &mem) override;

  void mem_copy_to(device_memory &mem) override;

  void mem_copy_from(device_memory &mem, size_t y, size_t w, size_t h, size_t elem) override;

  void mem_zero(device_memory &mem) override;

  void mem_free(device_memory &mem) override;

  device_ptr mem_alloc_sub_ptr(device_memory &mem, size_t offset, size_t /*size*/) override;

  virtual void const_copy_to(const char *name, void *host, size_t size) override;

  void global_alloc(device_memory &mem);

  void global_free(device_memory &mem);

  void tex_alloc(device_texture &mem);

  void tex_free(device_texture &mem);

  /* Graphics resources interoperability. */
  virtual bool should_use_graphics_interop() override;

  virtual unique_ptr<DeviceQueue> gpu_queue_create() override;

  // NOTE(sirgienko) Create this methods to avoid some compilation problems on Windows with host
  // side compilation (MSVC)
  void *usm_aligned_alloc_host(size_t memory_size, size_t alignment);
  void usm_free(void *usm_ptr);
};

CCL_NAMESPACE_END

#endif
