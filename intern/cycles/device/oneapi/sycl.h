/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2021-2022 Intel Corporation */

#pragma once

#ifdef WITH_ONEAPI
#  ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wdouble-promotion"
#    pragma GCC diagnostic ignored "-Wfloat-conversion"
#    pragma GCC diagnostic ignored "-Wredundant-decls"
#  endif /* __GNUC__ */
#  include "CL/sycl.hpp"
#  ifdef __GNUC__
#    pragma GCC diagnostic pop
#  endif /* __GNUC__ */
#endif
