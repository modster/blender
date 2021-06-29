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
 */

#include "vk_shader.hh"

#include "GPU_platform.h"

namespace blender::gpu {

char *VKLogParser::parse_line(char *log_line, GPULogItem &log_item)
{
  log_line = skip_name_and_stage(log_line);
  log_line = skip_separators(log_line, ":");

  /* Parse error line & char numbers. */
  if (at_number(log_line)) {
    char *error_line_number_end;
    log_item.cursor.row = parse_number(log_line, &error_line_number_end);
    log_line = error_line_number_end;
  }
  log_line = skip_separators(log_line, ": ");

  /* Skip to message. Avoid redundant info. */
  log_line = skip_severity_keyword(log_line, log_item);
  log_line = skip_separators(log_line, ": ");

  return log_line;
}

char *VKLogParser::skip_name_and_stage(char *log_line)
{
  char *name_skip = skip_until(log_line, '.');
  if (name_skip == log_line) {
    return log_line;
  }

  return skip_until(name_skip, ':');
}

char *VKLogParser::skip_severity_keyword(char *log_line, GPULogItem &log_item)
{
  return skip_severity(log_line, log_item, "error", "warning");
}

}  // namespace blender::gpu
