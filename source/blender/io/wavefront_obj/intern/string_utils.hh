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
 * The Original Code is Copyright (C) 2020 Blender Foundation.
 * All rights reserved.
 */

namespace blender::io::obj {

void read_next_line(std::ifstream &file, std::string &r_line);
void split_line_key_rest(StringRef line, StringRef &r_line_key, StringRef &r_rest_line);
void split_by_char(StringRef in_string, const char delimiter, Vector<StringRef> &r_out_list);
void copy_string_to_float(StringRef src, const float fallback_value, float &r_dst);
void copy_string_to_float(Span<StringRef> src,
                          const float fallback_value,
                          MutableSpan<float> r_dst);
void copy_string_to_int(StringRef src, const int fallback_value, int &r_dst);
void copy_string_to_int(Span<StringRef> src, const int fallback_value, MutableSpan<int> r_dst);
std::string replace_all_occurences(StringRef original, StringRef to_remove, StringRef to_add);

}  // namespace blender::io::obj
