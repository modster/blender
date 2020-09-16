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

#include <iostream>
#include <fstream>
#include <sstream>

#include "BLI_float3.hh"
#include "BLI_span.hh"
#include "BLI_string_ref.hh"
#include "BLI_vector.hh"

#include "string_utils.hh"

namespace blender::io::obj {
using std::string;

/**
 * Store multiple lines separated by an escaped newline character: `\\n`.
 * Use this before doing any parse operations on the read string.
 */
void read_next_line(std::ifstream &file, string &r_line)
{
  std::string new_line;
  while (file.good() && !r_line.empty() && r_line.back() == '\\') {
    new_line.clear();
    const bool ok = static_cast<bool>(std::getline(file, new_line));
    /* Remove the last backslash character. */
    r_line.pop_back();
    r_line.append(new_line);
    if (!ok || new_line.empty()) {
      return;
    }
  }
}

/**
 * Split a line string into the first word (key) and the rest of the line.
 * Also remove leading & trailing spaces as well as `\r` carriage return
 * character if present.
 */
void split_line_key_rest(StringRef line, StringRef &r_line_key, StringRef &r_rest_line)
{
  if (line.is_empty()) {
    return;
  }

  const int64_t pos_split{line.find_first_of(' ')};
  if (pos_split == StringRef::not_found) {
    /* Use the first character if no space is found in the line. It's usually a comment like:
     * #This is a comment. */
    r_line_key = line.substr(0, 1);
  }
  else {
    r_line_key = line.substr(0, pos_split);
  }

  r_rest_line = line = line.drop_prefix(r_line_key.size());
  if (r_rest_line.is_empty()) {
    return;
  }

  /* Remove any leading spaces, trailing spaces & \r character, if any. */
  const int64_t leading_space{r_rest_line.find_first_not_of(' ')};
  if (leading_space != StringRef::not_found) {
    r_rest_line = r_rest_line.drop_prefix(leading_space);
  }

  /* Another way is to do a test run before the actual parsing to find the newline
   * character and use it in the getline. */
  const int64_t carriage_return{r_rest_line.find_first_of('\r')};
  if (carriage_return != StringRef::not_found) {
    r_rest_line = r_rest_line.drop_prefix(carriage_return);
  }

  const int64_t trailing_space{r_rest_line.find_last_not_of(' ')};
  if (trailing_space != StringRef::not_found) {
    /* The position is of a character that is not ' ', so count of characters is position + 1. */
    r_rest_line = r_rest_line.substr(0, trailing_space + 1);
  }
}

/**
 * Split the given string by the delimiter and fill the given vector.
 * If an intermediate string is empty, or space or null character, it is not appended to the
 * vector.
 */
void split_by_char(StringRef in_string, const char delimiter, Vector<StringRef> &r_out_list)
{
  r_out_list.clear();

  while (!in_string.is_empty()) {
    const int64_t pos_delim{in_string.find_first_of(delimiter)};
    const int64_t word_len = pos_delim == StringRef::not_found ? in_string.size() : pos_delim;

    StringRef word{in_string.data(), word_len};
    if (!word.is_empty() && !(word == " " && !(word[0] == '\0'))) {
      r_out_list.append(word);
    }
    if (pos_delim == StringRef::not_found) {
      return;
    }
    /* Skip the word already stored. */
    in_string = in_string.drop_prefix(word_len);
    /* Skip all delimiters. */
    in_string = in_string.drop_prefix(
        std::min(in_string.find_first_not_of(delimiter), in_string.size()));
  }
}

/**
 * Convert the given string to float and assign it to the destination value.
 *
 * Catches exception if the string cannot be converted to a float. The destination value
 * is set to the given fallback value in that case.
 */

void copy_string_to_float(StringRef src, const float fallback_value, float &r_dst)
{
  try {
    r_dst = std::stof(string(src));
  }
  catch (const std::invalid_argument &inv_arg) {
    std::cerr << "Bad conversion to float:'" << inv_arg.what() << "':'" << src << "'" << std::endl;
    r_dst = fallback_value;
  }
  catch (const std::out_of_range &out_of_range) {
    std::cerr << "Out of range for float:'" << out_of_range.what() << ":'" << src << "'"
              << std::endl;
    r_dst = fallback_value;
  }
}

/**
 * Convert all members of the Span of strings to floats and assign them to the float
 * array members. Usually used for values like coordinates.
 *
 * Catches exception if any string cannot be converted to a float. The destination
 * float is set to the given fallback value in that case.
 */
void copy_string_to_float(Span<StringRef> src,
                          const float fallback_value,
                          MutableSpan<float> r_dst)
{
  BLI_assert(src.size() >= r_dst.size());
  for (int i = 0; i < r_dst.size(); ++i) {
    copy_string_to_float(src[i], fallback_value, r_dst[i]);
  }
}

/**
 * Convert the given string to int and assign it to the destination value.
 *
 * Catches exception if the string cannot be converted to an integer. The destination
 * int is set to the given fallback value in that case.
 */
void copy_string_to_int(StringRef src, const int fallback_value, int &r_dst)
{
  try {
    r_dst = std::stoi(string(src));
  }
  catch (const std::invalid_argument &inv_arg) {
    std::cerr << "Bad conversion to int:'" << inv_arg.what() << "':'" << src << "'" << std::endl;
    r_dst = fallback_value;
  }
  catch (const std::out_of_range &out_of_range) {
    std::cerr << "Out of range for int:'" << out_of_range.what() << ":'" << src << "'"
              << std::endl;
    r_dst = fallback_value;
  }
}

/**
 * Convert the given strings to ints and fill the destination int buffer.
 *
 * Catches exception if any string cannot be converted to an integer. The destination
 * int is set to the given fallback value in that case.
 */
void copy_string_to_int(Span<StringRef> src, const int fallback_value, MutableSpan<int> r_dst)
{
  BLI_assert(src.size() >= r_dst.size());
  for (int i = 0; i < r_dst.size(); ++i) {
    copy_string_to_int(src[i], fallback_value, r_dst[i]);
  }
}

std::string replace_all_occurences(StringRef original, StringRef to_remove, StringRef to_add)
{
  std::string clean{original};
  while (true) {
    const std::string::size_type pos = clean.find(to_remove);
    if (pos == std::string::npos) {
      break;
    }
    clean.replace(pos, to_add.size(), to_add);
  }
  return clean;
}

}  // namespace blender::io::obj
