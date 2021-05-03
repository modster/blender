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
 * The Original Code is Copyright (C) 2016 Kévin Dietrich.
 * All rights reserved.
 */

/** \file
 * \ingroup balembic
 */

#include "abc_reader_archive.h"

#include "BKE_main.h"

#include "BLI_path_util.h"
#include "BLI_string.h"

#ifdef WIN32
#  include "utfconv.h"
#endif

#include <fstream>

using Alembic::Abc::chrono_t;
using Alembic::Abc::ErrorHandler;
using Alembic::Abc::Exception;
using Alembic::Abc::IArchive;
using Alembic::Abc::kWrapExisting;
using Alembic::Abc::TimeSamplingPtr;
using Alembic::Abc::TimeSamplingType;

namespace blender::io::alembic {

static IArchive open_archive(const std::string &filename,
                             const std::vector<std::istream *> &input_streams)
{
  try {
    Alembic::AbcCoreOgawa::ReadArchive archive_reader(input_streams);

    return IArchive(archive_reader(filename), kWrapExisting, ErrorHandler::kThrowPolicy);
  }
  catch (const Exception &e) {
    std::cerr << e.what() << '\n';

    /* Inspect the file to see whether it's actually a HDF5 file. */
    char header[4]; /* char(0x89) + "HDF" */
    std::ifstream the_file(filename.c_str(), std::ios::in | std::ios::binary);
    if (!the_file) {
      std::cerr << "Unable to open " << filename << std::endl;
    }
    else if (!the_file.read(header, sizeof(header))) {
      std::cerr << "Unable to read from " << filename << std::endl;
    }
    else if (strncmp(header + 1, "HDF", 3) != 0) {
      std::cerr << filename << " has an unknown file format, unable to read." << std::endl;
    }
    else {
      std::cerr << filename << " is in the obsolete HDF5 format, unable to read." << std::endl;
    }

    if (the_file.is_open()) {
      the_file.close();
    }
  }

  return IArchive();
}

ArchiveReader::ArchiveReader(struct Main *bmain, const char *filename)
{
  char abs_filename[FILE_MAX];
  BLI_strncpy(abs_filename, filename, FILE_MAX);
  BLI_path_abs(abs_filename, BKE_main_blendfile_path(bmain));

#ifdef WIN32
  UTF16_ENCODE(abs_filename);
  std::wstring wstr(abs_filename_16);
  m_infile.open(wstr.c_str(), std::ios::in | std::ios::binary);
  UTF16_UN_ENCODE(abs_filename);
#else
  m_infile.open(abs_filename, std::ios::in | std::ios::binary);
#endif

  m_streams.push_back(&m_infile);

  m_archive = open_archive(abs_filename, m_streams);
}

bool ArchiveReader::valid() const
{
  return m_archive.valid();
}

Alembic::Abc::IObject ArchiveReader::getTop()
{
  return m_archive.getTop();
}

TimeInfo ArchiveReader::getTimeInfo()
{
  const uint32_t num_time_sampling_ptrs = m_archive.getNumTimeSamplings();

  chrono_t min_time = std::numeric_limits<chrono_t>::max();
  chrono_t max_time = -std::numeric_limits<chrono_t>::max();

  for (uint32_t i = 0; i < num_time_sampling_ptrs; ++i) {
    const Alembic::Abc::index_t max_samples = m_archive.getMaxNumSamplesForTimeSamplingIndex(i);

    /* This can only happen in very old files, predating the original Blender Alembic support,
     * however let's make sure this case is handled. */
    if (max_samples == INDEX_UNKNOWN) {
      continue;
    }

    const TimeSamplingPtr time_sampling_ptr = m_archive.getTimeSampling(i);
    assert(time_sampling_ptr);

    const TimeSamplingType &time_sampling_type = time_sampling_ptr->getTimeSamplingType();

    /* Avoid the default time sampling, it should be at index 0, but we never know. */
    if (time_sampling_ptr->getNumStoredTimes() == 1 &&
        time_sampling_ptr->getStoredTimes()[0] == 0.0 &&
        time_sampling_type.getTimePerCycle() == 1.0) {
      continue;
    }

    min_time = std::min(min_time, time_sampling_ptr->getSampleTime(0));
    max_time = std::max(max_time, time_sampling_ptr->getSampleTime(max_samples - 1));
  }

  return {min_time, max_time};
}

}  // namespace blender::io::alembic
