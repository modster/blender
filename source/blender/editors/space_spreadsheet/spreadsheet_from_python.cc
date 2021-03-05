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
 */

#include <Python.h>

#include "RNA_access.h"

#include "DNA_space_types.h"

#include "BKE_context.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "../../python/intern/bpy_rna.h"

#include "spreadsheet_from_python.hh"

namespace blender::ed::spreadsheet {

class PythonSpreadsheetDrawer : public SpreadsheetDrawer {
 private:
  PyObject *py_drawer_;

 public:
  PythonSpreadsheetDrawer(PyObject *py_drawer) : py_drawer_(py_drawer)
  {
    BLI_assert(py_drawer_ != nullptr);
    BLI_assert(py_drawer_ != Py_None);
    Py_INCREF(py_drawer_);

    PyObject *py_column_amount = PyObject_CallMethod(py_drawer_, "get_column_amount", "");
    this->tot_columns = PyLong_AsLong(py_column_amount);
    Py_DECREF(py_column_amount);

    PyObject *py_row_amount = PyObject_CallMethod(py_drawer_, "get_row_amount", "");
    this->tot_rows = PyLong_AsLong(py_row_amount);
    Py_DecRef(py_row_amount);
  }

  ~PythonSpreadsheetDrawer() override
  {
    PyGILState_STATE gilstate = PyGILState_Ensure();
    Py_DECREF(py_drawer_);
    PyGILState_Release(gilstate);
  }

  void draw_top_row_cell(int column_index, const CellDrawParams &params) const override
  {
    PyGILState_STATE gilstate = PyGILState_Ensure();
    PyObject *py_cell_content = PyObject_CallMethod(
        py_drawer_, "get_top_row_cell", "i", column_index);
    this->draw_cell_content(params, py_cell_content);
    Py_DecRef(py_cell_content);
    PyGILState_Release(gilstate);
  }

  void draw_left_column_cell(int row_index, const CellDrawParams &params) const override
  {
    PyGILState_STATE gilstate = PyGILState_Ensure();
    PyObject *py_cell_content = PyObject_CallMethod(
        py_drawer_, "get_left_column_cell", "i", row_index);
    this->draw_cell_content(params, py_cell_content);
    Py_DecRef(py_cell_content);
    PyGILState_Release(gilstate);
  }

  void draw_content_cell(int row_index,
                         int column_index,
                         const CellDrawParams &params) const override
  {
    PyGILState_STATE gilstate = PyGILState_Ensure();
    PyObject *py_cell_content = PyObject_CallMethod(
        py_drawer_, "get_content_cell", "ii", row_index, column_index);
    this->draw_cell_content(params, py_cell_content);
    Py_DecRef(py_cell_content);
    PyGILState_Release(gilstate);
  }

 private:
  void draw_cell_content(const CellDrawParams &params, PyObject *py_cell_content) const
  {
    if (py_cell_content == nullptr) {
      return;
    }
    if (py_cell_content == Py_None) {
      return;
    }
    if (PyUnicode_Check(py_cell_content)) {
      const char *str = PyUnicode_AsUTF8(py_cell_content);
      uiDefIconTextBut(params.block,
                       UI_BTYPE_LABEL,
                       0,
                       ICON_NONE,
                       str,
                       params.xmin,
                       params.ymin,
                       params.width,
                       params.height,
                       nullptr,
                       0,
                       0,
                       0,
                       0,
                       nullptr);
    }
  }
};

std::unique_ptr<SpreadsheetDrawer> spreadsheet_drawer_from_python(const bContext *C)
{
  SpaceSpreadsheet *sspreadsheet = CTX_wm_space_spreadsheet(C);
  PointerRNA sspreadsheet_rna;
  RNA_pointer_create(nullptr, &RNA_SpaceSpreadsheet, sspreadsheet, &sspreadsheet_rna);

  std::unique_ptr<SpreadsheetDrawer> drawer;

  PyGILState_STATE gilstate = PyGILState_Ensure();
  PyObject *py_module = PyImport_ImportModule("bpy_spreadsheet");
  PyObject *py_get_drawer_func = PyObject_GetAttrString(py_module, "get_spreadsheet_drawer");
  PyObject *py_sspreadsheet = pyrna_struct_CreatePyObject(&sspreadsheet_rna);
  PyObject *py_drawer = PyObject_CallFunction(py_get_drawer_func, "O", py_sspreadsheet);
  if (py_drawer != Py_None) {
    drawer = std::make_unique<PythonSpreadsheetDrawer>(py_drawer);
  }
  Py_DECREF(py_drawer);
  Py_DECREF(py_sspreadsheet);
  PyGILState_Release(gilstate);
  return drawer;
}

}  // namespace blender::ed::spreadsheet
