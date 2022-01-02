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

#include <fstream>
#include <iostream>

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/ObjectCache.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Object/ObjectFile.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>

#include "FN_llvm.hh"

#include "BLI_vector.hh"

namespace blender::fn {

static llvm::Function &create_add_loop_function(llvm::Module &module)
{
  llvm::LLVMContext &context = module.getContext();
  llvm::IRBuilder<> builder{context};
  llvm::FunctionType &function_type = *llvm::FunctionType::get(
      builder.getVoidTy(),
      {builder.getFloatTy()->getPointerTo(),
       builder.getFloatTy(),
       builder.getFloatTy()->getPointerTo(),
       builder.getInt64Ty()},
      false);
  llvm::Function &function = *llvm::Function::Create(
      &function_type,
      llvm::GlobalValue::LinkageTypes::ExternalLinkage,
      "Add Span and Single",
      &module);

  llvm::Value *src_array_ptr_v = function.getArg(0);
  llvm::Value *src_value_v = function.getArg(1);
  llvm::Value *dst_array_ptr_v = function.getArg(2);
  llvm::Value *array_size_v = function.getArg(3);

  llvm::BasicBlock *entry_bb = llvm::BasicBlock::Create(context, "entry", &function);
  llvm::BasicBlock *loop_entry_bb = llvm::BasicBlock::Create(context, "loop_entry", &function);
  llvm::BasicBlock *loop_body_bb = llvm::BasicBlock::Create(context, "loop_body", &function);
  llvm::BasicBlock *loop_end_bb = llvm::BasicBlock::Create(context, "loop_end", &function);

  builder.SetInsertPoint(entry_bb);
  builder.CreateBr(loop_entry_bb);

  builder.SetInsertPoint(loop_entry_bb);
  llvm::PHINode *index_v = builder.CreatePHI(builder.getInt64Ty(), 2);
  index_v->addIncoming(builder.getInt64(0), entry_bb);
  llvm::Value *is_less_than_v = builder.CreateICmpSLT(index_v, array_size_v);
  builder.CreateCondBr(is_less_than_v, loop_body_bb, loop_end_bb);

  builder.SetInsertPoint(loop_body_bb);
  llvm::Value *load_ptr_v = builder.CreateGEP(src_array_ptr_v, index_v);
  llvm::Value *store_ptr_v = builder.CreateGEP(dst_array_ptr_v, index_v);
  llvm::Value *value_from_array_v = builder.CreateLoad(builder.getFloatTy(), load_ptr_v);
  llvm::Value *new_value_v = builder.CreateFAdd(value_from_array_v, src_value_v);
  builder.CreateStore(new_value_v, store_ptr_v);
  llvm::Value *next_index_v = builder.CreateAdd(index_v, builder.getInt64(1));
  index_v->addIncoming(next_index_v, loop_body_bb);
  builder.CreateBr(loop_entry_bb);

  builder.SetInsertPoint(loop_end_bb);
  builder.CreateRetVoid();

  return function;
}

void playground()
{
  std::cout << "Start\n";

  static bool initialized = []() {
    /* Set assembly syntax flavour. */
    char const *args[] = {"some-random-name-for-the-parser", "--x86-asm-syntax=intel"};
    llvm::cl::ParseCommandLineOptions(std::size(args), args);

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    return true;
  }();
  UNUSED_VARS(initialized);

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = std::make_unique<llvm::Module>("My Module", context);

  llvm::legacy::FunctionPassManager function_pass_manager{module.get()};
  function_pass_manager.add(llvm::createInstructionCombiningPass());
  std::cout << "A\n";

  llvm::Function &add_function = create_add_loop_function(*module);

  BLI_assert(!llvm::verifyModule(*module, &llvm::outs()));

  llvm::Module *module_ptr = &*module;
  std::unique_ptr<llvm::ExecutionEngine> ee{llvm::EngineBuilder(std::move(module)).create()};
  ee->finalizeObject();

  const uint64_t function_ptr = ee->getFunctionAddress(add_function.getName().str());
  using FuncType = void (*)(const float *, float, float *, int);
  const FuncType generated_function = (FuncType)function_ptr;
  UNUSED_VARS(generated_function, module_ptr);

  /*
  LLVMTargetMachineEmitToFile((LLVMTargetMachineRef)ee->getTargetMachine(),
                              llvm::wrap(module_ptr),
                              (char *)"C:\\Users\\jacques\\Documents\\machine_code.txt",
                              LLVMAssemblyFile,
                              nullptr);
  */
  add_function.print(llvm::outs());
}

}  // namespace blender::fn
