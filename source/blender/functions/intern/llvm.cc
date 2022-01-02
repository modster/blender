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

#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/Analysis/LoopAccessAnalysis.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/ObjectCache.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Object/ObjectFile.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Vectorize.h>
#include <llvm/Transforms/Vectorize/LoopVectorize.h>

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

class FunctionOptimizer {
 private:
  llvm::LLVMContext context_;
  std::unique_ptr<llvm::ExecutionEngine> execution_engine_;

 public:
  AddFuncType generated_function;

 public:
  void initialize()
  {
    char const *args[] = {"some-random-name-for-the-parser", "--x86-asm-syntax=intel"};
    llvm::cl::ParseCommandLineOptions(std::size(args), args);

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
  }

  void generate_function()
  {
    std::unique_ptr<llvm::Module> module_owned = std::make_unique<llvm::Module>("My Module",
                                                                                context_);
    llvm::Module &ir_module = *module_owned;
    std::string error;
    const llvm::Target *target = llvm::TargetRegistry::lookupTarget(
        llvm::sys::getDefaultTargetTriple(), error);
    llvm::TargetOptions target_options;
    llvm::StringMap<bool> cpu_feature_map;
    llvm::sys::getHostCPUFeatures(cpu_feature_map);
    llvm::SubtargetFeatures subtarget_features;
    for (const auto &item : cpu_feature_map) {
      subtarget_features.AddFeature(item.getKey(), item.getValue());
    }
    llvm::TargetMachine *target_machine = target->createTargetMachine(
        llvm::sys::getDefaultTargetTriple(),
        llvm::sys::getHostCPUName().str(),
        subtarget_features.getString(),
        target_options,
        {},
        {},
        llvm::CodeGenOpt::Aggressive);

    ir_module.setDataLayout(target_machine->createDataLayout());
    ir_module.setTargetTriple(target_machine->getTargetTriple().normalize());

    std::cout << ir_module.getTargetTriple() << "\n";
    std::cout << ir_module.getDataLayout().getStringRepresentation() << "\n";

    llvm::Function &add_function = create_add_loop_function(ir_module);

    BLI_assert(!llvm::verifyModule(ir_module, &llvm::outs()));
    add_function.print(llvm::outs());

    llvm::LoopAnalysisManager loop_analysis_manager;
    llvm::FunctionAnalysisManager function_analysis_manager;
    llvm::CGSCCAnalysisManager cgscc_anaylysis_manager;
    llvm::ModuleAnalysisManager module_analysis_manager;

    llvm::PassBuilder pass_builder{false, target_machine};

    function_analysis_manager.registerPass([&] { return pass_builder.buildDefaultAAPipeline(); });

    pass_builder.registerModuleAnalyses(module_analysis_manager);
    pass_builder.registerCGSCCAnalyses(cgscc_anaylysis_manager);
    pass_builder.registerFunctionAnalyses(function_analysis_manager);
    pass_builder.registerLoopAnalyses(loop_analysis_manager);
    pass_builder.crossRegisterProxies(loop_analysis_manager,
                                      function_analysis_manager,
                                      cgscc_anaylysis_manager,
                                      module_analysis_manager);

    llvm::ModulePassManager module_pass_manager = pass_builder.buildPerModuleDefaultPipeline(
        llvm::PassBuilder::OptimizationLevel::O3);

    module_pass_manager.run(ir_module, module_analysis_manager);

    add_function.print(llvm::outs());

    llvm::EngineBuilder engine_builder{std::move(module_owned)};
    engine_builder.setOptLevel(llvm::CodeGenOpt::Aggressive);
    execution_engine_ = std::unique_ptr<llvm::ExecutionEngine>(
        engine_builder.create(target_machine));
    execution_engine_->finalizeObject();

    const uint64_t function_ptr = execution_engine_->getFunctionAddress(
        add_function.getName().str());

    this->generated_function = (AddFuncType)function_ptr;

    LLVMTargetMachineEmitToFile((LLVMTargetMachineRef)target_machine,
                                llvm::wrap(&ir_module),
                                (char *)"C:\\Users\\jacques\\Documents\\machine_code.txt",
                                LLVMAssemblyFile,
                                nullptr);
  }
};

AddFuncType get_compiled_add_function()
{
  static FunctionOptimizer *optimizer = []() {
    static FunctionOptimizer optimizer;
    optimizer.initialize();
    optimizer.generate_function();
    return &optimizer;
  }();
  return optimizer->generated_function;
}

}  // namespace blender::fn
