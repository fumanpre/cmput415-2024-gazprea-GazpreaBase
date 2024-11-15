
#include <assert.h>

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"

#include "BackEnd.h"

BackEnd::BackEnd() : loc(mlir::UnknownLoc::get(&context)) {
    // Load Dialects.
    context.loadDialect<mlir::LLVM::LLVMDialect>();
    context.loadDialect<mlir::arith::ArithDialect>();
    context.loadDialect<mlir::scf::SCFDialect>();
    context.loadDialect<mlir::cf::ControlFlowDialect>();
    context.loadDialect<mlir::memref::MemRefDialect>(); 

    // Initialize the MLIR context 
    builder = std::make_shared<mlir::OpBuilder>(&context);

    // Open a new context and module.
    module = mlir::ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());

    // initialize types
    intTy = builder->getI32Type();
    ptrTy = mlir::LLVM::LLVMPointerType::get(&context);
    memreftype = mlir::MemRefType::get({mlir::ShapedType::kDynamic}, intTy);

    setupPrintf();
}

void BackEnd::printInt(mlir::Value integer) {
    mlir::LLVM::GlobalOp formatString;
    if (!(formatString = module.lookupSymbol<mlir::LLVM::GlobalOp>("intFormat"))) {
        llvm::errs() << "missing format string!\n";
    }
    
    mlir::Value formatStringPtr = builder->create<mlir::LLVM::AddressOfOp>(loc, formatString);
    
    mlir::ValueRange args = {formatStringPtr, integer}; 
    
    mlir::LLVM::LLVMFuncOp printfFunc = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf");
    
    builder->create<mlir::LLVM::CallOp>(loc, printfFunc, args);
    
}

int BackEnd::emitMain() {
    mlir::Type intType = mlir::IntegerType::get(&context, 32);
    auto mainType = mlir::LLVM::LLVMFunctionType::get(intType, {}, false);
    mlir::LLVM::LLVMFuncOp mainFunc = builder->create<mlir::LLVM::LLVMFuncOp>(loc, "main", mainType);
    mlir::Block *entry = mainFunc.addEntryBlock();
    builder->setInsertionPointToStart(entry);

    printNewline();

    // Init constants
    constZero = builder->create<mlir::arith::ConstantIntOp>(loc, 0, intTy); 
    constOne = builder->create<mlir::arith::ConstantIntOp>(loc, 1, intTy);
    constMax = builder->create<mlir::arith::ConstantIntOp>(loc, 2147483647, intTy);

    mlir::Value zero = builder->create<mlir::LLVM::ConstantOp>(loc, intType, builder->getIntegerAttr(intType, 0));
    builder->create<mlir::LLVM::ReturnOp>(builder->getUnknownLoc(), zero);

    module.dump();

    if (mlir::failed(mlir::verify(module))) {
        module.emitError("module failed to verify");
        return -1;
    }
    return 0;
}

int BackEnd::lowerDialects() {
    // Set up the MLIR pass manager to iteratively lower all the Ops
    mlir::PassManager pm(&context);

    // Lower SCF to CF (ControlFlow)
    pm.addPass(mlir::createConvertSCFToCFPass());

    // Lower Arith to LLVM
    pm.addPass(mlir::createArithToLLVMConversionPass());

    // Lower MemRef to LLVM
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());

    // Lower CF to LLVM
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());

    // Finalize the conversion to LLVM dialect
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());

    // Run the passes
    if (mlir::failed(pm.run(module))) {
        llvm::errs() << "Pass pipeline failed\n";
        return 1;
    }
    return 0;
}

void BackEnd::dumpLLVM(std::ostream &os) {  
    // The only remaining dialects in our module after the passes are builtin
    // and LLVM. Setup translation patterns to get them to LLVM IR.
    mlir::registerBuiltinDialectTranslation(context);
    mlir::registerLLVMDialectTranslation(context);
    llvm_module = mlir::translateModuleToLLVMIR(module, llvm_context);

    // Create llvm ostream and dump into the output file
    llvm::raw_os_ostream output(os);
    output << *llvm_module;
}

void BackEnd::setupPrintf() {
    // Create the global string "\n"
    mlir::Type charType = mlir::IntegerType::get(&context, 8);
    auto gvalue = mlir::StringRef("\n\0", 2);
    auto type = mlir::LLVM::LLVMArrayType::get(charType, gvalue.size());
    builder->create<mlir::LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                               mlir::LLVM::Linkage::Internal, "newline",
                               builder->getStringAttr(gvalue), /*alignment=*/0);

    // Create a function declaration for printf, the signature is:
    //   * `i32 (i8*, ...)`
    mlir::Type intType = mlir::IntegerType::get(&context, 32);
    auto llvmI8PtrTy = mlir::LLVM::LLVMPointerType::get(charType);
    auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(intType, llvmI8PtrTy,
                                                        /*isVarArg=*/true);

    // Insert the printf function into the body of the parent module.
    builder->create<mlir::LLVM::LLVMFuncOp>(loc, "printf", llvmFnType);
}

void BackEnd::printNewline() {
    /* Note: a lot of this comes from the MLIR "toy" tutorial */
    mlir::LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<mlir::LLVM::GlobalOp>("newline"))) {
        llvm::errs() << "missing format string!\n";
        return;
    }

    // Get the pointer to the first character in the global string.
    mlir::Value globalPtr = builder->create<mlir::LLVM::AddressOfOp>(loc, global);
    mlir::Value cst0 = builder->create<mlir::LLVM::ConstantOp>(loc, builder->getI64Type(),
                                                        builder->getIndexAttr(0));

    mlir::Type charType = mlir::IntegerType::get(&context, 8);
    mlir::Value newLine = builder->create<mlir::LLVM::GEPOp>(loc,
                          mlir::LLVM::LLVMPointerType::get(charType),
                          globalPtr, mlir::ArrayRef<mlir::Value>({cst0, cst0}));

    mlir::LLVM::LLVMFuncOp printfFunc = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf");
    builder->create<mlir::LLVM::CallOp>(loc, printfFunc, newLine);
}
