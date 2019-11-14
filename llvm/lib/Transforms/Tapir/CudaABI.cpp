#include <iostream>
#include <set>
#include <sstream>

#include "llvm/Transforms/Tapir/CudaABI.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Utils/EscapeEnumerator.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/TapirUtils.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Vectorize.h"
//#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/IR/LegacyPassManager.h"
//#include "llvm/Transforms/Tapir/Utils.h"

using namespace llvm;


namespace {

  template <class F>
  Function *getFunction(Module &M, const char *name) {
    return cast<Function>(M.getOrInsertFunction(name, 
                          TypeBuilder<F, false>::get(M.getContext())));
  }

  template <class B>
  Value *convertInteger(B &b, Value *from, Value *to, 
                        const std::string &name) 
  {
    auto fromType = dyn_cast<IntegerType>(from->getType());
    assert(fromType && "expect integer type!");

    auto toType = dyn_cast<IntegerType>(to->getType());
    assert(toType && "exepct integer type!");
    
    if (fromType->getBitWidth() > toType->getBitWidth()){
      return b.CreateTrunc(from, toType, name);
    } else if (fromType->getBitWidth() < toType->getBitWidth()) {
      return b.CreateZExt(from, toType, name);
    } else {
      return from;
    }
  }
}


CudaABI::CudaABI() { }

Value *CudaABI::GetOrCreateWorker8(Function &F) {
  Module *M = F.getParent();
  LLVMContext &C = M->getContext();
  return ConstantInt::get(C, APInt(16, 8));
}

void CudaABI::createSync(SyncInst &SI, 
                         ValueToValueMapTy &DetachCtxToStackFrame) 
{
    // currently a no-op... 
}                         

Function *CudaABI::createDetach(DetachInst &DI, 
                               ValueToValueMapTy &DetachCtxToStackFrame,
                               DominatorTree &DT, AssumptionCache &AC) 
{

  BasicBlock *detB     = DI.getParent();
  BasicBlock *Spawned  = DI.getDetached();
  BasicBlock *Continue = DI.getContinue();

  Instruction *CI = nullptr;
  Function *ExtractedFunc = extractDetachBodyToFunction(DI, DT, AC, &CI);

  // Replace the detach with a branch to the continuation. 
  BranchInst *ContinueBr = BranchInst::Create(Continue);
  ReplaceInstWithInst(&DI, ContinueBr);

  // Rewrite phis in the detached block... 
  {
    BasicBlock::iterator BI = Spawned->begin();
    while(PHINode *pn = dyn_cast<PHINode>(BI)) {
      pn->removeIncomingValue(detB);
      ++BI;
    }
  }
  return ExtractedFunc;
}

void CudaABI::preProcessFunction(Function &F) {
}

void CudaABI::postProcessFunction(Function &F) {
}

void CudaABI::postProcessingHelper(Function &F) {
}

bool CudaABI::processMain(Function &F) {
  return true;
}

bool CudaABILoopSpawning::processLoop() {
  Loop *L = OrigLoop;
  //L->dumpVerbose();

  // We are currently limited to a simple canonical loop structure
  // where we make the following assumptions and check assertions below. 
  // Soon we will expand this extraction mechanism to handle more 
  // complex loops. 

  using TypeVec  = std::vector<Type*>;
  using ValueVec = std::vector<Value*>;

  LLVMContext &Ctx = L->getHeader()->getContext();
  IRBuilder<>  Builder(Ctx);

  Type* voidTy = Type::getVoidTy(Ctx);
  IntegerType* i8Ty = Type::getInt8Ty(Ctx);
  IntegerType* i16Ty = Type::getInt16Ty(Ctx);
  IntegerType* i32Ty = Type::getInt32Ty(Ctx);
  IntegerType* i64Ty = Type::getInt64Ty(Ctx);
  PointerType* voidPtrTy = Type::getInt8PtrTy(Ctx);

  // An LLVM transformation is able (in some cases) to transform a 
  // loop to contain a phi node that exists at the entry block
  PHINode* loopNode = L->getCanonicalInductionVariable();
  assert(loopNode && "expected canonical loop");

  // Only handle loops where the induction variable is initialized 
  // to a constant integral value. 
  Value* loopStart = loopNode->getIncomingValue(0);
  assert(loopStart && "expected canonical loop start");

  auto cs = dyn_cast<ConstantInt>(loopStart);
  bool startsAtZero = cs && cs->isZero();

  BasicBlock* exitBlock = L->getUniqueExitBlock();
  assert(exitBlock && "expected canonical exit block");

  // And assume that a branch instruction exists here
  BasicBlock* branchBlock = exitBlock->getSinglePredecessor();
  assert(branchBlock && "expected canonical branch block");

  BranchInst* endBranch = dyn_cast<BranchInst>(branchBlock->getTerminator());
  assert(endBranch && "expected canonical end branch instruction");

  // Get the branch condition in order to extract the end loop value
  // which we also currently assume is constant
  Value* endBranchCond = endBranch->getCondition();
  CmpInst* cmp = dyn_cast<CmpInst>(endBranchCond);
  assert(cmp && "expected canonical comparison instruction");

  Value* loopEnd = cmp->getOperand(1);
  assert(loopEnd && "expected canonical loop end");

  BasicBlock* latchBlock = L->getLoopLatch();
  Instruction* li = latchBlock->getFirstNonPHI();
  unsigned op = li->getOpcode();
  assert(op == Instruction::Add || op == Instruction::Sub &&
         "expected add or sub in loop latch");
  assert(li->getOperand(0)== loopNode);
  Value* stride = li->getOperand(1);
  cs = dyn_cast<ConstantInt>(stride);
  bool isUnitStride = cs && cs->isOne();

  BasicBlock* entryBlock = L->getBlocks()[0];
  Function* hostFunc = entryBlock->getParent();
  Module& hostModule = *hostFunc->getParent();

  // Assume a detach exists here  and this basic block contains the body
  // of the kernel function we will be generating.
  DetachInst* detach = dyn_cast<DetachInst>(entryBlock->getTerminator());
  assert(detach && "expected canonical loop entry detach");

  BasicBlock* Body = detach->getDetached();

  // extract the externally defined variables
  // these will be passed in as CUDA arrays

  std::set<Value*> values;
  values.insert(loopNode);

  std::set<Value*> extValues;

  for(Instruction& ii : *Body){
    if (dyn_cast<ReattachInst>(&ii)) {
      continue;
    }

    for(Use& u : ii.operands()) {
      Value* v = u.get();
      if (isa<Constant>(v)) {
        continue;
      }
      if (values.find(v) == values.end()){
        extValues.insert(v);
      }
    }

    values.insert(&ii);
  }

  TypeVec paramTypes;
  paramTypes.push_back(i64Ty);
  paramTypes.push_back(i64Ty);
  paramTypes.push_back(i64Ty);

  for(Value* v : extValues){
    if(auto pt = dyn_cast<PointerType>(v->getType())){
      if(auto at = dyn_cast<ArrayType>(pt->getElementType())){
        paramTypes.push_back(PointerType::get(at->getElementType(), 0));
      } else {
        paramTypes.push_back(pt);
      }
    } else {
      v->dump();
      assert(false && "expected a pointer or array type");
    }
  }

  // Create the GPU function. 
  FunctionType* funcTy = FunctionType::get(voidTy, paramTypes, false);

  Module ptxModule("ptxModule", Ctx);

  // each kernel function is assigned a unique ID by which the kernel
  // entry point function is named e.g. run0 for kernel ID 0

  size_t kernelRunId = nextKernelId++;

  std::stringstream kstr;
  kstr << "run" << kernelRunId;

  Function* f = Function::Create(funcTy,
    Function::ExternalLinkage, kstr.str().c_str(), &ptxModule);

  // the first parameter defines the extent of the index space
  // i.e. number of threads to launch
  auto aitr = f->arg_begin();
  aitr->setName("runSize");
  Value* runSizeParam = aitr;
  ++aitr;
  aitr->setName("runStart");
  Value* runStartParam = aitr;
  ++aitr;

  aitr->setName("runStride");
  Value* runStrideParam = aitr;
  ++aitr;

  std::map<Value*, Value*> m;

  // set and parameter names and map values to be replaced

  size_t i = 0;

  for(Value* v : extValues){
    std::stringstream sstr;
    sstr << "arg" << i;

    m[v] = aitr;
    aitr->setName(sstr.str());
    ++aitr;
    ++i;
  }

  // Create the entry block which will be used to compute the thread ID
  // and simply return if the thread ID is beyond the run size

  BasicBlock* br = BasicBlock::Create(Ctx, "entry", f);

  Builder.SetInsertPoint(br);

  using SREGFunc = uint32_t();

  // calls to NVPTX intrinsics to get the thread index, block size,
  // and grid dimensions

  Value* threadIdx = Builder.CreateCall(getFunction<SREGFunc>(ptxModule,
    "llvm.nvvm.read.ptx.sreg.tid.x"));

  Value* blockIdx = Builder.CreateCall(getFunction<SREGFunc>(ptxModule,
    "llvm.nvvm.read.ptx.sreg.ctaid.x"));

  Value* blockDim = Builder.CreateCall(getFunction<SREGFunc>(ptxModule,
    "llvm.nvvm.read.ptx.sreg.ntid.x"));

  Value* threadId =
    Builder.CreateAdd(threadIdx, Builder.CreateMul(blockIdx, blockDim), "threadId");

  // Convert the thread ID into the proper integer type of the loop variable
  threadId = convertInteger(Builder, threadId, loopNode, "threadId");
  if (!isUnitStride) {
    threadId = Builder.CreateMul(threadId, runStrideParam);
  }
  if (!startsAtZero) {
    threadId = Builder.CreateAdd(threadId, runStartParam);
  }

  // return block to exit if thread ID is greater than or equal to run size
  BasicBlock* rb = BasicBlock::Create(Ctx, "exit", f);
  BasicBlock* bb = BasicBlock::Create(Ctx, "body", f);

  Value* cond = Builder.CreateICmpUGE(threadId, runSizeParam);
  Builder.CreateCondBr(cond, rb, bb);

  Builder.SetInsertPoint(rb);
  Builder.CreateRetVoid();

  Builder.SetInsertPoint(bb);

  // map the thread ID into the new values as we clone the instructions
  // of the function

  m[loopNode] = threadId;

  BasicBlock::InstListType& il = bb->getInstList();

  // clone instructions of the body basic block,  remapping values as needed

  std::set<Value*> extReads;
  std::set<Value*> extWrites;
  std::map<Value*, Value*> extVars;

  for(Instruction& ii : *Body){
    
    if (dyn_cast<ReattachInst>(&ii)) {
      continue;
    }

    // determine if we are reading or writing the external variables
    // i.e. those passed as CUDA arrays

    Instruction* ic = ii.clone();

    if (auto li = dyn_cast<LoadInst>(&ii)) {
      Value* v = li->getPointerOperand();
      auto itr = extVars.find(v);
      if (itr != extVars.end()) {
        extReads.insert(itr->second);
      }
    } else  if(auto si = dyn_cast<StoreInst>(&ii)){
      Value* v = si->getPointerOperand();
      auto itr = extVars.find(v);
      if (itr != extVars.end()) {
        extWrites.insert(itr->second);
      }
    } else if(auto gi = dyn_cast<GetElementPtrInst>(&ii)) {
      // if this is a GEP  into one of the external variables then keep track of
      // which external variable it originally came from.
          Value* v = gi->getPointerOperand();
      if (extValues.find(v) != extValues.end()) {
        extVars[gi] = v;
        if (isa<ArrayType>(gi->getSourceElementType())) {
          auto cgi = dyn_cast<GetElementPtrInst>(ic);
          cgi->setSourceElementType(m[v]->getType());
        }
      }
    }  

    // remap values as we are cloning the instructions

    for(auto& itr : m){
      ic->replaceUsesOfWith(itr.first, itr.second);
    }

    il.push_back(ic);
    m[&ii] = ic;
  }

  Builder.CreateRetVoid();

  // add the necessary NVPTX to mark the global function
  NamedMDNode* annotations =
    ptxModule.getOrInsertNamedMetadata("nvvm.annotations");

  SmallVector<Metadata*, 3> av;

  av.push_back(ValueAsMetadata::get(f));
  av.push_back(MDString::get(ptxModule.getContext(), "kernel"));
  av.push_back(ValueAsMetadata::get(llvm::ConstantInt::get(i32Ty, 1)));

  annotations->addOperand(MDNode::get(ptxModule.getContext(), av));

  // remove the basic blocks corresponding to the original LLVM loop

  BasicBlock* predecessor = L->getLoopPreheader();
  entryBlock->removePredecessor(predecessor);
  BasicBlock* successor = exitBlock->getSingleSuccessor();

  BasicBlock* hostBlock = BasicBlock::Create(Ctx, "host.block", hostFunc);

  Builder.SetInsertPoint(predecessor->getTerminator());
  Builder.CreateBr(hostBlock);
  predecessor->getTerminator()->removeFromParent();

successor->removePredecessor(exitBlock);

  {
    std::set<BasicBlock*> visited;
    visited.insert(exitBlock);

    std::vector<BasicBlock*> next;
    next.push_back(entryBlock);

    while(!next.empty()){
      BasicBlock* b = next.back();
      next.pop_back();

      for(BasicBlock* bn : successors(b)) {
        if(visited.find(bn) == visited.end()){
          next.push_back(bn);
        }
      }

      b->dropAllReferences();
      b->removeFromParent();
      visited.insert(b);
    }
  }

  exitBlock->dropAllReferences();
  exitBlock->removeFromParent();

// find the NVPTX module pass which will create the PTX code

  const Target* target = nullptr;

  for(TargetRegistry::iterator itr =  TargetRegistry::targets().begin(),
      itrEnd =  TargetRegistry::targets().end(); itr != itrEnd; ++itr){
    if(std::string(itr->getName()) == "nvptx64"){
      target = &*itr;
      break;
    }
  }

  assert(target && "failed to find NVPTX target");

  Triple triple(sys::getDefaultTargetTriple());
  triple.setArch(Triple::nvptx64);

  TargetMachine* targetMachine =
      target->createTargetMachine(triple.getTriple(),
                                  "sm_70",
                                  "",
                                  TargetOptions(),
                                  Reloc::Static,
                                  CodeModel::Kernel,
                                  CodeGenOpt::Aggressive);

  DataLayout layout("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:"
    "64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:"
    "64:64-v128:128:128-n16:32:64");

  ptxModule.setDataLayout(layout);

  legacy::PassManager* passManager = new legacy::PassManager;

  passManager->add(createVerifierPass());

  // add in our optimization passes

  //passManager->add(createInstructionCombiningPass());
  passManager->add(createReassociatePass());
  passManager->add(createGVNPass());
  passManager->add(createCFGSimplificationPass());
  passManager->add(createSLPVectorizerPass());
  //passManager->add(createBreakCriticalEdgesPass());
  passManager->add(createConstantPropagationPass());
  passManager->add(createDeadInstEliminationPass());
  passManager->add(createDeadStoreEliminationPass());
  //passManager->add(createInstructionCombiningPass());
  passManager->add(createCFGSimplificationPass());

  SmallVector<char, 65536> buf;
  raw_svector_ostream ostr(buf);

  bool fail =
  targetMachine->addPassesToEmitFile(*passManager,
                                     ostr,
                                     &ostr,
                                     TargetMachine::CodeGenFileType::CGFT_AssemblyFile,
                                     false);

  assert(!fail && "failed to emit PTX");

  passManager->run(ptxModule);

  delete passManager;

  std::string ptx = ostr.str().str();

  Constant* pcs = ConstantDataArray::getString(Ctx, ptx);

  // create a global string to hold the PTX code

  GlobalVariable* ptxGlobal =
    new GlobalVariable(hostModule,
                       pcs->getType(),
                       true,
                       GlobalValue::PrivateLinkage,
                       pcs,
                       "ptx");

  Value* kernelId = ConstantInt::get(i32Ty, kernelRunId);

  Value* ptxStr = Builder.CreateBitCast(ptxGlobal, voidPtrTy);

  Builder.SetInsertPoint(hostBlock);

  // finally, replace where the original loop was with calls to the GPU runtime

  using InitCUDAFunc = void();

  Builder.CreateCall(getFunction<InitCUDAFunc>(hostModule,
      "__kitsune_cuda_init"), {});

  using InitKernelFunc = void(uint32_t, const char*);

  Builder.CreateCall(getFunction<InitKernelFunc>(hostModule,
      "__kitsune_gpu_init_kernel"), {kernelId, ptxStr});

  for(Value* v : extValues){
    Value* elementSize = nullptr;
    Value* vptr;
    Value* fieldName;
    Value* size = nullptr;

    // TODO: fix
    // this is a temporary hack to get the size of the field
    // it will currently only work for a limited case

    if (auto bc = dyn_cast<BitCastInst>(v)) {
      auto ci = dyn_cast<CallInst>(bc->getOperand(0));
      assert(ci && "unable to detect field size");

      Value* bytes = ci->getOperand(0);
      assert(bytes->getType()->isIntegerTy(64));

      auto pt = dyn_cast<PointerType>(v->getType());
      auto it = dyn_cast<IntegerType>(pt->getElementType());
      assert(it && "expected integer type");

      Constant* fn = ConstantDataArray::getString(Ctx, ci->getName());

      GlobalVariable* fieldNameGlobal =
        new GlobalVariable(hostModule,
                           fn->getType(),
                           true,
                           GlobalValue::PrivateLinkage,
                           fn,
                           "field.name");
      fieldName = Builder.CreateBitCast(fieldNameGlobal, voidPtrTy);
      vptr = Builder.CreateBitCast(v, voidPtrTy);

      elementSize = ConstantInt::get(i32Ty, it->getBitWidth()/8);

      size = Builder.CreateUDiv(bytes, ConstantInt::get(i64Ty, it->getBitWidth()/8));
    } else if(auto ai = dyn_cast<AllocaInst>(v)) {
      Constant* fn = ConstantDataArray::getString(Ctx, ai->getName());

      GlobalVariable* fieldNameGlobal =
        new GlobalVariable(hostModule,
                           fn->getType(),
                           true,
                           GlobalValue::PrivateLinkage,
                           fn,
                           "field.name");

      fieldName = Builder.CreateBitCast(fieldNameGlobal, voidPtrTy);

      vptr = Builder.CreateBitCast(v, voidPtrTy);

      auto at = dyn_cast<ArrayType>(ai->getAllocatedType());
      assert(at && "expected array type");

      elementSize = ConstantInt::get(i32Ty,
        at->getElementType()->getPrimitiveSizeInBits()/8);

      size = ConstantInt::get(i64Ty, at->getNumElements());
    }

    uint8_t m = 0;
    if (extReads.find(v) != extReads.end()) {
      m |= 0b01;
    }

    if (extWrites.find(v) != extWrites.end()) {
      m |= 0b10;
    }

  Value* mode = ConstantInt::get(i8Ty, m);

    TypeVec params = {i32Ty, voidPtrTy, voidPtrTy, i32Ty, i64Ty, i8Ty};

    Function* initFieldFunc =
      llvm::Function::Create(FunctionType::get(voidTy, params, false),
                             llvm::Function::ExternalLinkage,
                             "__kitsune_gpu_init_field",
                             &hostModule);

    Builder.CreateCall(initFieldFunc,
      {kernelId, fieldName, vptr, elementSize, size, mode});
  }

  using SetRunSizeFunc = void(uint32_t, uint64_t, uint64_t, uint64_t);

  Value* runSize = Builder.CreateSub(loopEnd, loopStart);

  runSize = convertInteger(Builder, runSize, threadId, "run.size");

  Value* runStart = convertInteger(Builder, loopStart, threadId, "run.start");

  Builder.CreateCall(getFunction<SetRunSizeFunc>(hostModule,
    "__kitsune_gpu_set_run_size"), {kernelId, runSize, runStart, runStart});

  using RunKernelFunc = void(uint32_t);

  Builder.CreateCall(getFunction<RunKernelFunc>(hostModule,
    "__kitsune_gpu_run_kernel"), {kernelId});

  using FinishFunc = void();

  Builder.CreateCall(getFunction<FinishFunc>(hostModule,
    "__kitsune_gpu_finish"), {});

  Builder.CreateBr(successor);

  // hostModule.dump();

  // ptxModule.dump();

  return true;
}
  