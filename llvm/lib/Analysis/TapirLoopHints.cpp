//=- TapirLoopHints.cpp - Utilities for metadata on tapir loops -------------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for hints on tapir loops
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/TapirLoopHints.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Metadata.h"

using namespace llvm;

/// Find hints specified in the loop metadata and update local values.
void llvm::TapirLoopHints::getHintsFromMetadata() {
  MDNode *LoopID = TheLoop->getLoopID();
  if (!LoopID)
    return;

  // First operand should refer to the loop id itself.
  assert(LoopID->getNumOperands() > 0 && "requires at least one operand");
  assert(LoopID->getOperand(0) == LoopID && "invalid loop id");

  for (unsigned i = 1, ie = LoopID->getNumOperands(); i < ie; ++i) {
    const MDString *S = nullptr;
    SmallVector<Metadata *, 4> Args;

    // The expected hint is either a MDString or a MDNode with the first
    // operand a MDString.
    if (const MDNode *MD = dyn_cast<MDNode>(LoopID->getOperand(i))) {
      if (!MD || MD->getNumOperands() == 0)
        continue;
      S = dyn_cast<MDString>(MD->getOperand(0));
      for (unsigned i = 1, ie = MD->getNumOperands(); i < ie; ++i)
        Args.push_back(MD->getOperand(i));
    } else {
      S = dyn_cast<MDString>(LoopID->getOperand(i));
      assert(Args.size() == 0 && "too many arguments for MDString");
    }

    if (!S)
      continue;

    // Check if the hint starts with the loop metadata prefix.
    StringRef Name = S->getString();
    if (Args.size() == 1)
      setHint(Name, Args[0]);
  }
}

bool llvm::TapirLoopHints::validate(StringRef Name, unsigned V) {
  if (Name == nameStrategy) {
    switch (TapirSpawnStrategy(V)) {
    case TapirSpawnStrategy::Sequential:
    case TapirSpawnStrategy::DivideAndConquer:
    case TapirSpawnStrategy::GPU:
      return true;
    default:
      return false;
    }
  } else if (Name == nameGrainSize) {
    return true;
  } else if (Name == nameLoopTarget) {
    if (std::optional<TTID> TT = createTTIDFrom(V)) {
      switch (*TT) {
      case TTID::None:
      case TTID::Serial:
      case TTID::Cuda:
      case TTID::Hip:
      case TTID::Lambda:
      case TTID::OMPTask:
      case TTID::OpenCilk:
      case TTID::OpenMP:
      case TTID::Qthreads:
      case TTID::Realm:
        return true;
      default:
        return false;
      }
    } else {
      return false;
    }
  } else if (Name == nameThreadsPerBlock) {
    return V <= KITSUNE_MAX_FIXED_THREADS_PER_BLOCK;
  } else if (Name == nameAutotuneLaunch) {
    return true;
  } else {
    llvm_unreachable("TapirLoopHints::validate: Name not handled");
  }
}

bool llvm::TapirLoopHints::canCreateMetadata(StringRef Name,
                                             const ValueType &V) const {
  if (Name == nameLoopTarget)
    return getLoopTarget().has_value();
  return true;
}

unsigned llvm::TapirLoopHints::toMetadataValue(
    StringRef Name, const llvm::TapirLoopHints::ValueType &V) const {
  assert(canCreateMetadata(Name, V) && "Cannot get metadata value for hint");
  if (std::holds_alternative<bool>(V))
    return std::get<bool>(V);
  else if (std::holds_alternative<unsigned>(V))
    return std::get<unsigned>(V);
  else if (std::holds_alternative<TapirSpawnStrategy>(V))
    return unsigned(std::get<TapirSpawnStrategy>(V));
  else if (std::holds_alternative<std::optional<TTID>>(V))
    return unsigned(*std::get<std::optional<TTID>>(V));
  else
    llvm_unreachable("toMetadataValue: type not handled");
}

void llvm::TapirLoopHints::setHint(StringRef Name, Metadata *Arg) {
  if (!Name.starts_with(namePrefix))
    return;
  const ConstantInt *C = mdconst::dyn_extract<ConstantInt>(Arg);
  if (!C)
    return;

  unsigned Val = C->getZExtValue();
  if (not TapirLoopHints::validate(Name, Val))
    report_fatal_error(Twine("Invalid loop hint value: '") + Name + "'");
  else if (Name == nameStrategy)
    hints[Name] = TapirSpawnStrategy(Val);
  else if (Name == nameGrainSize)
    hints[Name] = Val;
  else if (Name == nameLoopTarget)
    hints[Name] = TTID(Val);
  else if (Name == nameThreadsPerBlock)
    hints[Name] = Val;
  else if (Name == nameAutotuneLaunch)
    hints[Name] = bool(Val);
  else
    llvm_unreachable("TapirLoopHints::setHint: Hint name not handled");
}

/// Create a new hint from name / value pair.
MDNode *llvm::TapirLoopHints::createHintMetadata(StringRef Name,
                                                 unsigned V) const {
  LLVMContext &Context = TheLoop->getHeader()->getContext();
  Metadata *MDs[] = {
      MDString::get(Context, Name),
      ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Context), V))};
  return MDNode::get(Context, MDs);
}

/// Matches metadata with hint name.
bool llvm::TapirLoopHints::matchesHintMetadataName(MDNode *Node,
                                                   const Hints &Hints) const {
  MDString *Name = dyn_cast<MDString>(Node->getOperand(0));
  if (!Name)
    return false;

  // KITSUNE FIXME: Search for the full name.
  for (const auto &i : Hints)
    if (Name->getString().ends_with(i.first))
      return true;
  return false;
}

/// Sets current hints into loop metadata, keeping other values intact.
void llvm::TapirLoopHints::writeHintsToMetadata(const Hints &Hints) {
  if (Hints.size() == 0)
    return;

  LLVMContext &Context = TheLoop->getHeader()->getContext();
  SmallVector<Metadata *, 4> MDs;

  // Reserve first location for self reference to the LoopID metadata node.
  TempMDTuple TempNode = MDNode::getTemporary(Context, std::nullopt);
  MDs.push_back(TempNode.get());

  // If the loop already has metadata, then ignore the existing operands.
  MDNode *LoopID = TheLoop->getLoopID();
  if (LoopID) {
    for (unsigned i = 1, ie = LoopID->getNumOperands(); i < ie; ++i) {
      MDNode *Node = cast<MDNode>(LoopID->getOperand(i));
      // If node in update list, ignore old value.
      if (!matchesHintMetadataName(Node, Hints))
        MDs.push_back(Node);
    }
  }

  // Now, add the missing hints.
  for (const auto &i : Hints) {
    StringRef Name = i.first;
    const ValueType &V = i.second;
    if (canCreateMetadata(Name, V))
      MDs.push_back(createHintMetadata(Name, toMetadataValue(Name, V)));
  }

  // Replace current metadata node with new one.
  MDNode *NewLoopID = MDNode::get(Context, MDs);
  // Set operand 0 to refer to the loop id itself.
  NewLoopID->replaceOperandWith(0, NewLoopID);

  TheLoop->setLoopID(NewLoopID);
}

/// Sets current hints into loop metadata, keeping other values intact.
void llvm::TapirLoopHints::writeHintsToClonedMetadata(const Hints &Hints,
                                                      ValueToValueMapTy &VMap) {
  if (Hints.size() == 0)
    return;

  LLVMContext &Context =
      cast<BasicBlock>(VMap[TheLoop->getHeader()])->getContext();
  SmallVector<Metadata *, 4> MDs;

  // Reserve first location for self reference to the LoopID metadata node.
  TempMDTuple TempNode = MDNode::getTemporary(Context, std::nullopt);
  MDs.push_back(TempNode.get());

  // If the loop already has metadata, then ignore the existing operands.
  MDNode *OrigLoopID = TheLoop->getLoopID();
  if (!OrigLoopID)
    return;

  if (MDNode *LoopID = dyn_cast_or_null<MDNode>(VMap.MD()[OrigLoopID])) {
    for (unsigned i = 1, ie = LoopID->getNumOperands(); i < ie; ++i) {
      MDNode *Node = cast<MDNode>(LoopID->getOperand(i));
      // If node in update list, ignore old value.
      if (!matchesHintMetadataName(Node, Hints))
        MDs.push_back(Node);
    }
  }

  // Now, add the missing hints.
  for (const auto &i : Hints) {
    StringRef Name = i.first;
    const ValueType &V = i.second;
    if (canCreateMetadata(Name, V))
      MDs.push_back(createHintMetadata(Name, toMetadataValue(Name, V)));
  }

  // Replace current metadata node with new one.
  MDNode *NewLoopID = MDNode::get(Context, MDs);
  // Set operand 0 to refer to the loop id itself.
  NewLoopID->replaceOperandWith(0, NewLoopID);

  // Set the metadata on the terminator of the cloned loop's latch.
  BasicBlock *ClonedLatch = cast<BasicBlock>(VMap[TheLoop->getLoopLatch()]);
  assert(ClonedLatch && "Cloned Tapir loop does not have a single latch.");
  ClonedLatch->getTerminator()->setMetadata(LLVMContext::MD_loop, NewLoopID);
}

void llvm::TapirLoopHints::clearHintsMetadata() {
  LLVMContext &Context = TheLoop->getHeader()->getContext();
  SmallVector<Metadata *, 4> MDs;

  // Reserve first location for self reference to the LoopID metadata node.
  TempMDTuple TempNode = MDNode::getTemporary(Context, std::nullopt);
  MDs.push_back(TempNode.get());

  // If the loop already has metadata, then ignore the existing operands.
  MDNode *LoopID = TheLoop->getLoopID();
  if (LoopID) {
    for (unsigned i = 1, ie = LoopID->getNumOperands(); i < ie; ++i) {
      MDNode *Node = cast<MDNode>(LoopID->getOperand(i));
      // If node in update list, ignore old value.
      if (!matchesHintMetadataName(Node, hints))
        MDs.push_back(Node);
    }
  }

  // Replace current metadata node with new one.
  MDNode *NewLoopID = MDNode::get(Context, MDs);
  // Set operand 0 to refer to the loop id itself.
  NewLoopID->replaceOperandWith(0, NewLoopID);

  TheLoop->setLoopID(NewLoopID);
}

/// Returns true if Tapir-loop hints require loop outlining during lowering.
bool llvm::hintsDemandOutlining(const TapirLoopHints &Hints) {
  switch (Hints.getStrategy()) {
  case TapirSpawnStrategy::DivideAndConquer:
  case TapirSpawnStrategy::GPU:
    return true;
  case TapirSpawnStrategy::Sequential:
    return false;
  default:
    llvm_unreachable("hintsDemandOutlining: SpawningStrategy not handled");
  }
}
