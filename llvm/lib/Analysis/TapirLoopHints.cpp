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
  MDNode *loopID = theLoop->getLoopID();
  if (!loopID)
    return;

  // First operand should refer to the loop id itself.
  assert(loopID->getNumOperands() > 0 && "requires at least one operand");
  assert(loopID->getOperand(0) == loopID && "invalid loop id");

  for (unsigned i = 1, ie = loopID->getNumOperands(); i < ie; ++i) {
    const MDString *s = nullptr;
    SmallVector<Metadata *, 4> args;

    // The expected hint is either a MDString or a MDNode with the first
    // operand a MDString.
    if (const MDNode *md = dyn_cast<MDNode>(loopID->getOperand(i))) {
      if (!md || md->getNumOperands() == 0)
        continue;
      s = dyn_cast<MDString>(md->getOperand(0));
      for (unsigned i = 1, ie = md->getNumOperands(); i < ie; ++i)
        args.push_back(md->getOperand(i));
    } else {
      s = dyn_cast<MDString>(loopID->getOperand(i));
      assert(args.size() == 0 && "too many arguments for MDString");
    }

    if (!s)
      continue;

    // Check if the hint starts with the loop metadata prefix.
    StringRef name = s->getString();
    if (args.size() == 1)
      setHint(name, args[0]);
  }
}

bool llvm::TapirLoopHints::validate(StringRef name, unsigned v) {
  if (name == loopMDNameStrategy) {
    switch (TapirSpawnStrategy(v)) {
    case TapirSpawnStrategy::Basic:
    case TapirSpawnStrategy::DivideAndConquer:
    case TapirSpawnStrategy::GPU:
    case TapirSpawnStrategy::Sequential:
      return true;
    }
    return false;
  } else if (name == loopMDNameGrainSize) {
    return true;
  } else if (name == loopMDNameLoopTarget) {
    return createTTIDFrom(v).has_value();
  } else if (name == loopMDNameThreadsPerBlock) {
    return v <= KITSUNE_MAX_FIXED_THREADS_PER_BLOCK;
  } else if (name == loopMDNameAutotuneLaunch) {
    return true;
  } else if (name == loopMDNamePerfectDepth) {
    return true;
  } else if (name == loopMDNamePerfectLevel) {
    return true;
  } else {
    llvm_unreachable("TapirLoopHints::validate: Name not handled");
  }
}

bool llvm::TapirLoopHints::canCreateMetadata(StringRef name,
                                             const ValueType &v) const {
  if (name == loopMDNameLoopTarget)
    return getLoopTarget().has_value();
  return true;
}

unsigned llvm::TapirLoopHints::toMetadataValue(
    StringRef name, const llvm::TapirLoopHints::ValueType &v) const {
  assert(canCreateMetadata(name, v) && "Cannot get metadata value for hint");
  if (std::holds_alternative<bool>(v))
    return std::get<bool>(v);
  else if (std::holds_alternative<unsigned>(v))
    return std::get<unsigned>(v);
  else if (std::holds_alternative<TapirSpawnStrategy>(v))
    return unsigned(std::get<TapirSpawnStrategy>(v));
  else if (std::holds_alternative<std::optional<TTID>>(v))
    return unsigned(*std::get<std::optional<TTID>>(v));
  else
    llvm_unreachable("toMetadataValue: type not handled");
}

void llvm::TapirLoopHints::setHint(StringRef name, Metadata *arg) {
  if (!name.starts_with(loopMDNamePrefix))
    return;
  const auto *c = mdconst::dyn_extract<ConstantInt>(arg);
  if (!c)
    return;

  unsigned val = c->getZExtValue();
  if (not TapirLoopHints::validate(name, val))
    report_fatal_error(Twine("Invalid loop hint value: '") + name + "'");
  else if (name == loopMDNameStrategy)
    hints[name] = TapirSpawnStrategy(val);
  else if (name == loopMDNameGrainSize)
    hints[name] = val;
  else if (name == loopMDNameLoopTarget)
    hints[name] = TTID(val);
  else if (name == loopMDNameThreadsPerBlock)
    hints[name] = val;
  else if (name == loopMDNameAutotuneLaunch)
    hints[name] = bool(val);
  else if (name == loopMDNamePerfectDepth)
    hints[name] = val;
  else if (name == loopMDNamePerfectLevel)
    hints[name] = val;
  else
    llvm_unreachable("TapirLoopHints::setHint: Hint name not handled");
}

/// Create a new hint from name / value pair.
MDNode *llvm::TapirLoopHints::createHintMetadata(StringRef name,
                                                 unsigned v) const {
  LLVMContext &ctx = theLoop->getHeader()->getContext();
  Type *i32 = Type::getInt32Ty(ctx);
  Metadata *mds[] = {MDString::get(ctx, name),
                     ConstantAsMetadata::get(ConstantInt::get(i32, v))};
  return MDNode::get(ctx, mds);
}

/// Matches metadata with hint name.
bool llvm::TapirLoopHints::matchesHintMetadataName(MDNode *node,
                                                   const Hints &hints) const {
  auto *name = dyn_cast<MDString>(node->getOperand(0));
  if (!name)
    return false;

  // KITSUNE FIXME: Search for the full name.
  for (const auto &i : hints)
    if (name->getString().ends_with(i.first))
      return true;
  return false;
}

/// Sets current hints into loop metadata, keeping other values intact.
void llvm::TapirLoopHints::writeHintsToMetadata(const Hints &hints) {
  if (hints.size() == 0)
    return;

  LLVMContext &ctx = theLoop->getHeader()->getContext();
  SmallVector<Metadata *, 4> mds;

  // Reserve first location for self reference to the LoopID metadata node.
  TempMDTuple tempNode = MDNode::getTemporary(ctx, {});
  mds.push_back(tempNode.get());

  // If the loop already has metadata, then ignore the existing operands.
  if (MDNode *loopID = theLoop->getLoopID()) {
    for (unsigned i = 1, ie = loopID->getNumOperands(); i < ie; ++i) {
      auto *node = cast<MDNode>(loopID->getOperand(i));
      // If node in update list, ignore old value.
      if (!matchesHintMetadataName(node, hints))
        mds.push_back(node);
    }
  }

  // Now, add the missing hints.
  for (const auto &i : hints) {
    StringRef name = i.first;
    const ValueType &v = i.second;
    if (canCreateMetadata(name, v))
      mds.push_back(createHintMetadata(name, toMetadataValue(name, v)));
  }

  // Replace current metadata node with new one. The first operand should refer
  // to itself.
  MDNode *newLoopID = MDNode::get(ctx, mds);
  newLoopID->replaceOperandWith(0, newLoopID);

  theLoop->setLoopID(newLoopID);
}

/// Sets current hints into loop metadata, keeping other values intact.
void llvm::TapirLoopHints::writeHintsToClonedMetadata(const Hints &hints,
                                                      ValueToValueMapTy &vmap) {
  if (hints.size() == 0)
    return;

  LLVMContext &ctx = cast<BasicBlock>(vmap[theLoop->getHeader()])->getContext();
  SmallVector<Metadata *, 4> mds;

  // Reserve first location for self reference to the LoopID metadata node.
  TempMDTuple tempNode = MDNode::getTemporary(ctx, {});
  mds.push_back(tempNode.get());

  // If the loop already has metadata, then ignore the existing operands.
  MDNode *origLoopID = theLoop->getLoopID();
  if (!origLoopID)
    return;

  if (MDNode *loopID = dyn_cast_or_null<MDNode>(vmap.MD()[origLoopID])) {
    for (unsigned i = 1, ie = loopID->getNumOperands(); i < ie; ++i) {
      auto *node = cast<MDNode>(loopID->getOperand(i));
      // If node in update list, ignore old value.
      if (!matchesHintMetadataName(node, hints))
        mds.push_back(node);
    }
  }

  // Now, add the missing hints.
  for (const auto &i : hints) {
    StringRef name = i.first;
    const ValueType &v = i.second;
    if (canCreateMetadata(name, v))
      mds.push_back(createHintMetadata(name, toMetadataValue(name, v)));
  }

  // Replace current metadata node with new one.
  MDNode *newLoopID = MDNode::get(ctx, mds);
  // Set operand 0 to refer to the loop id itself.
  newLoopID->replaceOperandWith(0, newLoopID);

  // Set the metadata on the terminator of the cloned loop's latch.
  auto *clonedLatch = cast<BasicBlock>(vmap[theLoop->getLoopLatch()]);
  assert(clonedLatch && "Cloned Tapir loop does not have a single latch.");
  clonedLatch->getTerminator()->setMetadata(LLVMContext::MD_loop, newLoopID);
}

void llvm::TapirLoopHints::clearHintsMetadata() {
  LLVMContext &ctx = theLoop->getHeader()->getContext();
  SmallVector<Metadata *, 4> mds;

  // Reserve first location for self reference to the LoopID metadata node.
  TempMDTuple tempNode = MDNode::getTemporary(ctx, {});
  mds.push_back(tempNode.get());

  // If the loop already has metadata, then ignore the existing operands.
  if (MDNode *loopID = theLoop->getLoopID()) {
    for (unsigned i = 1, ie = loopID->getNumOperands(); i < ie; ++i) {
      auto *node = cast<MDNode>(loopID->getOperand(i));
      // If node in update list, ignore old value.
      if (!matchesHintMetadataName(node, hints))
        mds.push_back(node);
    }
  }

  // Replace current metadata node with new one.
  MDNode *newLoopID = MDNode::get(ctx, mds);
  // Set operand 0 to refer to the loop id itself.
  newLoopID->replaceOperandWith(0, newLoopID);

  theLoop->setLoopID(newLoopID);
}

void TapirLoopHints::clearStrategy() {
  hints[loopMDNameStrategy] = defaultTapirSpawnStrategy;
  writeHintsToMetadata({{loopMDNameStrategy, defaultTapirSpawnStrategy}});
}

void TapirLoopHints::clearClonedLoopMetadata(ValueToValueMapTy &vmap) {
  writeHintsToClonedMetadata({{loopMDNameStrategy, defaultTapirSpawnStrategy}},
                             vmap);
}

void TapirLoopHints::setAlreadyStripMined() {
  hints[loopMDNameGrainSize] = 1U;
  writeHintsToMetadata({{loopMDNameGrainSize, 1U}});
}

/// Returns true if Tapir-loop hints require loop outlining during lowering.
bool llvm::hintsDemandOutlining(const TapirLoopHints &hints) {
  switch (hints.getStrategy()) {
  case TapirSpawnStrategy::Basic:
  case TapirSpawnStrategy::DivideAndConquer:
  case TapirSpawnStrategy::GPU:
    return true;
  case TapirSpawnStrategy::Sequential:
    return false;
  }
  llvm_unreachable("hintsDemandOutlining: SpawningStrategy not handled");
}
