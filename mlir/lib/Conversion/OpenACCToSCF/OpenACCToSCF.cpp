//===- OpenACCToSCF.cpp - conversion from OpenACC to SCF dialect ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/OpenACCToSCF/OpenACCToSCF.h"

#include "../PassDetail.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

/// Conversion patterns.
namespace {
class LoopOpConversion : public OpConversionPattern<acc::LoopOp> {
public:
  using OpConversionPattern<acc::LoopOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(acc::LoopOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

class ParallelOpConversion : public OpConversionPattern<acc::ParallelOp> {
public:
  using OpConversionPattern<acc::ParallelOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(acc::ParallelOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

namespace {

class ConvertOpenACCToSCFPass
    : public ConvertOpenACCToSCFBase<ConvertOpenACCToSCFPass> {
  void runOnOperation() override {
    OwningRewritePatternList patterns;
    populateOpenACCToSCFConversionPatterns(patterns, &getContext());
    ConversionTarget target(getContext());
    target
        .addLegalDialect<scf::SCFDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

// Converts acc::parallel { acc::loop { scf::for { body } } } to scf::parallel { body }
LogicalResult
ParallelOpConversion::matchAndRewrite(acc::ParallelOp op, ArrayRef<Value> operands,
                                 ConversionPatternRewriter &rewriter) const {
  // We only continue if first op in the loop is an scf::for and the first op in
  // a parallel is a loop. If not, the transform is a noop
  auto loop = dyn_cast<acc::LoopOp>(op.region().begin()->begin());
  if(!loop) return success();
  auto fop = dyn_cast<scf::ForOp>(loop.region().begin()->begin());
  if(!fop) return success(); 

  SmallVector<Value, 8> steps = {fop.step()} ; 
  SmallVector<Value, 8> ivs = {fop.getInductionVar()};  
  SmallVector<Value, 8> upperBoundTuple = {fop.upperBound()};
  SmallVector<Value, 8> lowerBoundTuple = {fop.lowerBound()};
  
  if(auto collapse = loop.collapse()){ 
    for(uint64_t i=0; i< *collapse - 1; i++){
      fop = dyn_cast<scf::ForOp>(fop.region().begin()->begin());
      if(!fop) return failure();
      steps.push_back(fop.step());
      upperBoundTuple.push_back(fop.upperBound());
      lowerBoundTuple.push_back(fop.lowerBound()); 
      ivs.push_back(fop.getInductionVar()); 
    }
  }

  // The fact that we construct an op that contains regions by creating it and
  // then mutating the region insode is grotesque, but seems to be the MLIR
  // Way™.
  scf::ParallelOp par = rewriter.create<scf::ParallelOp>(
    op.getLoc(), lowerBoundTuple, upperBoundTuple, steps); 

  BlockAndValueMapping map;
  for(auto dim : llvm::zip(ivs, par.getInductionVars())){
    Value iv, newiv;
    std::tie(iv, newiv) = dim; 
    map.map(iv, newiv);
  }

  rewriter.eraseBlock(&(*par.region().begin())); 
  rewriter.cloneRegionBefore(fop.region(), par.region(),
                             par.region().begin(), map);

  rewriter.replaceOp(op, par.results());

  return success(); 
}

// Not clear what a standalone loop means
LogicalResult
LoopOpConversion::matchAndRewrite(acc::LoopOp loop, ArrayRef<Value> operands,
                                 ConversionPatternRewriter &rewriter) const {

  return success();
}

void mlir::populateOpenACCToSCFConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ParallelOpConversion>(ctx);
}

std::unique_ptr<Pass>
mlir::createConvertOpenACCToSCFPass() {
  return std::make_unique<ConvertOpenACCToSCFPass>();
}
