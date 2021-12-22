//===- OpenACCToSCF.cpp - conversion from OpenACC to SCF dialect ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/OpenACCToSCF/OpenACCToSCF.h"

#include "../PassDetail.h"
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

LogicalResult
LoopOpConversion::matchAndRewrite(acc::LoopOp loop, ArrayRef<Value> operands,
                                 ConversionPatternRewriter &rewriter) const {
  
  // TODO: handle nested case

  // We assert that the first op in the loop is an scf::for
  auto fop = dyn_cast<scf::ForOp>(loop.region().begin()->begin());
  if(!fop)
    return failure(); 

  SmallVector<Value, 8> steps = {fop.step()} ; 
  SmallVector<Value, 8> upperBoundTuple = {fop.upperBound()};
  SmallVector<Value, 8> lowerBoundTuple = {fop.lowerBound()};

  // The fact that we construct an op that contains regions by creating it and
  // then mutating the region insode is grotesque, but seems to be the MLIR
  // Way™.
  scf::ParallelOp par = rewriter.create<scf::ParallelOp>(
    fop.getLoc(), lowerBoundTuple, upperBoundTuple, steps); 
 
  rewriter.eraseBlock(par.getBody());
  rewriter.inlineRegionBefore(fop.region(), par.region(),
                              par.region().end());
  rewriter.replaceOp(loop, par.results());

  // I believe we have to delete these ops explicitly, as they are not removed
  // by the removal of the surrounding acc::Loop op. I hate it.
  rewriter.eraseOp(loop.getBody()->getTerminator()); 
  rewriter.eraseOp(fop); 
  return success();

}

void mlir::populateOpenACCToSCFConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<LoopOpConversion>(ctx);
}

std::unique_ptr<Pass>
mlir::createConvertOpenACCToSCFPass() {
  return std::make_unique<ConvertOpenACCToSCFPass>();
}
