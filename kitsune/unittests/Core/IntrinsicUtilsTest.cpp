//==- IntrinsicUtilsTest.cpp - Unit tests for Kitsune intrinsic utilities --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/IntrinsicUtils.h"
#include "kitsune/Core/Tapir.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(KitIntrinsicUtils, isKitIntrinsic) {
  // This is not a comprehensive list, it just attempts to ensure some sanity.
  EXPECT_TRUE(isKitIntrinsic(Intrinsic::kit_async_cpu_threads_launch));
  EXPECT_TRUE(isKitIntrinsic(Intrinsic::kit_async_gpu_kernel_launch));
  EXPECT_TRUE(isKitIntrinsic(Intrinsic::kit_async_gpu_prefetch_dtoh));
  EXPECT_TRUE(isKitIntrinsic(Intrinsic::kit_async_gpu_prefetch_htod));
  EXPECT_TRUE(isKitIntrinsic(Intrinsic::kit_cpu_threads_launch));
  EXPECT_TRUE(isKitIntrinsic(Intrinsic::kit_cpu_threads_sync));
  EXPECT_TRUE(isKitIntrinsic(Intrinsic::kit_gpu_memcpy_dtoh));
  EXPECT_TRUE(isKitIntrinsic(Intrinsic::kit_gpu_memcpy_htod));
  EXPECT_TRUE(isKitIntrinsic(Intrinsic::kit_gpu_stream_sync));

  EXPECT_FALSE(isKitIntrinsic(Intrinsic::prefetch));
  EXPECT_FALSE(isKitIntrinsic(Intrinsic::memcpy));
  EXPECT_FALSE(isKitIntrinsic(Intrinsic::memset));
}

TEST(KitIntrinsicUtils, isKitIntrinsicAsync) {
  // This ought to be a comprehensive list.
  EXPECT_TRUE(isKitIntrinsicAsync(Intrinsic::kit_async_gpu_kernel_launch));
  EXPECT_TRUE(isKitIntrinsicAsync(Intrinsic::kit_async_cpu_threads_launch));
  EXPECT_TRUE(isKitIntrinsicAsync(Intrinsic::kit_async_gpu_memcpy_dtoh));
  EXPECT_TRUE(isKitIntrinsicAsync(Intrinsic::kit_async_gpu_memcpy_htod));
  EXPECT_TRUE(isKitIntrinsicAsync(Intrinsic::kit_async_gpu_prefetch_dtoh));
  EXPECT_TRUE(isKitIntrinsicAsync(Intrinsic::kit_async_gpu_prefetch_htod));

  EXPECT_FALSE(isKitIntrinsicAsync(Intrinsic::kit_gpu_stream_sync));
  EXPECT_FALSE(isKitIntrinsicAsync(Intrinsic::kit_cpu_threads_sync));
  EXPECT_FALSE(isKitIntrinsicAsync(Intrinsic::kit_gpu_memcpy_dtoh));
  EXPECT_FALSE(isKitIntrinsicAsync(Intrinsic::kit_gpu_memcpy_htod));
  EXPECT_FALSE(isKitIntrinsicAsync(Intrinsic::kit_runtime_finalize));
  EXPECT_FALSE(isKitIntrinsicAsync(Intrinsic::kit_runtime_initialize));
}

TEST(KitIntrinsicUtils, isKitIntrinsicBlocking) {
  // Most Kitsune intrinsics are blocking. This is mainly here to ensure that
  // this function does not report async calls as blocking. At least the async
  // calls here ought to be comprehensive.
  EXPECT_FALSE(isKitIntrinsicBlocking(Intrinsic::kit_async_gpu_kernel_launch));
  EXPECT_FALSE(isKitIntrinsicBlocking(Intrinsic::kit_async_cpu_threads_launch));
  EXPECT_FALSE(isKitIntrinsicBlocking(Intrinsic::kit_async_gpu_memcpy_dtoh));
  EXPECT_FALSE(isKitIntrinsicBlocking(Intrinsic::kit_async_gpu_memcpy_htod));
  EXPECT_FALSE(isKitIntrinsicBlocking(Intrinsic::kit_async_gpu_prefetch_dtoh));
  EXPECT_FALSE(isKitIntrinsicBlocking(Intrinsic::kit_async_gpu_prefetch_htod));

  EXPECT_TRUE(isKitIntrinsicBlocking(Intrinsic::kit_gpu_stream_sync));
  EXPECT_TRUE(isKitIntrinsicBlocking(Intrinsic::kit_cpu_threads_launch));
  EXPECT_TRUE(isKitIntrinsicBlocking(Intrinsic::kit_cpu_threads_sync));
  EXPECT_TRUE(isKitIntrinsicBlocking(Intrinsic::kit_gpu_memcpy_dtoh));
  EXPECT_TRUE(isKitIntrinsicBlocking(Intrinsic::kit_gpu_memcpy_htod));
  EXPECT_TRUE(isKitIntrinsicBlocking(Intrinsic::kit_runtime_finalize));
  EXPECT_TRUE(isKitIntrinsicBlocking(Intrinsic::kit_runtime_initialize));
}

TEST(KitIntrinsicUtils, isKitIntrinsicCPU) {
  // This is not a comprehensive list. It's mainly here to ensure some measure
  // of sanity.
  EXPECT_TRUE(isKitIntrinsicCPU(Intrinsic::kit_async_cpu_threads_launch));
  EXPECT_TRUE(isKitIntrinsicCPU(Intrinsic::kit_cpu_threads_launch));

  EXPECT_FALSE(isKitIntrinsicCPU(Intrinsic::kit_async_gpu_kernel_launch));
  EXPECT_FALSE(isKitIntrinsicCPU(Intrinsic::kit_gpu_thread_id_x));
  EXPECT_FALSE(isKitIntrinsicCPU(Intrinsic::kit_mobile_alloc));
  EXPECT_FALSE(isKitIntrinsicCPU(Intrinsic::kit_reduce_0));
  EXPECT_FALSE(isKitIntrinsicCPU(Intrinsic::kit_runtime_initialize));
}

TEST(KitIntrinsicUtils, isKitIntrinsicGPU) {
  // This is not a comprehensive list. It's mainly here to ensure some measure
  // of sanity.
  EXPECT_TRUE(isKitIntrinsicGPU(Intrinsic::kit_async_gpu_kernel_launch));
  EXPECT_TRUE(isKitIntrinsicGPU(Intrinsic::kit_gpu_thread_id_x));

  EXPECT_FALSE(isKitIntrinsicGPU(Intrinsic::kit_async_cpu_threads_launch));
  EXPECT_FALSE(isKitIntrinsicGPU(Intrinsic::kit_cpu_threads_launch));
  EXPECT_FALSE(isKitIntrinsicGPU(Intrinsic::kit_mobile_alloc));
  EXPECT_FALSE(isKitIntrinsicGPU(Intrinsic::kit_reduce_0));
  EXPECT_FALSE(isKitIntrinsicGPU(Intrinsic::kit_runtime_initialize));
}

TEST(KitIntrinsicUtils, getStreamFromLaunch) {
  LLVMContext ctx;
  Module m("", ctx);
  Function *f = Intrinsic::getOrInsertDeclaration(
      &m, Intrinsic::kit_async_gpu_kernel_launch);

  FunctionType *fty = f->getFunctionType();
  PointerType *ptr = PointerType::getUnqual(ctx);
  Type *i32 = Type::getInt32Ty(ctx);
  Type *i64 = Type::getInt64Ty(ctx);
  Type *f32 = Type::getFloatTy(ctx);

  Constant *g = m.getOrInsertGlobal("g", ptr);
  Constant *cnull = ConstantPointerNull::get(ptr);
  Constant *c0_4 = ConstantInt::get(i32, 0);
  Constant *c0_8 = ConstantInt::get(i64, 0);
  Constant *ctt = ConstantInt::get(i32, unsigned(TTID::Cuda));
  Constant *cfp = ConstantFP::get(f32, 3.14159);

  CallInst *call0 = CallInst::Create(
      fty, f, {ctt, cnull, cnull, c0_8, c0_8, c0_8, c0_4, cnull, g});
  CallInst *call1 = CallInst::Create(
      fty, f, {ctt, cnull, cnull, c0_8, c0_8, c0_8, c0_4, cnull, g, cfp});

  EXPECT_EQ(getStreamFromLaunch(*call0), g);
  EXPECT_EQ(getStreamFromLaunch(*call1), g);

  call0->deleteValue();
  call1->deleteValue();
}

TEST(KitIntrinsicUtils, getTTIDFromKitIntrCall) {
  LLVMContext ctx;
  Module m("", ctx);
  Type *i32 = Type::getInt32Ty(ctx);

  Constant *c0 = ConstantInt::get(i32, 0);
  Constant *c1 = ConstantInt::get(i32, unsigned(TTID::Serial));
  Constant *c_1 = ConstantInt::get(i32, -1, /*isSigned=*/true);

  Function *min = Intrinsic::getOrInsertDeclaration(&m, Intrinsic::umin, {i32});
  Function *init =
      Intrinsic::getOrInsertDeclaration(&m, Intrinsic::kit_runtime_initialize);
  Function *fin =
      Intrinsic::getOrInsertDeclaration(&m, Intrinsic::kit_runtime_finalize);

  CallInst *callMin = CallInst::Create(min->getFunctionType(), min, {c0, c1});
  CallInst *callInit = CallInst::Create(init->getFunctionType(), init, {c_1});
  CallInst *callFin = CallInst::Create(fin->getFunctionType(), fin, {c1});

  EXPECT_FALSE(getTTIDFromKitIntrCall(*callMin).has_value());
  EXPECT_FALSE(getTTIDFromKitIntrCall(*callInit).has_value());
  EXPECT_EQ(getTTIDFromKitIntrCall(*callFin), TTID::Serial);

  callMin->deleteValue();
  callInit->deleteValue();
  callFin->deleteValue();
}

} // namespace
