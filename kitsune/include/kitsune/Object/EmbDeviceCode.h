//==- EmbDeviceCode.h - Wrapper object for embedded device code -*- C++ -*--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Wrapper object for embedded device code. These are generally constructed
// by the embedded device code parser.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_OBJECT_EMB_DEVICE_CODE_H
#define KITSUNE_OBJECT_EMB_DEVICE_CODE_H

#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <optional>

namespace llvm {

class MemoryBuffer;

namespace object {
class Binary;
} // namespace object

/// An embedded device code object. This consists of a reference to the code
/// buffer with metadata about the code itself.
struct EmbDeviceCode {
public:
  /// The identifier for the embedded device code. We intentionally do not reuse
  /// any constants from elsewhere in the code. This does mean that this list
  /// needs to kept updated as new devices are added, but hopefully that is not
  /// too much of a maintenance burden.
  ///
  /// The values for each element of this enumeration follow a specific format.
  ///
  ///   .--------------------------------------------.
  ///   |     UNUSED     |  FORMAT  |   MACHINE ID   |
  ///   `--------------------------------------------'
  ///   64              32         16                0
  ///
  /// The architecture is a 16-bit value that identifies a specific device.
  /// The format currently must be one of the following:
  ///
  ///     01 - AMDGPU
  ///     02 - NVIDIA SASS
  ///     03 - NVIDIA PTX
  ///
  /// The \ref Format enum is used to represent one of these values.
  ///
  /// NOTE: Neither <ARCH> nor <FORMAT> above should be 0 i.e. all valid
  /// machine id's and format's must be non-zero. 0 values for the machine id
  /// are only used for sentinels.
  ///
  enum Id : uint64_t {
    /// AMDGPU machine ids
    /// @{
    /// Sentinel one before the first valid AMDGPU id
    AMDGPU_lo = 0x10000,

    // gfx < 9
    GFX600 = 0x10001,
    GFX601 = 0x10002,
    GFX602 = 0x10003,
    GFX700 = 0x10004,
    GFX701 = 0x10005,
    GFX702 = 0x10006,
    GFX703 = 0x10007,
    GFX704 = 0x10008,
    GFX705 = 0x10009,
    GFX801 = 0x1000A,
    GFX802 = 0x1000B,
    GFX803 = 0x1000C,
    GFX805 = 0x1000D,
    GFX810 = 0x1000E,
    // gfx 9
    GFX900 = 0x1000F,
    GFX902 = 0x10010,
    GFX904 = 0x10011,
    GFX906 = 0x10012,
    GFX908 = 0x10013,
    GFX90A = 0x10014,
    GFX90C = 0x10015,
    // gfx 9.4
    GFX940 = 0x10016,
    GFX941 = 0x10017,
    GFX942 = 0x10018,
    GFX950 = 0x10019,
    // gfx 10.1
    GFX1010 = 0x1001A,
    GFX1011 = 0x1001B,
    GFX1012 = 0x1001C,
    GFX1013 = 0x1001D,
    // gfx 10.3
    GFX1030 = 0x1001E,
    GFX1031 = 0x1001F,
    GFX1032 = 0x10020,
    GFX1033 = 0x10021,
    GFX1034 = 0x10022,
    GFX1035 = 0x10023,
    GFX1036 = 0x10024,
    // gfx 11
    GFX1100 = 0x10025,
    GFX1101 = 0x10026,
    GFX1102 = 0x10027,
    GFX1103 = 0x10028,
    GFX1150 = 0x10029,
    GFX1151 = 0x1002A,
    GFX1152 = 0x1002B,
    GFX1153 = 0x1002C,
    // gfx 12
    GFX1200 = 0x1002D,
    GFX1201 = 0x1002E,

    /// Sentinel one past the last valid AMDGPU id
    AMDGPU_hi = 0x1002F,

    /// @}

    /// NVIDIA SASS ID's. These correspond to a cuda architectures of the form
    /// sm_NNNN.
    ///
    /// @{
    /// Sentinel one before the first valid NVSASS id
    NVSASS_lo = 0x20000,

    SM_30 = 0x20001,
    SM_32 = 0x20002,
    SM_35 = 0x20003,
    SM_37 = 0x20004,
    SM_50 = 0x20005,
    SM_52 = 0x20006,
    SM_53 = 0x20007,
    SM_60 = 0x20008,
    SM_61 = 0x20009,
    SM_62 = 0x2000A,
    SM_70 = 0x2000B,
    SM_72 = 0x2000C,
    SM_75 = 0x2000D,
    SM_80 = 0x2000E,
    SM_86 = 0x2000F,
    SM_87 = 0x20010,
    SM_89 = 0x20011,
    SM_90 = 0x20012,
    SM_90A = 0x20013,
    SM_100 = 0x20014,
    SM_100A = 0x20015,
    SM_101 = 0x20016,
    SM_101A = 0x20017,
    SM_120 = 0x20018,
    SM_120A = 0x20019,

    /// Sentinel one past the last valid NVSASS id
    NVSASS_hi = 0x2001A,
    /// @}

    /// NVIDIA PTX ID's. These correspond to a virtual architectures of the form
    /// compute_NNNN.
    ///
    /// @{

    /// Sentinel one before the first valid NVPTX id
    NVPTX_lo = 0x30000,

    COMPUTE_30 = 0x30001,
    COMPUTE_32 = 0x30002,
    COMPUTE_35 = 0x30003,
    COMPUTE_37 = 0x30004,
    COMPUTE_50 = 0x30005,
    COMPUTE_52 = 0x30006,
    COMPUTE_53 = 0x30007,
    COMPUTE_60 = 0x30008,
    COMPUTE_61 = 0x30009,
    COMPUTE_62 = 0x3000A,
    COMPUTE_70 = 0x3000B,
    COMPUTE_72 = 0x3000C,
    COMPUTE_75 = 0x3000D,
    COMPUTE_80 = 0x3000E,
    COMPUTE_86 = 0x3000F,
    COMPUTE_87 = 0x30010,
    COMPUTE_89 = 0x30011,
    COMPUTE_90 = 0x30012,
    COMPUTE_90A = 0x30013,
    COMPUTE_100 = 0x30014,
    COMPUTE_100A = 0x30015,
    COMPUTE_101 = 0x30016,
    COMPUTE_101A = 0x30017,
    COMPUTE_120 = 0x30018,
    COMPUTE_120A = 0x30019,

    /// Sentinel one past the last valid NVPTX id
    NVPTX_hi = 0x3001A,

    /// @}
  };

  /// The format of the device code buffer i.e. what the binary data in the
  /// buffer represents. The actual values of the elements of this enum must be
  /// exactly what would be encoded to form the EmbDeviceCode::Id. None of the
  /// elements of this enum must be 0.
  enum BinaryFormat : uint64_t {
    AMDGPU = 0x10000, /// AMD GPU machine code
    NVSASS = 0x20000, /// NVIDIA machine code
    NVPTX = 0x30000,  /// NVIDIA PTX code
  };

private:
  /// The binary file containing into which this device code is embedded.
  const object::Binary &bin;

  /// The identifier describing the format of the device code.
  Id id;

  /// The code buffer.
  StringRef code;

  /// The name of the code buffer. This is derived from the name of the parent
  /// file into which this was embedded, the binary format, the target
  /// architecture and the magic number of the code buffer.
  std::string name;

  /// If \ref bin is a static archive, the embedded device code will be a static
  /// archive obtained by linking the device code found in all members of the
  /// source archive. In this case, the generated archive is owned by this
  /// object, unlike the source binary which is owned by the caller. Device code
  /// from more than one tapir target may be present, so there may be multiple
  /// combined archives.
  // std::unique_ptr<MemoryBuffer> archive;

private:
  EmbDeviceCode(const object::Binary &bin, Id id, StringRef code,
                StringRef hostFileName);

public:
  /// The tapir target that generated this embedded device code.
  TTID getTTID() const;

  /// Get the name of the code buffer. This is derived from the name of the
  /// parent file into which this was embedded, the binary format, the target
  /// and the magic number of the code buffer.
  StringRef getName() const { return name; }

  /// Get the code.
  StringRef getCode() const { return code; }

  /// The raw machine ID.
  Id getId() const { return id; }

  /// The device code format.
  BinaryFormat getBinaryFormat() const {
    return static_cast<BinaryFormat>(uint64_t(id) & uint64_t(maskFormat));
  }

  /// The architecture as a string.
  StringRef getArch() const;

  /// Is this embedded device code a static archive.
  bool isArchive() const;

  /// Is this embedded device code a relocatable object file. Dynamic shared
  /// objects (DSO's) are not considered objects by this method. Use
  /// \ref isShared to check for that.
  bool isObject() const;

  /// Is this embedded device code a shared library.
  bool isShared() const;

private:
  /// The mask used to extract the machine id from the \ref Id.
  static constexpr uint64_t maskMachine = 0xffff;

  /// The mask used to extract the format from the \ref Id.
  static constexpr uint64_t maskFormat = 0xffff0000;

  /// The mask used to extract the unused bits form the \ref Id. This is only
  /// needed to check that the id is valid.
  static constexpr uint64_t maskUnused = 0xffffffff00000000;

public:
  /// Construct an embedded device code object generated by the given tapir
  /// target contained in a binary file if it exists.
  static Expected<std::optional<EmbDeviceCode>>
  create(const object::Binary &bin, TTID tt);

  /// Convert the string to an embedded device code id.
  static Expected<EmbDeviceCode::Id> getIdFor(StringRef s);

  /// Convert the integer to an embedded device code id.
  static Expected<EmbDeviceCode::Id> getIdFor(uint64_t n);
};

} // namespace llvm

#endif // KITSUNE_OBJECT_EMB_DEVICE_CODE_H
