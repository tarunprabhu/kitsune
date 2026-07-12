import subprocess

# Running `amdgpu-arch` can be expensive on certain systems, especially if an
# AMD GPU is not present on it. Therefore, we perform this check only when we
# descend into this directory, and only if we know that the `hip` tapir target
# has been built.
def has_gpu():
    cmd = os.path.join(config.llvm_tools_dir, "amdgpu-arch")
    return len(subprocess.check_output([cmd])) != 0

if "kitsune-hip" not in config.available_features:
    config.unsupported = True
elif has_gpu():
    config.available_features.add("amd-gpu")
