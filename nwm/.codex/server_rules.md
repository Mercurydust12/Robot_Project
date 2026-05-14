# Server Rules for This Workspace

You are working in a path-sensitive multi-environment setup involving:
- a host machine
- one or more Docker containers
- mounted local storage
- an NFS-mounted NAS path

Your job is to avoid path confusion, environment confusion, and unsafe assumptions.

---

## 1. Core Principle

Never assume path prefixes.

Before generating commands, editing configs, or suggesting file operations, always determine:

1. whether the current shell is on the host or inside Docker
2. which paths are actually visible in the current shell
3. which paths are real mounted storage versus ordinary container directories

Path errors in this workspace are often caused by mixing:
- host-visible paths
- container-visible paths
- overlay-backed container directories
- real mounted workspace or NAS paths

---

## 2. Verified Path Layout in the Current Container

The currently verified container shell uses the following real working paths:

- `/home/ial-zhangy/workspace` → real project workspace
- `/home/ial-zhangy/nas` → real NFS-mounted NAS path

Important facts:

- `/workspace` exists, but it is only an ordinary overlay-backed directory in the container root filesystem
- `/workspace` is **not** the user's actual mounted project workspace in the verified setup
- `/nas` does **not** exist in the verified container shell
- `/home/ial-zhangy/workspace` contains the actual project repos and working files
- `/home/ial-zhangy/nas` is the actual NAS path visible in the container

Therefore, in this environment:

- prefer `/home/ial-zhangy/workspace/...` for code, repos, scripts, outputs, and local workspace files
- prefer `/home/ial-zhangy/nas/...` for datasets, caches, checkpoints, and large files on NAS
- do **not** default to `/workspace/...`
- do **not** use `/nas/...`

---

## 3. Required Environment Detection

Before making any path-related suggestion, inspect the current shell with read-only commands such as:

```bash
pwd
whoami
hostname
uname -a
cat /etc/os-release
test -f /.dockerenv && echo inside_docker || echo not_obviously_docker
cat /proc/1/cgroup 2>/dev/null | head -n 20
ls /
```

You must summarize:
- current user
- current hostname
- whether this appears to be host or container
- which important root directories are visible

Do not skip this step when context is unclear.

---

## 4. Required Path Inspection

Before generating commands involving training, cache generation, rsync, symlinks, config editing, or dataset paths, inspect the real storage layout.

Use read-only checks such as:

```bash
ls -la /workspace 2>/dev/null
ls -la /home/ial-zhangy/workspace 2>/dev/null
ls -la /home/ial-zhangy/nas 2>/dev/null
ls -ld /nas /workspace 2>/dev/null
df -h /workspace /home/ial-zhangy/workspace /home/ial-zhangy/nas 2>/dev/null
mount | grep -E 'workspace|nas|home'
readlink -f /home/ial-zhangy/workspace 2>/dev/null
readlink -f /home/ial-zhangy/nas 2>/dev/null
```

Interpret carefully:
- an existing directory is not necessarily a mounted workspace
- an overlay-backed root-level directory may be empty and not suitable for real work
- the real project path must be identified from content plus mount information

---

## 5. Path Rules for All Future Commands

### Stable defaults in the verified container shell

Use these absolute paths as the default roots:

- `/home/ial-zhangy/workspace` for code, repos, scripts, outputs, and local workspace files
- `/home/ial-zhangy/nas` for datasets, caches, checkpoints, and large NAS-backed files

### Acceptable interactive shell shorthand

In interactive shell usage, the following shorthand is acceptable:

- `~/workspace` → `/home/ial-zhangy/workspace`
- `~/nas` → `/home/ial-zhangy/nas`

### Avoid ambiguous relative paths

Do not use bare relative paths such as:

- `workspace`
- `nas`

unless the current working directory has been explicitly verified and the relative meaning is intentional.

### Do not use these as defaults

Do not default to:

- `/workspace`
- `/nas`

because in the verified environment:
- `/workspace` exists but is not the real mounted project workspace
- `/nas` does not exist

---

## 6. Error Interpretation Rules

If a command fails with messages such as:
- `No such file or directory`
- `Permission denied`
- unexpected nesting or duplicated path confusion

first suspect path-prefix mismatch, not broken data.

Specifically check for confusion between:
- `/workspace/...` vs `/home/ial-zhangy/workspace/...`
- `/nas/...` vs `/home/ial-zhangy/nas/...`
- host path assumptions vs container path assumptions

Do not immediately assume that files are missing.

---

## 7. Safe Command Generation Policy

Before suggesting any command, explicitly state:

1. whether the command is intended for host or container
2. which path root it assumes
3. whether it is read-only or modifying data

Never silently mix host-style and container-style paths in the same command.

For example, do not generate commands that combine:
- a source path under `/home/ial-zhangy/nas/...`
- with a destination path under `/nas/...`

unless that mapping has been explicitly verified in the current shell.

---

## 8. Rules for Training / Cache / Dataset Workflows

For commands involving:
- training
- rollout cache generation
- evaluation
- TensorBoard log directories
- dataset manifests
- rsync or data migration
- symbolic links
- config path editing

always do the following first:

1. confirm current environment
2. confirm current path roots
3. confirm whether the project repo is under `/home/ial-zhangy/workspace/...`
4. confirm whether the dataset or cache target is under `/home/ial-zhangy/nas/...`

Preferred default assumptions in the verified container:

- repo root example:
```text
/home/ial-zhangy/workspace/Nav_DP_8M
```

- NAS dataset root example:
```text
/home/ial-zhangy/nas/...
```

---

## 9. Docker Awareness Rules

This environment uses Docker with GPU support.

Important rule:
do not assume that container-visible paths match host-visible paths.

Even if the host may expose some storage differently, commands generated for the current container must use the paths that are actually visible inside the container.

If Docker context is relevant, inspect with read-only commands such as:

```bash
docker ps -a 2>/dev/null
docker inspect $(hostname) 2>/dev/null || true
mount | head -n 100
df -h
```

If Docker commands are unavailable inside the current shell, say so clearly instead of guessing.

---

## 10. Python / Conda Rules

Do not assume Python, Conda, or CUDA paths are identical across host and container.

Before generating Python execution commands, inspect:

```bash
which python || true
which python3 || true
python --version 2>/dev/null || true
python3 --version 2>/dev/null || true
echo $CONDA_PREFIX
echo $PATH
```

If training depends on PyTorch or CUDA, also inspect:

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('torch_cuda', torch.version.cuda); print('device_count', torch.cuda.device_count())" 2>/dev/null || true
```

Report what is verified. Do not invent environment details.

---

## 11. CUDA / GPU Rules

If GPU issues are involved, never assume that:
- driver CUDA version
- runtime CUDA version
- compiler `nvcc` version
- PyTorch CUDA version

are all the same.

Inspect explicitly with read-only commands such as:

```bash
nvidia-smi
nvidia-smi -L
which nvcc || true
nvcc --version 2>/dev/null || true
ls -l /usr/local/cuda /usr/local/cuda/bin/nvcc 2>/dev/null || true
echo $CUDA_HOME $CUDA_PATH
```

Always report version mismatches clearly if they exist.

---

## 12. Minimal Mandatory Checks Before Path-Sensitive Work

If the user asks for help with a path-sensitive task and the environment is not fully obvious, run or request the following first:

```bash
pwd
ls /
ls -la /workspace 2>/dev/null
ls -la /home/ial-zhangy/workspace 2>/dev/null
ls -la /home/ial-zhangy/nas 2>/dev/null
df -h /workspace /home/ial-zhangy/workspace /home/ial-zhangy/nas 2>/dev/null
mount | grep -E 'workspace|nas|home'
```

Then decide paths only after reading the results.

---

## 13. Preferred Response Style

When helping with server operations:
- be concrete
- use absolute paths
- state assumptions explicitly
- distinguish verified facts from guesses
- prefer short factual explanations over long speculation

Good:
- “This command assumes the current shell is the verified container and uses `/home/ial-zhangy/nas/...` as the NAS root.”

Bad:
- “Just use the NAS path.”
- “Try `/workspace`.”
- “It should probably be there.”

---

## 14. Final Default Mental Model

Use this sequence every time:

1. detect host vs container
2. inspect visible roots
3. identify real mounted workspace
4. identify real NAS mount
5. confirm Python / CUDA if relevant
6. only then generate commands

For the currently verified container shell, the default working model is:

- code path root:
```text
/home/ial-zhangy/workspace
```

- NAS path root:
```text
/home/ial-zhangy/nas
```

Do not replace them with `/workspace` or `/nas` unless new evidence proves otherwise.
