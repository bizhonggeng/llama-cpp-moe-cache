# 编译移交文档：llama.cpp → Windows / CUDA 12 / AVX2

> **目标机**：Windows，4 GB VRAM + 32 GB RAM（OS 约占 10 GB，模型可用约 22 GB）
> **模型**：Qwen3.6-35B-A3B Q4_K_M，4K 上下文
> **基线**：llama.cpp 主线，tg 18 t/s
> **开发机**：Ubuntu 24.04 x86_64
> **构建方式**：GitHub Actions `windows-2022` runner（**不做交叉编译**）
> 编制日期：2026-10-04

---

## 0. 流程总览

```
Ubuntu 24.04 开发机                 GitHub                    Windows 目标机
─────────────────────            ────────                  ─────────────────
1. clone llama.cpp          →    2. push 到 GitHub     →    5. 下载 artifact
2. 合入 PR #27861                 3. 触发 workflow           6. 预检（量真实体积/dims）
3. 应用小显存适配补丁               4. 编译 ~20-40 min         7. A/B 测速
4. push                                                    8. 回传结果
```

**关键约束：不能交叉编译。** Ubuntu 上装 mingw 无法产出可用的 CUDA Windows 二进制——CUDA 的 Windows 工具链依赖 MSVC 的 `cl.exe` 与 `.lib` 导入库。所以构建必须发生在 Windows runner 上。

**runner 不需要 GPU。** `nvcc` 编译只需要 toolkit，不需要设备。测速回到目标机做。

---

## 1. 收益预期

| 项 | 结论 |
|---|---|
| 内存 | 模型 18–20 GB，可用 22 GB → **全部装得下，余量 0.7–2.7 GB** |
| `--no-mmap` | **开启**（22 GB 下可以，消除缺页抖动） |
| 专家缓存预算 | 显存可腾约 2.4 GB → 36 槽位 → **约 +55%** |
| **预期落点** | **18 t/s → 约 27–28 t/s** |

不是 +84%（那是 RTX 4090 24 GB 的数字）。4 GB 卡的瓶颈是每槽位 67.5 MiB，40 层各存一份。

---

## 2. Ubuntu 开发机：准备源码（第 1 步）

```bash
# 2.1 工具
sudo apt update && sudo apt install -y git cmake python3 python3-pip

# 2.2 取源码
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# 2.3 合入 MoE 专家缓存 PR #27861
git fetch origin pull/27861/head:pr-27861
git merge pr-27861
#  若冲突：git merge --abort → 把冲突文件清单发回来，不要自行解决

# 2.4 应用小显存适配补丁（幂等）
python3 ../moe-cache/apply_budget_patch.py --check    # 先干跑
python3 ../moe-cache/apply_budget_patch.py
#  期待 6 行 PATCHED、0 行 FAILED
#  FAILED: anchor text not found = 上一步 PR 没合上，回 2.3

# 2.5 确认改上了
git status
git diff --stat
#  应看到 src/llama-moecache.cpp、src/llama-moecache.h 新增，
#  以及 common/arg.cpp、include/llama.h 等 6 处修改
```

### 2.6 本地可以直接验证模型 dims（Ubuntu 上有模型文件的话）

```bash
pip install -r gguf-py/requirements.txt
python3 gguf-py/scripts/gguf_dump.py /path/to/Qwen3.6-35B-A3B-Q4_K_M.gguf > dims.txt
```

---

## 3. 推送并触发构建（第 2–4 步）

### 3.1 建仓库

在 GitHub 上新建一个**空仓库**（`llama-cpp-moe-cache`），然后：

```bash
cd llama.cpp
git remote set-url origin git@github.com:<你的账号>/llama-cpp-moe-cache.git
git checkout -b build-win-cuda12
git push -u origin build-win-cuda12
```

> **公开仓库**：Actions 分钟数不限。
> **私有仓库**：免费账户每月 2000 分钟，单次构建约 20–40 min，够用但别反复触发。

### 3.2 放工作流文件

把 `cicd/build-win-cuda12.yml` 放到仓库的 `.github/workflows/` 下：

```bash
mkdir -p .github/workflows
cp ../cicd/build-win-cuda12.yml .github/workflows/
git add .github/workflows/build-win-cuda12.yml
git commit -m "ci: build Windows CUDA 12 AVX2 with MoE expert cache"
git push
```

push 即触发（workflow 里配了 `push: branches: [build-win-cuda12]`）。
也可在 Actions 页手动 `Run workflow`，此时可临时改两个输入：

| 输入 | 默认 | 说明 |
|---|---|---|
| `cuda_arch` | `86` | 目标 GPU 的 sm 号 |
| `avx512` | `true` | 目标 CPU 支持 AVX-512 选 `true`，只有 AVX2 选 `false` |

### 3.3 工作流做了什么（要点）

| 步骤 | 说明 |
|---|---|
| `Jimver/cuda-toolkit@v0.2.30` | **windows runner 没有预装 CUDA**，必须用 action 装。method 用 `network` 只拉需要的子包，比 `local`（约 3 GB 全量）快很多。**锁 12.4.1，符合"只到 CUDA 12"的约束** |
| `-DGGML_NATIVE=OFF` | **关键**。不开的话 CMake 会按 runner CPU 加 `-march=native`，产物会被钉死在 runner 的 CPU 上 |
| `avx512=true` → `AVX512=ON BMI2=ON` | 目标机支持 AVX-512，走这条路。AVX-512 主要加速 Q4_K_M 的点积累加，对本任务的 CPU MoE 路径有实质帮助 |
| `avx512=false` → `AVX512=OFF BMI2=OFF` | 保守档，只走 AVX2。用于对照与降级 |
| `-DGGML_AVX2=ON` 始终开 | ggml-cpu 有运行时分派，同时开 AVX2 + AVX-512 是安全的，会选最高可用档 |
| `-DCMAKE_CUDA_ARCHITECTURES=%CUDA_ARCH%` | **不指定的话会为多个 arch 各编译一遍 PTX，构建时间翻数倍** |
| 打包 CUDA 运行时 DLL | `cudart64_*.dll` / `cublas64_*.dll` / `cublasLt64_*.dll` 一起打进 artifact，避免目标机缺 DLL |

### 3.4 目标机 GPU arch 确认

```powershell
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

| 卡 | arch |
|---|---|
| GTX 10xx / P100 | `61` / `60` |
| RTX 20xx / T4 | `75` |
| **RTX 3050 / 3050 Ti / 3060** | **`86`** |
| RTX 40xx | `89` |
| RTX 50xx | `120` |

不是 86 就在 `Run workflow` 里填真实值，或改 yml 里的默认值。

### 3.5 目标机 CPU 指令集确认（构建前必做）

"支持 AVX-512"值得先核实一遍——11/12 代 Intel 桌面 CPU 上部分型号虽然硬件有 AVX-512 但被 fuse 掉了。

```powershell
:: 看 CPU 型号
wmic cpu get name

:: 权威做法是 Sysinternals coreinfo
coreinfo -f
::  输出里要同时出现这两行才算真的可用：
::    AVX512F     * Supports AVX-512 Foundation
::    AVX512VL    * Supports AVX-512 VL
```

若只有 `AVX2 *`，把 workflow 的 `avx512` 输入改成 `false`。
若不确定，**先用 `false` 构建一版跑通，再用 `true` 构建一版对比**——两版都能跑，因为 ggml-cpu 有运行时分派。

> **一个真实存在的反效果**：部分 Intel 消费级 CPU 使用 AVX-512 时会触发降频，实测可能反而比 AVX2 慢。所以 `avx512` 这个开关是**必须实测对比**的，不要假定它一定更快。第 7.2 节给了对比方法。

---

## 4. 取回产物（第 5 步）

Actions 页 → 对应的 run → **Artifacts** → 下载 `llama-cpp-win-cuda12-sm86-avx512`（若 `avx512=false` 则是 `…-avx2`）。

解压后应有：

```
llama-server.exe
llama-cli.exe
llama-bench.exe
llama-quant.exe  (等)
cudart64_12.dll
cublas64_12.dll
cublasLt64_12.dll
LICENSE-llama.cpp.txt
```

**整个目录拷到 Windows 目标机**（U 盘 / 局域网 / 任意方式）。CUDA DLL 与 exe 放同一目录即可，不需要在目标机装 CUDA Toolkit——但**显卡驱动必须是支持 CUDA 12 的版本**（≥ 525.60）。

---

## 5. 目标机预检（第 6 步，别跳过）

### 5.1 量真实模型体积

```powershell
dir Qwen3.6-35B-A3B-Q4_K_M*.gguf
```

### 5.2 量真实 dims

```powershell
python gguf-py\scripts\gguf_dump.py Qwen3.6-35B-A3B-Q4_K_M.gguf > dims.txt
```

（若目标机没 Python，用 Ubuntu 开发机上第 2.6 节生成的那份。）

回填：

| 项 | 默认 | 实际 |
|---|---|---|
| 文件体积（GB） | 20.0 | ___ |
| `*.embedding_length` | 2048 | ___ |
| `*.feed_forward_length` | 512 | ___ |
| `*.block_count` | 40 | ___ |
| `*.expert_count` | 256 | ___ |
| `*.expert_used_count` | 8 | ___ |

> 架构前缀可能是 `qwen35moe` 或其它，按同名后缀取。
> 若 `block_count` 含 nextn/MTP 块，**MoE 层数要减掉它**。

### 5.3 决策树（可用内存约 22 GB）

```
文件体积 S
│
├─ S <= 20.0 GB              ← 最可能的落点
│     ✅ 路线 A：--no-mmap --mlock
│        余量 0.7~2.7 GB，专家全部常驻，无缺页抖动
│        缓存预算 2458 MiB
│
├─ 20.0 < S <= 21.0 GB
│     ⚠️ 路线 B：--no-mmap 仍能跑，余量 < 1 GB
│        去掉 --mlock，缓存预算降到 2048 MiB
│
└─ S > 21.0 GB
      ❌ 路线 C：装不下。回传，改换 IQ4_XS / Q4_K_S（约 17-18 GB）
```

### 5.4 用真实 dims 重算预算

```powershell
python ..\moe-cache\budget.py --hidden <h> --intermediate <ff> --layers <L> --experts <E> --topk <k> --bpw 4.5 --vram 4 --free 2.4
```

---

## 6. 运行命令（第 7 步）

### 6.1 路线 A（默认）

```powershell
llama-server.exe ^
  -m Qwen3.6-35B-A3B-Q4_K_M.gguf ^
  -ngl 999 -ncmoe 999 ^
  --moe-expert-cache-budget 2458 ^
  -c 4096 -ub 512 ^
  -ctk q8_0 -ctv q8_0 ^
  -fa on ^
  --no-mmap --mlock ^
  -t 8
```

### 6.2 路线 B

去掉 `--mlock`，预算改 `2048`。

### 6.3 参数说明

- `-ncmoe 999`：40 层路由专家全部推给 CPU（4 GB 卡必须）
- `-c 4096`：4K 上下文，不要开大
- `-ub 512`：压住 prefill 运行时峰值内存
- `-t 8`：先用 8，第 7 节扫甜点
- `--mlock` 在 Windows 需 `SeLockMemoryPrivilege`，普通管理员会话通常失败。**日志报 mlock 失败就去掉**——`--no-mmap` 本身已保证常驻，mlock 只是防长时间闲置后被换出

### 6.4 认这三行日志

**启用成功：**
```
moecache: budget 2458 MiB -> 36 slots/layer across 40 layer(s), 1.69 MiB per expert (2430.0 MiB device, floor 16)
```

**自行禁用（预算低于盈亏下限）—— 设计行为，不是 bug：**
```
moecache: budget of 1024 MiB fits only 15 slot(s)/layer across 40 layer(s) (1.69 MiB per expert); 16 are needed to beat master (2x top_k=8) - cache disabled
```
看到这行 = 显存腾不够，回传改方向。

**每 512 步报命中率：**
```
moe-cache: steps=512 hits=... misses=... hit-rate=..%
```

### 6.5 若启动即崩溃

先确认是不是指令集问题：

```powershell
:: 用 AVX2 档的产物再试一次
llama-server.exe -m ...gguf -ngl 999 -ncmoe 999 -c 4096 ...
```

若 AVX2 档能跑、AVX-512 档崩溃，说明目标机 CPU 的 AVX-512 实际不可用（被 fuse 掉或型号不符）——回第 3.5 节用 `coreinfo` 核实，并把结论回传。

---

## 7. A/B 基准协议

```powershell
:: 基线 mmap（对照）
llama-bench.exe -m ...gguf -ngl 999 -ncmoe 999 -c 4096 -ub 512 -ctk q8_0 -ctv q8_0 -fa on -t 8 -p 0 -n 128 -r 5

:: 只加 --no-mmap（分离"常驻消除抖动"的贡献）
llama-bench.exe -m ...gguf -ngl 999 -ncmoe 999 -c 4096 -ub 512 -ctk q8_0 -ctv q8_0 -fa on --no-mmap -t 8 -p 0 -n 128 -r 5

:: --no-mmap + 缓存
llama-bench.exe -m ...gguf -ngl 999 -ncmoe 999 --moe-expert-cache-budget 2458 -c 4096 -ub 512 -ctk q8_0 -ctv q8_0 -fa on --no-mmap -t 8 -p 0 -n 128 -r 5
```

要求：
- 各跑 **3 次取中位数**（MoE CPU 路径噪声大）
- 每次之间静置 30 s
- 记录 `tg128`，**同时记录 `pp512`**
- 缓存是 decode-only，`pp512` 应基本不变；降幅 > 10% 说明常驻挤了显存，判失败并调低预算

**中间那组必须做**，否则会把"常驻消除抖动"的收益误记到缓存头上。

回传表格：

| 配置 | tg128 (t/s) | pp512 (t/s) | 命中率 | 备注 |
|---|---|---|---|---|
| 基线 mmap | 18.0（对照） | | | |
| `--no-mmap` | | | | 分离常驻贡献 |
| `--no-mmap` + 缓存 2458 | | | | |

### 7.1 线程数扫描（顺手做）

```powershell
for %t in (4 6 8 10 12) do llama-bench.exe -m ...gguf -ngl 999 -ncmoe 999 -c 4096 -ub 512 --no-mmap -t %t -p 0 -n 128 -r 3
```
基线是内存带宽受限，线程甜点常见 6–10。零风险，**与缓存收益叠加**。

### 7.2 AVX-512 vs AVX2 对比（本次新增，别跳过）

因为警告过 AVX-512 在部分 Intel 消费级 CPU 上会触发降频，**必须实测**，不能假定它更快：

```powershell
:: 两版产物各跑一次，同一命令
llama-bench-avx512.exe -m ...gguf -ngl 999 -ncmoe 999 -c 4096 -ub 512 -ctk q8_0 -ctv q8_0 -fa on --no-mmap -t 8 -p 0 -n 128 -r 5
llama-bench-avx2.exe   -m ...gguf -ngl 999 -ncmoe 999 -c 4096 -ub 512 -ctk q8_0 -ctv q8_0 -fa on --no-mmap -t 8 -p 0 -n 128 -r 5
```

判据：
- AVX-512 版更快 → 用它
- AVX-512 版持平或更慢 → **用 AVX2 版**（降频吃掉了收益，且更省电、发热更低）
- 若 AVX-512 版启动就崩 → 目标机实际不支持，用 AVX2 版并回传 CPU 型号

> 建议顺序：**先用 `avx512=false` 构建一版跑通基线**，再用 `true` 构建一版对比。跑通优先于跑快。

---

## 8. 故障排查

### 8.1 构建阶段（GitHub Actions）

| 症状 | 处理 |
|---|---|
| CUB / CCCL 相关编译错误 | 在 Configure 步骤加 `-DGGML_CUDA_CUB_3DOT2=ON` |
| `nvcc not found` | CUDA action 失败，看它的 log artifact（workflow 里配了 `log-file-suffix`） |
| 找不到 MSVC | 确认 `runs-on: windows-2022`（不是 `ubuntu-*`）；VS generator 会自动找工具链 |
| 构建超时（> 6 h） | 检查 `CMAKE_CUDA_ARCHITECTURES` 是否只填了一个 arch |
| `budget knob MISSING` | workflow 里"Report the applied MoE cache patch"步骤会报。补丁没推上去，回第 2.4 步 |

### 8.2 运行阶段（Windows 目标机）

| 症状 | 处理 |
|---|---|
| 缺 `cudart64_12.dll` | DLL 没和 exe 放同目录，或驱动太旧（需 ≥ 525.60） |
| 非法指令崩溃 | 用 `avx512=false` 重新构建一版。若 AVX2 版正常，说明目标机 AVX-512 实际不可用（见 3.5 / 6.5） |
| CUDA 13 相关报错 | 本工作流锁 CUDA 12.4.1，不应出现。若出现，检查 `Jimver/cuda-toolkit` 的 `cuda:` 输入是否被改 |
| 启动即 OOM | 模型 > 21 GB，走路线 C |
| 长会话突然变慢 | OS 把页换出了。申请锁页权限并加 `--mlock` |
| 速度断崖下跌 | Windows WDDM 静默分页。查 `nvidia-smi` 的 shared memory 占用 |

---

## 9. 已知风险与禁止事项

| 项 | 说明 |
|---|---|
| **不要尝试交叉编译** | mingw 无法产出可用的 Windows CUDA 二进制，CUDA 工具链依赖 MSVC 的 cl.exe 与 .lib |
| **CUDA 锁 12.4.1，不要升 13** | 目标机只支持到 CUDA 12；llama.cpp 当前 CMake 对 CUDA 13 的 arch 探测也有已知问题 |
| **不要假定 AVX-512 一定更快** | 部分 Intel 消费级 CPU 用 AVX-512 会降频。必须按 7.2 实测对比，慢就用 AVX2 档 |
| **不要叠加 MTP / 投机解码** | PR #27861 是 `n_tokens == 1` 的 decode-only 路径，投机解码绕过缓存，**两者不叠加** |
| **不要超额分配预算** | koren1712 实测 8 GiB slab 8.70 t/s vs 16 GiB slab 7.16 t/s，命中数完全相同。超过 8× top_k 是纯亏 |
| **补丁未经编译验证** | 按 PR #27861 代码锚点做字符串替换写成。构建报错请回传完整 stderr，**不要自行改 C++** |
| **PR #27861 未合入主干** | 已 open 约 2 个月，维护者态度不明。这是 fork，不是主线特性 |

---

## 10. 验收与回传

### 10.1 验收标准

- [ ] GitHub Actions 构建成功，artifact 能下载
- [ ] artifact 里 `llama-server.exe` 与三个 CUDA DLL 齐全
- [ ] 目标机能启动，日志出现第 6.4 节三行之一
- [ ] A/B 三组表格填完
- [ ] **AVX-512 vs AVX2 对比已做（7.2）**
- [ ] `pp512` 降幅 < 10%
- [ ] 第 5.2 节真实体积与 dims 已回填

### 10.2 回传清单

1. `dims.txt`
2. 第 5.2 节真实参数表（**含文件体积**）
3. 目标机 CPU 型号 + `coreinfo` 里 AVX-512 是否真的可用
4. 预检结论：路线 A / B / C
5. 启动日志前 100 行（含 moecache 行与 mlock 结果）
6. 第 7 节 A/B 表格（三组）
7. 第 7.1 节线程扫描结果
8. 第 7.2 节 AVX-512 vs AVX2 对比，及最终选用哪一档
9. 若构建失败：Actions 的完整 log

---

## 附：交付文件

| 文件 | 用途 |
|---|---|
| `cicd/build-win-cuda12.yml` | GitHub Actions 工作流（放到仓库 `.github/workflows/`） |
| `moe-cache/0001-moe-expert-cache-upstream-PR27861.patch` | PR #27861 原文（826 行 / 12 文件） |
| `moe-cache/apply_budget_patch.py` | 小显存适配：把"填槽位数"改成"填显存预算"，加 2× top_k 盈亏下限 |
| `moe-cache/budget.py` | 预算计算器 |
| `moe-cache/README.md` | 补丁原理与完整说明 |
