# build-win.ps1
# Build llama-server / llama-bench / llama-cli / quants for Windows x64 + CUDA 12.
# Run on the TARGET Windows machine (the GitHub-hosted windows runner's CUDA
# silent installer fails with 0xE0E00019, so we build where CUDA already works).
#
# The checkout MUST be the patched build-win-cuda12 branch (MoE expert cache +
# budget knob). If you cloned fresh:  git clone -b build-win-cuda12 <repo>
#
# PREREQS on the target machine:
#   - Visual Studio 2022 (or Build Tools) with "Desktop development with C++"
#   - NVIDIA CUDA Toolkit 12.x (nvcc on PATH or CUDA_PATH set)
#   - CMake >= 3.20, Git
#
# DETECT AVX-512 BEFORE choosing -Avx512. Download coreinfo from Sysinternals:
#   coreinfo.exe -f | findstr /i AVX
# If it does NOT print "AVX512" then pass -Avx512:$false (build AVX2 only).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File build-win.ps1  -CudaArch 86 -Avx512:$false
#   powershell -ExecutionPolicy Bypass -File build-win.ps1  -CudaArch 89 -Avx512:$true

param(
    [int]   $CudaArch = 86,     # 86=RTX 3050/3050Ti/3060, 89=RTX 40xx, 75=RTX 20xx, 61=GTX 10xx
    [bool]  $Avx512 = $false,   # set $true ONLY if coreinfo.exe shows the CPU supports AVX-512
    [string]$CudaPath = ""      # optional; default = $env:CUDA_PATH or Program Files detection
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
if (-not (Test-Path (Join-Path $root 'src\llama-moecache.cpp'))) {
    throw "Not the patched checkout: src\llama-moecache.cpp not found. Checkout branch build-win-cuda12."
}

# --- resolve CUDA ---
if ($CudaPath -eq "" -and $env:CUDA_PATH) { $CudaPath = $env:CUDA_PATH }
if ($CudaPath -eq "") {
    $cand = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
    if (Test-Path $cand) {
        $ver = Get-ChildItem $cand -Directory | Sort-Object Name -Descending | Select-Object -First 1
        if ($ver) { $CudaPath = $ver.FullName }
    }
}
if ($CudaPath -eq "" -or -not (Test-Path "$CudaPath\bin\nvcc.exe")) {
    throw "CUDA toolkit not found. Install CUDA 12.x or pass -CudaPath <dir>. nvcc must be at <dir>\bin\nvcc.exe"
}
Write-Host "CUDA_PATH = $CudaPath"

# --- toolchain ---
$vs = vswhere 2>$null
$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmake) { throw "cmake not found on PATH" }
Write-Host "cmake: $($cmake.Source)"

# --- AVX flags ---
if ($Avx512) {
    $avxFlags = @('-DGGML_AVX512=ON', '-DGGML_BMI2=ON')
    $tag = 'avx512'
    Write-Host "AVX-512 requested; verify with coreinfo.exe"
} else {
    $avxFlags = @('-DGGML_AVX512=OFF', '-DGGML_BMI2=OFF')
    $tag = 'avx2'
    Write-Host "Building AVX2 (avx512=$Avx512)"
}

# --- configure ---
$buildDir = Join-Path $root 'build'
& cmake -S $root -B $buildDir `
    -G "Visual Studio 17 2022" -A x64 `
    -DGGML_CUDA=ON `
    -DCMAKE_CUDA_ARCHITECTURES=$CudaArch `
    -DGGML_NATIVE=OFF `
    -DGGML_AVX=ON `
    -DGGML_AVX2=ON `
    -DGGML_FMA=ON `
    -DGGML_F16C=ON `
    $avxFlags `
    -DLLAMA_BUILD_SERVER=ON `
    -DBUILD_SHARED_LIBS=OFF `
    -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed ($LASTEXITCODE)" }

# --- build ---
& cmake --build $buildDir --config Release -j $env:NUMBER_OF_PROCESSORS
if ($LASTEXITCODE -ne 0) { throw "cmake build failed ($LASTEXITCODE)" }

# --- bundle ---
$dist = Join-Path $root "dist"
New-Item -ItemType Directory -Force -Path $dist | Out-Null
Copy-Item "$buildDir\bin\Release\*.exe" $dist
Copy-Item "$CudaPath\bin\cudart64_*.dll" $dist
Copy-Item "$CudaPath\bin\cublas64_*.dll"   $dist
Copy-Item "$CudaPath\bin\cublasLt64_*.dll" $dist
if (Test-Path (Join-Path $root 'LICENSE')) { Copy-Item (Join-Path $root 'LICENSE') (Join-Path $dist 'LICENSE-llama.cpp.txt') }

Write-Host ""
Write-Host "=== BUILD OK: sm$CudaArch-$tag ==="
Get-ChildItem $dist | Where-Object { -not $_.PSIsContainer } | Select-Object Name, @{n='SizeMB';e={[math]::Round($_.Length/1MB,2)}}
Write-Host "Artifacts in: $dist"