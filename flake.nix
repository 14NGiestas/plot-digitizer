{
  description = "Nix flake for plot-digitizer";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/50ab793786d9de88ee30ec4e4c24fb4236fc2674"; # pinned intentionally (matches 14NGiestas/mfi lock)
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true; # required for CUDA packages
        };
        # Nix package overrides that add ultralytics (and its sub-dependency
        # ultralytics-thop) to whichever Python package set they are applied
        # to via `python.override { packageOverrides = aiPackageOverrides; }`.
        # torch / torchvision are intentionally omitted: they are
        # GPU-flavour-specific and must be installed separately, e.g.
        #   pip install torch torchvision \
        #     --index-url https://download.pytorch.org/whl/<rocm|cu124|cu118>
        aiPackageOverrides = pyfinal: _pyprev: {
          ultralytics-thop = pyfinal.buildPythonPackage rec {
            pname = "ultralytics-thop";
            version = "2.0.19";
            format = "wheel";
            src = pkgs.fetchurl {
              url = "https://files.pythonhosted.org/packages/6a/74/af3e40919305f16968ea3ab88d84b511d710dd281eb5dafaf4897579dd22/ultralytics_thop-2.0.19-py3-none-any.whl";
              sha256 = "0fd7jb4gk47xkjh0rlf6awd24ss0xy636wim7ynz5w437gmvm051";
            };
            propagatedBuildInputs = with pyfinal; [ numpy ];
            doCheck = false;
          };
          ultralytics = pyfinal.buildPythonPackage rec {
            pname = "ultralytics";
            version = "8.3.53";
            format = "wheel";
            src = pkgs.fetchurl {
              url = "https://files.pythonhosted.org/packages/5e/12/bdb1a1c0cd48054fd472f28dac65d7ff88b2f34c9097ac80e47fe9257f3b/ultralytics-8.3.53-py3-none-any.whl";
              sha256 = "1hj88nk0yvlnc4i15i2qpywakydbxnh1dmd8482m41sy38sx02i4";
            };
            propagatedBuildInputs = (with pyfinal; [
              numpy matplotlib opencv4 pillow pyyaml requests scipy
              tqdm psutil pandas seaborn
            ]) ++ [
              pyfinal."py-cpuinfo"
              pyfinal."ultralytics-thop"
            ];
            doCheck = false;
          };
        };
        # Apply the AI overrides to the base Python so that `aiPythonPkgs
        # python.pkgs` finds ultralytics as a fallback for GPU shells whose
        # own package set does not carry it.
        python = pkgs.python312.override { packageOverrides = aiPackageOverrides; };
        commonSystemLibs = with pkgs; [
          # In this pinned nixpkgs revision, libxcb is under xorg.
          xorg.libxcb
        ];
        packagedCli = python.pkgs.buildPythonApplication {
          pname = "digitizer";
          version = "0.1.0";
          format = "pyproject";
          src = self;

          nativeBuildInputs = with python.pkgs; [
            hatchling
          ];

          buildInputs = commonSystemLibs;

          propagatedBuildInputs = with python.pkgs; [
            matplotlib
            numpy
            opencv4
            pandas
            scikit-image
            scikit-learn
            scipy
          ];

          pythonRemoveDeps = [
            "opencv-python"
          ];

          doCheck = true;
          checkPhase = ''
            ${python.interpreter} -m unittest discover -s tests -p 'test_*.py' -v
          '';
        };
        shellPythonPathHook = ''
          if [ -d "$PWD/src/digitizer" ]; then
            digitizer_src_root="$PWD"
          else
            digitizer_src_root="${self}"
          fi
          export DIGITIZER_SRC_ROOT="$digitizer_src_root"
          export PYTHONPATH="$digitizer_src_root/src''${PYTHONPATH:+:$PYTHONPATH}"
          # Ensure the dev-shell digitizer wrapper takes precedence over any
          # nix-profile-installed packaged CLI (e.g. from `nix profile install .`).
          export PATH="${digitizerShellCommand}/bin''${PATH:+:$PATH}"
        '';

        # Core Python packages shared across all shells
        corePythonPkgs = ps: with ps; [
          hatchling
          matplotlib
          numpy
          opencv4
          pandas
          pytest
          scikit-image
          scikit-learn
          scipy
        ];
        # Build a shellHook fragment that auto-creates an accelerator-specific
        # Python venv (with --system-site-packages so it inherits ultralytics
        # and other Nix-provided packages) and installs torch/torchvision from
        # the given PyTorch wheel index on first entry (or if torch is missing).
        # The venv is activated for the interactive shell session.
        mkAiVenvHook = { venvName, torchIndexUrl }: ''
          _ai_venv="''${DIGITIZER_SRC_ROOT:-$PWD}/.${venvName}"
          if [ ! -d "$_ai_venv" ] || ! "$_ai_venv/bin/python" -c "import torch; import torchvision; import numpy" 2>/dev/null; then
            echo "Setting up AI environment (${venvName}) — installing torch/torchvision..."
            python -m venv --system-site-packages "$_ai_venv"
            "$_ai_venv/bin/pip" install --quiet torch torchvision \
              --index-url ${torchIndexUrl}
            echo "torch/torchvision installed into ''${_ai_venv}."
          fi
          . "$_ai_venv/bin/activate"
        '';

        # `ps` is the shell-selected Python package set; `defaultPs` is a
        # fallback set used when shell-specific overlays do not export
        # `ultralytics` (observed in some CUDA legacy package-set layouts).
        # Prefer the shell package set first so accelerator-tuned builds win.
        aiPythonPkgs = defaultPs: ps:
          let
            hasShellUltralytics = ps ? ultralytics;
            hasDefaultUltralytics = defaultPs ? ultralytics;
          in
          pkgs.lib.optionals hasShellUltralytics [ ps.ultralytics ]
          ++ pkgs.lib.optionals (!hasShellUltralytics && hasDefaultUltralytics) [ defaultPs.ultralytics ];
        digitizerShellCommand = pkgs.writeShellScriptBin "digitizer" ''
          exec python -m digitizer "$@"
        '';

        # Factory: build a dev shell with optional extra system packages, Python packages, and hook
        mkPyShell = {
          shellPython ? python,
          extraPkgs ? [],
          extraPythonPkgs ? (_: []),
          shellHook ? "",
        }: pkgs.mkShell {
          packages = [
            (shellPython.withPackages (ps: corePythonPkgs ps ++ extraPythonPkgs ps))
            digitizerShellCommand
            pkgs.uv
          ] ++ commonSystemLibs ++ extraPkgs;
          shellHook = shellPythonPathHook + shellHook;
        };
        cpuShell = mkPyShell {
          extraPythonPkgs = aiPythonPkgs python.pkgs;
          shellHook = mkAiVenvHook {
            venvName = "venv-ai-cpu";
            torchIndexUrl = "https://download.pytorch.org/whl/cpu";
          } + ''
            echo "CPU shell ready. Ultralytics + torch active."
          '';
        };

        # GPU-specific shells are only meaningful on Linux
        gpuShells = pkgs.lib.optionalAttrs pkgs.stdenv.isLinux (
          let
            rocmPkgs = if pkgs ? pkgsRocm then pkgs.pkgsRocm else pkgs;
            cudaPkgs = if pkgs ? pkgsCuda then pkgs.pkgsCuda else pkgs;
            cudaLegacyPkgs = pkgs.cudaPackages_11_8;
            # Prefer legacy-set Python first (python310 when exposed) for CUDA
            # 11.8 compatibility, then progressively fall back to broader sets.
            cudaLegacyPython =
              if cudaLegacyPkgs ? python310 then cudaLegacyPkgs.python310
              else if cudaLegacyPkgs ? python then cudaLegacyPkgs.python
              else if cudaPkgs ? python312 then cudaPkgs.python312
              else pkgs.python312;
            cudaLegacyPythonSource =
              if cudaLegacyPkgs ? python310 then "cudaLegacyPkgs.python310"
              else if cudaLegacyPkgs ? python then "cudaLegacyPkgs.python"
              else if cudaPkgs ? python312 then "cudaPkgs.python312"
              else "pkgs.python312";
            cudaLegacyPythonVersion =
              if cudaLegacyPython ? pythonVersion then cudaLegacyPython.pythonVersion
              else if cudaLegacyPython ? version then cudaLegacyPython.version
              else "unknown";

            # --- ROCm / HIP (AMD GPU) ---
            rocmLibs = with rocmPkgs.rocmPackages; [
              rocm-runtime  # HSA runtime  (libhsa-runtime64.so)
              clr           # HIP + OpenCL (libamdhip64.so)
              rocblas       # ROCm BLAS
            ];

            # --- CUDA (NVIDIA GPU) ---
            cudaLibs = with cudaPkgs.cudaPackages; [
              cuda_cudart  # CUDA runtime
              libcublas    # cuBLAS
            ];

            # --- CUDA legacy (driver 470 class via CUDA 11.8) ---
            cudaLegacyLibs = with cudaLegacyPkgs; [
              cuda_cudart  # CUDA runtime
              libcublas    # cuBLAS
            ];
          in
          {
            # AMD GPU — ROCm/HIP
            # Tested on: Ryzen 7 8745HS (Hawk Point) with Radeon 780M iGPU (gfx1103 / RDNA3).
            # The iGPU shares system RAM (DDR5) as both CPU and GPU memory.
            #
            rocm = mkPyShell {
              shellPython = rocmPkgs.python312.override { packageOverrides = aiPackageOverrides; };
              extraPkgs = rocmLibs;
              extraPythonPkgs = aiPythonPkgs python.pkgs;
              shellHook = ''
                export ROCM_PATH="${rocmPkgs.rocmPackages.rocm-runtime}"
                export HIP_PATH="${rocmPkgs.rocmPackages.clr}"
                export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath rocmLibs}"''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
                # Hawk Point APU (Ryzen 8000-series) integrates Radeon 780M — gfx1103.
                # HSA_OVERRIDE_GFX_VERSION forces the correct ISA when the ROCm runtime
                # cannot auto-detect the iGPU (common on newer APUs).
                export HSA_OVERRIDE_GFX_VERSION="11.0.3"
              '' + mkAiVenvHook {
                venvName = "venv-ai-rocm";
                torchIndexUrl = "https://download.pytorch.org/whl/rocm6.2";
              } + ''
                echo "ROCm shell ready (gfx1103 / Radeon 780M). Ultralytics + torch active."
              '';
            };

            # NVIDIA GPU — CUDA
            cuda = mkPyShell {
              shellPython = cudaPkgs.python312.override { packageOverrides = aiPackageOverrides; };
              extraPkgs = cudaLibs;
              extraPythonPkgs = aiPythonPkgs python.pkgs;
              shellHook = ''
                export CUDA_PATH="${cudaPkgs.cudaPackages.cuda_cudart}"
                export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath cudaLibs}"''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
              '' + mkAiVenvHook {
                venvName = "venv-ai-cuda";
                torchIndexUrl = "https://download.pytorch.org/whl/cu124";
              } + ''
                echo "CUDA shell ready. Ultralytics + torch active."
              '';
            };

            # NVIDIA GPU — CUDA legacy (driver 470 class via CUDA 11.8 userspace)
            cuda-legacy = mkPyShell {
              shellPython = cudaLegacyPython.override { packageOverrides = aiPackageOverrides; };
              extraPkgs = cudaLegacyLibs;
              extraPythonPkgs = aiPythonPkgs python.pkgs;
              shellHook = ''
                export CUDA_PATH="${cudaLegacyPkgs.cuda_cudart}"
                export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath cudaLegacyLibs}"''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
              '' + mkAiVenvHook {
                venvName = "venv-ai-cuda-legacy";
                torchIndexUrl = "https://download.pytorch.org/whl/cu118";
              } + ''
                echo "CUDA legacy shell ready (Python ${cudaLegacyPythonVersion} from ${cudaLegacyPythonSource}, CUDA 11.8 userspace). Ultralytics + torch active."
              '';
            };
          }
        );
      in
      {
        packages.default = packagedCli;

        apps.default = {
          type = "app";
          program = "${pkgs.writeShellScript "digitizer-app" ''
            # Inside a dev shell, prefer that shell's Python only when digitizer
            # resolves from a src-layout path rather than an unrelated global
            # installation.
            in_nix_shell="''${IN_NIX_SHELL:-}"
            python_available=1
            digitizer_from_src=1

            if ! command -v python >/dev/null 2>&1; then
              python_available=0
            elif ! python >/dev/null 2>&1 <<'PY'
import os
from pathlib import Path
import digitizer
import sys

source_root = os.environ.get("DIGITIZER_SRC_ROOT")
if not source_root:
    sys.exit(1)

module_path = Path(digitizer.__file__).resolve()
src_package_dir = (Path(source_root) / "src" / "digitizer").resolve()
is_src_layout = src_package_dir in module_path.parents
sys.exit(0 if is_src_layout else 1)
PY
            then
              digitizer_from_src=0
            fi

            if [ -n "$in_nix_shell" ] && [ "$python_available" -eq 1 ] && [ "$digitizer_from_src" -eq 1 ]; then
              exec python -m digitizer "$@"
            fi
            exec ${packagedCli}/bin/digitizer "$@"
          ''}";
        };

        devShells = {
          # Default shell — CPU-only, works on all platforms, used in CI
          default = cpuShell;
          cpu-only = cpuShell;
        } // gpuShells;
      }
    );
}
