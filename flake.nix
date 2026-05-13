{
  description = "Nix flake for plot-digitizer";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true; # required for CUDA packages
        };
        python = pkgs.python312;
        commonSystemLibs = with pkgs; [
          # `xorg.libxcb` was renamed to `libxcb` in current Nixpkgs.
          libxcb
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

        # GPU-specific shells are only meaningful on Linux
        gpuShells = pkgs.lib.optionalAttrs pkgs.stdenv.isLinux (
          let
            rocmPkgs = pkgs.pkgsRocm;
            cudaPkgs = pkgs.pkgsCuda;

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
          in
          {
            # AMD GPU — ROCm/HIP
            # Tested on: Ryzen 7 8745HS (Hawk Point) with Radeon 780M iGPU (gfx1103 / RDNA3).
            # The iGPU shares system RAM (DDR5) as both CPU and GPU memory.
            #
            rocm = mkPyShell {
              shellPython = rocmPkgs.python312;
              extraPkgs = rocmLibs;
              extraPythonPkgs = ps: with ps; [ ultralytics ];
              shellHook = ''
                export ROCM_PATH="${rocmPkgs.rocmPackages.rocm-runtime}"
                export HIP_PATH="${rocmPkgs.rocmPackages.clr}"
                export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath rocmLibs}:$LD_LIBRARY_PATH"
                # Hawk Point APU (Ryzen 8000-series) integrates Radeon 780M — gfx1103.
                # HSA_OVERRIDE_GFX_VERSION forces the correct ISA when the ROCm runtime
                # cannot auto-detect the iGPU (common on newer APUs).
                export HSA_OVERRIDE_GFX_VERSION="11.0.3"
                echo "ROCm shell ready (gfx1103 / Radeon 780M)."
                echo "AI dependencies are included by default in this shell."
              '';
            };

            # NVIDIA GPU — CUDA
            cuda = mkPyShell {
              shellPython = cudaPkgs.python312;
              extraPkgs = cudaLibs;
              extraPythonPkgs = ps: with ps; [ ultralytics ];
              shellHook = ''
                export CUDA_PATH="${cudaPkgs.cudaPackages.cuda_cudart}"
                export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath cudaLibs}:$LD_LIBRARY_PATH"
                echo "CUDA shell ready."
                echo "AI dependencies are included by default in this shell."
              '';
            };

            # NVIDIA GPU — CUDA legacy (driver 470 / CUDA 11.4 class systems)
            cuda-legacy = mkPyShell {
              shellPython = cudaPkgs.python312;
              extraPkgs = cudaLibs;
              extraPythonPkgs = ps: with ps; [ ultralytics ];
              shellHook = ''
                export CUDA_PATH="${cudaPkgs.cudaPackages.cuda_cudart}"
                export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath cudaLibs}:$LD_LIBRARY_PATH"
                echo "CUDA legacy shell ready (Python 3.10)."
                echo "AI dependencies are included by default in this shell."
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
          default = mkPyShell {};
        } // gpuShells;
      }
    );
}
