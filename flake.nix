{
  description = "Nix flake for plot-digitizer";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      packages = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          python = pkgs.python312;
        in
        {
          default = python.pkgs.buildPythonApplication {
            pname = "digitizer";
            version = "0.1.0";
            format = "pyproject";
            src = self;

            nativeBuildInputs = with python.pkgs; [
              hatchling
            ];

            propagatedBuildInputs = with python.pkgs; [
              matplotlib
              numpy
              opencv4
              pandas
              scikit-image
              scikit-learn
              scipy
            ];

            doCheck = false;
          };
        });

      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        {
          default = pkgs.mkShell {
            packages = [
              (pkgs.python312.withPackages (ps: with ps; [
                hatchling
                matplotlib
                numpy
                opencv4
                pandas
                pytest
                scikit-image
                scikit-learn
                scipy
              ]))
              pkgs.uv
            ];
          };
        });
    };
}
