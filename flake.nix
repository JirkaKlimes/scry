{
  description = "grep for images: search by metadata, OCR text, and meaning";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      nixpkgs,
      rust-overlay,
      ...
    }:
    let
      inherit (nixpkgs) lib;
      forAllSystems =
        f:
        lib.genAttrs [ "x86_64-linux" "aarch64-linux" ] (
          system:
          f (
            import nixpkgs {
              inherit system;
              overlays = [ rust-overlay.overlays.default ];
            }
          )
        );
      toolchainFile = ./rust-toolchain.toml;
      manifest = (lib.importTOML ./Cargo.toml).package;
    in
    {
      packages = forAllSystems (
        pkgs:
        let
          # Same pinned version as the dev shell, without the editor components.
          rust = pkgs.rust-bin.fromRustupToolchain {
            inherit ((lib.importTOML toolchainFile).toolchain) channel;
            profile = "minimal";
          };
          rustPlatform = pkgs.makeRustPlatform {
            cargo = rust;
            rustc = rust;
          };
        in
        {
          default = rustPlatform.buildRustPackage {
            pname = manifest.name;
            inherit (manifest) version;
            src = lib.fileset.toSource {
              root = ./.;
              fileset = lib.fileset.unions [
                ./Cargo.toml
                ./Cargo.lock
                ./src
              ];
            };
            cargoLock.lockFile = ./Cargo.lock;
            # CI runs the tests. Here they would only add a second full build.
            doCheck = false;
            meta = {
              inherit (manifest) description;
              homepage = manifest.repository;
              license = with lib.licenses; [
                mit
                asl20
              ];
              mainProgram = "scry";
            };
          };
        }
      );

      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = [
            (pkgs.rust-bin.fromRustupToolchainFile toolchainFile)
            # Runs the working tree. --manifest-path keeps the caller's directory, so relative
            # paths work from anywhere.
            (pkgs.writeShellScriptBin "scry" ''
              exec cargo run --release --quiet --manifest-path "$SCRY_ROOT/Cargo.toml" -- "$@"
            '')
          ];
          shellHook = ''
            export SCRY_ROOT="$PWD"
          '';
        };
      });
    };
}
