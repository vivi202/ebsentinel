# ebsentinel
This my playground project exploring the use of eBPF (Extended Berkeley Packet Filter) to monitor system calls of processes and detect anomalies using machine learning techniques.

The project aims to combine eBPF's efficiency for tracing kernel events with the power of machine learning to identify abnormal behavior in real-time.

## Prerequisites

1. stable rust toolchains: `rustup toolchain install stable`
1. nightly rust toolchains: `rustup toolchain install nightly --component rust-src`
1. (if cross-compiling) rustup target: `rustup target add ${ARCH}-unknown-linux-musl`
1. (if cross-compiling) LLVM: (e.g.) `brew install llvm` (on macOS)
1. (if cross-compiling) C toolchain: (e.g.) [`brew install filosottile/musl-cross/musl-cross`](https://github.com/FiloSottile/homebrew-musl-cross) (on macOS)
1. bpf-linker: `cargo install bpf-linker` (`--no-default-features` on macOS)

## Build & Run

Use `cargo build`, `cargo check`, etc. as normal. Run your program with:

```shell
cargo run --release --config 'target."cfg(all())".runner="sudo -E"'
```

Cargo build scripts are used to automatically build the eBPF correctly and include it in the
program.

## Cross-compiling on macOS

Cross compilation should work on both Intel and Apple Silicon Macs.

```shell
CC=${ARCH}-linux-musl-gcc cargo build --package ebsentinel --release \
  --target=${ARCH}-unknown-linux-musl \
  --config=target.${ARCH}-unknown-linux-musl.linker=\"${ARCH}-linux-musl-gcc\"
```
The cross-compiled program `target/${ARCH}-unknown-linux-musl/release/ebsentinel` can be
copied to a Linux server or VM and run there.

# Overview
The `ebsentinel` project has three main components.

## `ebsentinel-rec` 
  A CLI tool to monitor system calls of a specific process and save them into a SQLite database.
  
## `ebsentinel-train` 
  A CLI tool for training the anomaly detection model based on the data collected by `ebsentinel-rec`.
  
## `ebsentinel`
The main CLI tool that uses the trained model to perform real-time anomaly detection on a running process.


# How to use this?
1. Create a directory to store the dataset and model configuration `mkdir test && cd test`
2. `ebsentinel-rec <PID>` to create the training dataset.
3. `ebsentinel-rec -t <PID>` to create the validation dataset.
4. `ebsentinel-train` to train the model.
5. `ebsentinel <PID> <THRESHOLD>` to detect anomalies in real-time.



