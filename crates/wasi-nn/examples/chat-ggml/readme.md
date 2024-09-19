# GGML Backend Chat Example
This example only is only tested with Phi-3 mini model on Ubuntu 24.04.

## Download the model
You may download the Phi-3 model from [hugging face](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf), and put it in a directory. This directory will be used later as the value of `nn-graph` in command line.

## Build
In this directory, run the following command to build the WebAssembly component:
```shell
cargo component build --target wasm32-wasip1
```

In the Wasmtime root directory, run the following command to build the Wasmtime CLI and run the WebAssembly component:
```shell
# build wasmtime with component-model and WASI-NN with GGML llama.cpp runtime support
cargo build -p wasmtime-cli --features wasmtime-wasi-nn/ggml

# run the component with wasmtime
./target/debug/wasmtime run --wasm component-model=y --wasi nn=y --wasi nn-graph=ggml::/path/to/phi-3/model/dir/ --dir ./crates/wasi-nn/examples/chat-ggml/fixture/::fixture crates/wasi-nn/examples/chat-ggml/target/wasm32-wasip1/release/chat_ggml.wasm "This is a prompt"
```
