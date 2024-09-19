#[allow(dead_code)]
mod bindings;

use bindings::{
    exports::wasi::cli::run::Guest,
    wasi::nn::{
        graph::{load, load_by_name, ExecutionTarget, GraphBuilder, GraphEncoding},
        tensor::{Tensor, TensorData, TensorDimensions, TensorType},
    },
};
use tracing_subscriber::fmt::format;
use std::{fs, io::{self, Write}};
use clap::Parser;

struct Component;

#[derive(Debug, Parser)]
#[command(author, about, version, long_about=None)]
struct Cli{
    prompt:String,
}

impl Guest for Component {
    #[export_name = "wasi:cli/run@0.2.0#run"]
    fn run() -> Result<(), ()> {
        let args=Cli::parse();

        let graph = load_by_name("fixture").unwrap();
        println!("Loaded graph into wasi-nn");

        let exec_context = graph.init_execution_context().unwrap();
        println!("Created wasi-nn execution context.");

        let prompt=format!("<|system|>You are a helpful assistant.<|end|><|user|>{}<|end|>", args.prompt);

        let data: TensorData = prompt.as_bytes().to_vec();
        let dimensions: TensorDimensions = vec![data.len() as u32];
        let tensor = Tensor::new(&dimensions, TensorType::U8, &data);
        exec_context.set_input("prompt", tensor);
        exec_context.compute();
        // println!("Waiting for result");
        for i in 0..5000 {
            let output = exec_context.get_output("output");
            print!("{}", String::from_utf8(output.unwrap().data()).unwrap());
            io::stdout().flush();
        }
        Ok(())
    }
}
