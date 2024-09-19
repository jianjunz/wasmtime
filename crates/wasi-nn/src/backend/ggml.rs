//! Implements a `wasi-nn` [`BackendInner`] using llm-chain.
//!
//! [llm-chain]: https://github.com/sobelio/llm-chain

use crate::backend::{
    BackendError, BackendExecutionContext, BackendFromDir, BackendGraph, BackendInner, Id,
};
use crate::wit::{ExecutionTarget, GraphEncoding, Tensor, TensorType};
use crate::{ExecutionContext, Graph};
use core::str;
use llama_cpp::{
    CompletionHandle, LlamaContextError, LlamaLoadError, LlamaModel, LlamaParams, LlamaSession,
    SessionParams,
};
use std::str::Utf8Error;
use std::{
    fs::File,
    io::Read,
    io::{stdout, Write},
    mem::size_of,
    path::Path,
    sync::Arc,
};

#[derive(Default)]
pub struct GgmlBackend();

impl BackendInner for GgmlBackend {
    fn encoding(&self) -> GraphEncoding {
        GraphEncoding::Ggml
    }

    fn load(&mut self, builders: &[&[u8]], target: ExecutionTarget) -> Result<Graph, BackendError> {
        // Because llama.cpp doesn't support loading model from memory, we take the first argument as a utf-8 encoded path to the model file.
        // https://github.com/ggerganov/llama.cpp/issues/6311
        if builders.len() != 1 {
            return Err(BackendError::InvalidNumberOfBuilders(1, builders.len()).into());
        }
        let path = Path::new(str::from_utf8(builders[0]).unwrap());
        let model = LlamaModel::load_from_file(path, LlamaParams::default())?;
        let graph = GgmlGraph { model };
        let box_: Box<dyn BackendGraph> = Box::new(graph);
        Ok(box_.into())
    }

    fn as_dir_loadable(&mut self) -> Option<&mut dyn BackendFromDir> {
        Some(self)
    }
}

impl BackendFromDir for GgmlBackend {
    fn load_from_dir(
        &mut self,
        path: &Path,
        target: ExecutionTarget,
    ) -> Result<Graph, BackendError> {
        let model = LlamaModel::load_from_file(
            path.join("Phi-3-mini-4k-instruct-q4.gguf"),
            LlamaParams::default(),
        )?;
        let graph = GgmlGraph { model };
        let box_: Box<dyn BackendGraph> = Box::new(graph);
        Ok(box_.into())
    }
}

struct GgmlGraph {
    model: LlamaModel,
}

unsafe impl Send for GgmlGraph {}
unsafe impl Sync for GgmlGraph {}

struct GgmlExecutionContext {
    session: LlamaSession,
    completion: Option<CompletionHandle>,
}

impl BackendExecutionContext for GgmlExecutionContext {
    fn set_input(&mut self, id: Id, tensor: &Tensor) -> Result<(), BackendError> {
        match id.name().unwrap() {
            "prompt" => {
                let prompt = std::str::from_utf8(tensor.data.as_slice())?;
                self.session.advance_context(prompt)?;
                Ok(())
            }
            "metadata" => {
                unimplemented!("metadata is not implemented yet")
            }
            _ => Err(BackendError::UnsupportedTensorType(
                ("Only tensor ID metadata and prompt are supported by GGML backend.").into(),
            )),
        }
    }

    fn compute(&mut self) -> Result<(), BackendError> {
        self.completion = Some(self.session.start_completing()?);
        Ok(())
    }

    fn get_output(&mut self, id: Id) -> Result<Tensor, BackendError> {
        let token = self.completion.as_mut().unwrap().next_token();
        let output = self.session.model().decode_tokens(token);
        let output_bytes = output.as_bytes();
        Ok(Tensor {
            data: output_bytes.to_vec(),
            dimensions: vec![output_bytes.len() as u32],
            ty: TensorType::U8,
        })

        // let output_str = self.completion.take().unwrap().into_string();
        // let output_bytes = output_str.as_bytes();
        // Ok(Tensor {
        //     data: output_bytes.to_vec(),
        //     dimensions: vec![output_bytes.len() as u32],
        //     ty: TensorType::U8,
        // })
    }
}

unsafe impl Send for GgmlExecutionContext {}
unsafe impl Sync for GgmlExecutionContext {}

impl BackendGraph for GgmlGraph {
    fn init_execution_context(&self) -> Result<ExecutionContext, BackendError> {
        let session = self.model.create_session(SessionParams::default())?;
        let context = GgmlExecutionContext {
            session,
            completion: None,
        };
        let box_: Box<dyn BackendExecutionContext> = Box::new(context);
        Ok(box_.into())
    }
}

/// Read a file into a byte vector.
fn read(path: &Path) -> anyhow::Result<Vec<u8>> {
    let mut file = File::open(path)?;
    let mut buffer = vec![];
    file.read_to_end(&mut buffer)?;
    Ok(buffer)
}

impl From<LlamaLoadError> for BackendError {
    fn from(error: LlamaLoadError) -> Self {
        BackendError::BackendAccess(anyhow::Error::new(error))
    }
}

impl From<LlamaContextError> for BackendError {
    fn from(error: LlamaContextError) -> Self {
        BackendError::BackendAccess(anyhow::Error::new(error))
    }
}

impl From<Utf8Error> for BackendError {
    fn from(error: Utf8Error) -> Self {
        BackendError::BackendAccess(anyhow::Error::new(error))
    }
}
