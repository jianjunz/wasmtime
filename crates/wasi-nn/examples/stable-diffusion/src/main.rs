use core::time;
use ndarray::{Array, Dim};
use rand::Rng;
use std::fs;
use std::io::BufRead;
use wasi::nn::graph::load_by_name;

mod schedulers;

const ENCODER_MODEL: &str = "E:\\Code\\stable-diffusion-v1-4\\text_encoder\\model.onnx";
const UNET_MODEL: &str = "E:\\Code\\stable-diffusion-v1-4\\unet\\model.onnx";
const DECODER_MODEL: &str = "E:\\Code\\stable-diffusion-v1-4\\vae_decoder\\model.onnx";
const N_STEPS: usize = 15;
const BLANK_TOKEN_VALUE: i32 = 49407;
const MODEL_MAX_LENGTH: usize = 77;

wit_bindgen::generate!({
    path: "../../wit",
    world: "ml",
});

use self::wasi::nn::{
    graph::{load, ExecutionTarget, Graph, GraphBuilder, GraphEncoding},
    tensor::{Tensor, TensorData, TensorDimensions, TensorType},
};

fn main() {
    let mut tokens = tokenize("a fireplace in an old cabin in the woods");
    let text = text_embeddings(&mut tokens).unwrap();
    println!("After text encoder, text dim: {:?}", text.dimensions());
    unet(text).unwrap();
    println!("After unet");
}

fn i32_to_u8_vec(input: &[i32]) -> Vec<u8> {
    let mut output = Vec::with_capacity(input.len() * 4);
    for &value in input {
        output.extend_from_slice(&value.to_ne_bytes());
    }
    output
}

fn f32_to_u8_vec(input: &[f32]) -> Vec<u8> {
    let mut output = Vec::with_capacity(input.len() * 4);
    for &value in input {
        output.extend_from_slice(&value.to_ne_bytes());
    }
    output
}

fn i64_to_u8_vec(input: &[i64]) -> Vec<u8> {
    let mut output = Vec::with_capacity(input.len() * 8);
    for &value in input {
        output.extend_from_slice(&value.to_ne_bytes());
    }
    output
}

fn u8_to_f64_vec(input: &[u8]) -> Vec<f64> {
    let chunks: Vec<&[u8]> = input.chunks(8).collect();
    let v: Vec<f64> = chunks
        .into_iter()
        .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
        .collect();

    v.into_iter().collect()
}

fn u8_to_f32_vec(input: &[u8]) -> Vec<f32> {
    let chunks: Vec<&[u8]> = input.chunks(4).collect();
    let v: Vec<f32> = chunks
        .into_iter()
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect();

    v.into_iter().collect()
}

fn tokenize(prompt: &str) -> Vec<i32> {
    use instant_clip_tokenizer::{Token, Tokenizer};

    let tokenizer = Tokenizer::new();
    let mut tokens = Vec::new();
    tokenizer.encode(prompt, &mut tokens);
    let tokens = tokens
        .into_iter()
        .map(Token::to_u16)
        .map(|x| x as i32)
        .collect::<Vec<i32>>();
    tokens
}

fn text_encoder(prompt: &Vec<i32>) -> Result<Vec<u8>, crate::wasi::nn::errors::Error> {
    // Load the SD model
    // let model: GraphBuilder = fs::read(ENCODER_MODEL).unwrap();
    // println!("Read text encoder model, size in bytes: {}", model.len());

    let graph = load(
        &[ENCODER_MODEL.as_bytes().to_vec()],
        GraphEncoding::Onnx,
        ExecutionTarget::Gpu,
    )
    .unwrap();
    let exec_context = Graph::init_execution_context(&graph).unwrap();
    let prompt_bytes = i32_to_u8_vec(&prompt);
    let prompt_dim = vec![1, prompt.len() as u32];
    let prompt_tensor = Tensor::new(&prompt_dim, TensorType::I32, &prompt_bytes);
    exec_context.set_input("input_ids", prompt_tensor).unwrap();
    exec_context.compute().unwrap();
    let output_tensor = exec_context.get_output("last_hidden_state").unwrap();
    println!("Output tensor size: {}", output_tensor.data().len());
    Ok(output_tensor.data())
}

fn text_embeddings(prompt: &mut Vec<i32>) -> Result<Tensor, crate::wasi::nn::errors::Error> {
    prompt.extend(vec![BLANK_TOKEN_VALUE; MODEL_MAX_LENGTH - prompt.len()]);
    let mut prompt_encoded = u8_to_f32_vec(&(text_encoder(prompt).unwrap()).as_slice());
    let prompt_embeddings_data = f32_to_u8_vec(prompt_encoded.as_slice());

    let default_prompt = vec![BLANK_TOKEN_VALUE; MODEL_MAX_LENGTH];
    let default_token = u8_to_f32_vec(&(text_encoder(&default_prompt).unwrap()).as_slice());
    let default_embeddings_data = f32_to_u8_vec(default_token.as_slice());

    let embeddings_dim = vec![2 as u32, MODEL_MAX_LENGTH as u32, 768 as u32];
    let embeddings_data = vec![prompt_embeddings_data, default_embeddings_data].concat();
    let embeddings = Tensor::new(&embeddings_dim, TensorType::Fp32, &embeddings_data);
    Ok(embeddings)
}

fn generate_latent_sample(height: usize, width: usize, init_noise_sigma: f64) -> Tensor {
    let mut rng = rand::thread_rng();
    let batch_size = 1;
    let channels = 4;
    let mut latent_sample = Array::zeros(Dim([batch_size, channels, height / 8, width / 8]));
    for b in 0..batch_size {
        for c in 0..channels {
            for h in 0..height / 8 {
                for w in 0..width / 8 {
                    latent_sample[[b as usize, c as usize, h as usize, w as usize]] =
                        rng.gen_range(-init_noise_sigma..init_noise_sigma);
                }
            }
        }
    }
    let latent_sample = latent_sample
        .iter()
        .map(|x| *x as f64)
        .collect::<Vec<f64>>();
    let mut latent_sample_bytes: Vec<u8> = vec![0; latent_sample.len() * 8];
    for i in 0..latent_sample.len() {
        let bytes = latent_sample[i].to_ne_bytes();
        latent_sample_bytes[i * 8..(i + 1) * 8].copy_from_slice(&bytes);
    }
    Tensor::new(
        &vec![
            batch_size as u32,
            channels as u32,
            (height / 8) as u32,
            (width / 8) as u32,
        ],
        TensorType::Fp64,
        &latent_sample_bytes,
    )
}

fn unet(tokens: Tensor) -> Result<Tensor, crate::wasi::nn::errors::Error> {
    let width = 512;
    let height = 512;
    // Load the UNet model
    // let model: GraphBuilder = fs::read(UNET_MODEL).unwrap();
    // println!("Read UNet model, size in bytes: {}", model.len());

    let graph = load(
        &[UNET_MODEL.as_bytes().to_vec()],
        GraphEncoding::Onnx,
        ExecutionTarget::Gpu,
    )
    .unwrap();
    let exec_context = Graph::init_execution_context(&graph).unwrap();

    let scheduler_config = schedulers::lms_discrete::LMSDiscreteSchedulerConfig {
        beta_start: 0.00085,
        beta_end: 0.012,
        beta_schedule: schedulers::BetaSchedule::ScaledLinear,
        prediction_type: schedulers::PredictionType::Epsilon,
        order: 4,
        train_timesteps: 1000,
    };
    let scheduler = schedulers::lms_discrete::LMSDiscreteScheduler::new(N_STEPS, scheduler_config);
    let timesteps = scheduler.timesteps();
    let latents = generate_latent_sample(width, height, scheduler.init_noise_sigma());

    for i in 0..timesteps.len() {
        let tokens_copied = Tensor::new(&tokens.dimensions(), tokens.ty(), &tokens.data());
        let latent_mode_input = Tensor::new(
            &vec![1, 4, (height / 8) as u32, (width / 8) as u32],
            TensorType::Fp64,
            &[latents.data(), latents.data()].concat(),
        );
        let latent_mode_input = scheduler.scale_model_input(
            u8_to_f64_vec(latent_mode_input.data().as_slice()),
            timesteps[i],
        );
        let latent_mode_tensor = Tensor::new(
            &vec![2, 4, (height / 8) as u32, (width / 8) as u32],
            TensorType::Fp32,
            &f32_to_u8_vec(&latent_mode_input),
        );
        let timestep_tensor = Tensor::new(
            &vec![1],
            TensorType::I64,
            &i64_to_u8_vec(&[timesteps[i] as i64]),
        );
        println!(
            "encoder_hidden_states, size: {}, dimensions: {:?}",
            &tokens_copied.data().len(),
            &tokens_copied.dimensions()
        );
        exec_context.set_input("encoder_hidden_states", tokens_copied);
        println!(
            "timestep, size: {}, dimensions: {:?}",
            &timestep_tensor.data().len(),
            &timestep_tensor.dimensions()
        );
        exec_context.set_input("timestep", timestep_tensor);
        println!(
            "sample, size: {}, dimensions: {:?}",
            &latent_mode_tensor.data().len(),
            &latent_mode_tensor.dimensions()
        );
        exec_context.set_input("sample", latent_mode_tensor);
        println!("Executing UNet model for timestep[{}] {}", i, timesteps[i]);
        exec_context.compute().unwrap();
        let output_tensor = exec_context.get_output("latent_sample").unwrap();
        let (noise_pred, noise_pred_text) = split_tensor(output_tensor);
    }
    Ok(Tensor::new(
        &vec![1, width as u32, height as u32],
        TensorType::Fp64,
        &vec![0],
    ))
}

fn split_tensor(input: Tensor) -> (Tensor, Tensor) {
    let mut input_data = u8_to_f32_vec(input.data().as_slice());
    let mut input_dim = input.dimensions();
    let mut input_type = input.ty();
    let mut split_dim = input_dim.clone();
    split_dim[1] = input_dim[0] / 2;
    let mut tensor1_data = vec![0 as f32; input_data.len() / 2];
    let mut tensor2_data = vec![0 as f32; input_data.len() / 2];
    let mut tensor1 = Tensor::new(
        &split_dim,
        input_type,
        &f32_to_u8_vec(tensor1_data.as_slice()),
    );
    let mut tensor2 = Tensor::new(
        &split_dim,
        input_type,
        &f32_to_u8_vec(tensor2_data.as_slice()),
    );
    (tensor1, tensor2)
}

fn decode(input: Tensor) -> String {
    let graph = load(
        &[DECODER_MODEL.as_bytes().to_vec()],
        GraphEncoding::Onnx,
        ExecutionTarget::Gpu,
    )
    .unwrap();
    let exec_context = Graph::init_execution_context(&graph).unwrap();
    exec_context.set_input("latent_sample", input);
    exec_context.compute().unwrap();
    let output = exec_context.get_output("sample");
}
