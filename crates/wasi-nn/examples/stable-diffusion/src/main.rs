use core::time;
use ndarray::{Array, Dim};
use rand::Rng;
use std::fs;
use std::io::BufRead;

mod schedulers;

const ENCODER_MODEL: &str = "fixture/text_encoder/model.onnx";
const N_STEPS: usize = 15;

wit_bindgen::generate!({
    path: "../../wit",
    world: "ml",
});

use self::wasi::nn::{
    graph::{load, ExecutionTarget, Graph, GraphBuilder, GraphEncoding},
    tensor::{Tensor, TensorData, TensorDimensions, TensorType},
};

fn main() {
    let tokens = tokenize("Make a picture of green tree with flowers around it");
    let text = text_encoder(&tokens).unwrap();
    unet(text).unwrap();
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

fn u8_to_f64_vec(input: &[u8]) -> Vec<f64> {
    let chunks: Vec<&[u8]> = input.chunks(8).collect();
    let v: Vec<f64> = chunks
        .into_iter()
        .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
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

fn text_encoder(prompt: &Vec<i32>) -> Result<Tensor, crate::wasi::nn::errors::Error> {
    // Load the SD model
    let model: GraphBuilder = fs::read(ENCODER_MODEL).unwrap();
    println!("Read text encoder model, size in bytes: {}", model.len());

    let graph = load(&[model], GraphEncoding::Onnx, ExecutionTarget::Cpu).unwrap();
    let exec_context = Graph::init_execution_context(&graph).unwrap();
    let prompt_bytes = i32_to_u8_vec(&prompt);
    let prompt_dim = vec![1, prompt_bytes.len() as u32];
    let prompt_tensor = Tensor::new(&prompt_dim, TensorType::I32, &prompt_bytes);
    exec_context.set_input("input_ids", prompt_tensor).unwrap();
    exec_context.compute().unwrap();
    exec_context.get_output("last_hidden_state")
}

fn generate_latent_sample(height: usize, width: usize, init_noise_sigma: f64) -> Tensor {
    let mut rng = rand::thread_rng();
    let mut latent_sample = Array::zeros(Dim([height, width]));
    for i in 0..height {
        for j in 0..width {
            latent_sample[[i as usize, j as usize]] =
                rng.gen_range(-init_noise_sigma..init_noise_sigma);
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
        &vec![1, height as u32, width as u32],
        TensorType::Fp64,
        &latent_sample_bytes,
    )
}

fn unet(tokens: Tensor) -> Result<Tensor, crate::wasi::nn::errors::Error> {
    let width = 512;
    let height = 512;
    // Load the UNet model
    let model: GraphBuilder = fs::read("fixture/unet/model.onnx").unwrap();
    println!("Read UNet model, size in bytes: {}", model.len());

    let graph = load(&[model], GraphEncoding::Onnx, ExecutionTarget::Gpu).unwrap();
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
            &vec![2, height as u32, width as u32],
            TensorType::Fp64,
            &[latents.data(), latents.data()].concat(),
        );
        let latent_mode_input = scheduler.scale_model_input(
            u8_to_f64_vec(latent_mode_input.data().as_slice()),
            timesteps[i],
        );
        let latent_mode_tensor = Tensor::new(
            &vec![2, height as u32, width as u32],
            TensorType::Fp16,
            &f32_to_u8_vec(&latent_mode_input),
        );
        let timestep_tensor = Tensor::new(
            &vec![1],
            TensorType::Fp16,
            &f32_to_u8_vec(&[timesteps[i] as f32]),
        );
        println!("encoder_hidden_states, size: {}, dimensions: {:?}", &tokens_copied.data().len(), &tokens_copied.dimensions());
        exec_context.set_input("encoder_hidden_states", tokens_copied);
        println!("timestep, size: {}, dimensions: {:?}", &timestep_tensor.data().len(), &timestep_tensor.dimensions());
        exec_context.set_input("timestep", timestep_tensor);
        println!("sample, size: {}, dimensions: {:?}", &latent_mode_tensor.data().len(), &latent_mode_tensor.dimensions());
        exec_context.set_input("sample", latent_mode_tensor);
        exec_context.compute().unwrap();
    }
    Ok(Tensor::new(
        &vec![1, width as u32, height as u32],
        TensorType::Fp64,
        &vec![0],
    ))
}
