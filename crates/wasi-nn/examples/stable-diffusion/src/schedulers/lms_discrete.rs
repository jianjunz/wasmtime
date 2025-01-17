use super::super::wasi::nn::tensor::Tensor;
use super::integrate::integrate;
use super::{BetaSchedule, PredictionType};
use ndarray::linspace;

#[derive(Debug, Clone)]
pub struct LMSDiscreteSchedulerConfig {
    /// The value of beta at the beginning of training.
    pub beta_start: f32,
    /// The value of beta at the end of training.
    pub beta_end: f32,
    /// How beta evolved during training.
    pub beta_schedule: BetaSchedule,
    /// number of diffusion steps used to train the model.
    pub train_timesteps: usize,
    /// coefficient for multi-step inference.
    /// https://github.com/huggingface/diffusers/blob/9b37ed33b5fa09e594b38e4e6f7477beff3bd66a/src/diffusers/schedulers/scheduling_lms_discrete.py#L189
    pub order: usize,
    /// prediction type of the scheduler function
    pub prediction_type: PredictionType,
}

impl Default for LMSDiscreteSchedulerConfig {
    fn default() -> Self {
        Self {
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: BetaSchedule::ScaledLinear,
            train_timesteps: 1000,
            order: 4,
            prediction_type: PredictionType::Epsilon,
        }
    }
}

pub struct LMSDiscreteScheduler {
    timesteps: Vec<f32>,
    sigmas: Vec<f32>,
    init_noise_sigma: f32,
    derivatives: Vec<Vec<f32>>,
    pub config: LMSDiscreteSchedulerConfig,
}

impl LMSDiscreteScheduler {
    pub fn new(inference_steps: usize, config: LMSDiscreteSchedulerConfig) -> Self {
        let betas: Vec<f32> = match config.beta_schedule {
            BetaSchedule::ScaledLinear => ndarray::linspace(
                config.beta_start.sqrt(),
                config.beta_end.sqrt(),
                config.train_timesteps,
            )
            .map(|x| x.powi(2))
            .collect::<Vec<f32>>(),
            BetaSchedule::Linear => {
                ndarray::linspace(config.beta_start, config.beta_end, config.train_timesteps)
                    .collect::<Vec<f32>>()
            }
            _ => unimplemented!(
                "LMSDiscreteScheduler only implements linear and scaled_linear betas."
            ),
        };

        let alphas: Vec<f32> = betas.iter().map(|&x| 1. - x).collect();
        let alphas_cumprod: Vec<f32> = alphas
            .iter()
            .scan(1.0, |state, &x| {
                *state *= x;
                Some(*state)
            })
            .collect();

        let timesteps = ndarray::linspace(
            (config.train_timesteps - 1) as f32,
            0.,
            inference_steps as usize,
        )
        .collect::<Vec<f32>>();

        let sigmas = alphas_cumprod
            .iter()
            .map(|&x| ((1. - x) / x).sqrt())
            .collect::<Vec<f32>>();
        let sigmas = Self::interp(
            timesteps.as_slice(), // x-coordinates at which to evaluate the interpolated values
            ndarray::range(0., sigmas.len() as f32, 1.)
                .collect::<Vec<f32>>()
                .as_slice(),
            sigmas.as_slice(),
        );

        // standard deviation of the initial noise distribution
        let init_noise_sigma: f32 = sigmas.iter().fold(f32::MIN, |a, &b| a.max(b));

        Self {
            timesteps: timesteps.try_into().unwrap(),
            sigmas: sigmas.try_into().unwrap(),
            init_noise_sigma,
            derivatives: vec![],
            config,
        }
    }

    pub fn timesteps(&self) -> &[f32] {
        self.timesteps.as_slice()
    }

    /// Scales the denoising model input by `(sigma^2 + 1)^0.5` to match the K-LMS algorithm.
    pub fn scale_model_input(&self, sample: Vec<f32>, timestep: f32) -> Vec<f32> {
        let step_index = self.timesteps.iter().position(|&t| t == timestep).unwrap();
        let sigma = self.sigmas[step_index];

        // https://github.com/huggingface/diffusers/blob/769f0be8fb41daca9f3cbcffcfd0dbf01cc194b8/src/diffusers/schedulers/scheduling_lms_discrete.py#L132
        sample
            .iter()
            .map(|x| (x / (sigma.powi(2) + 1.)).sqrt() as f32)
            .collect::<Vec<f32>>()
    }

    /// One-dimensional linear interpolation for monotonically increasing sample
    /// points, mimicking np.interp().
    ///
    /// Based on https://github.com/pytorch/pytorch/issues/50334#issuecomment-1000917964
    fn interp(timesteps: &[f32], range: &[f32], sigmas: &[f32]) -> Vec<f32> {
        let sz = timesteps.len();
        let mut result: Vec<f32> = vec![0.; sz];

        for i in 1..sz {
            let index = range.binary_search_by(|a| a.partial_cmp(&timesteps[i]).unwrap());
            if index.is_ok() {
                result[i] = sigmas[index.unwrap()];
            } else if index.unwrap_err() <= 0 {
                result[i] = sigmas[0];
            } else if index.unwrap_err() >= range.len() {
                result[i] = sigmas[sigmas.len() - 1]
            } else {
                let index_left = index.unwrap_err() - 1;
                let index_right = index_left + 1;
                let fractional_distance =
                    (timesteps[i] - range[index_left]) / (range[index_right] - range[index_left]);
                result[i] = sigmas[index_left]
                    + fractional_distance * (sigmas[index_right] - sigmas[index_left]);
            }
        }
        result.push(0.);
        result
    }

    /// Compute a linear multistep coefficient
    fn get_lms_coefficient(&mut self, order: usize, t: usize, current_order: usize) -> f32 {
        let lms_derivative = |tau| -> f32 {
            let mut prod = 1.0;
            for k in 0..order {
                if current_order == k {
                    continue;
                }
                prod *= (tau - self.sigmas[t - k])
                    / (self.sigmas[t - current_order] - self.sigmas[t - k]);
            }
            prod
        };

        // Integrate `lms_derivative` over two consecutive timesteps.
        // Absolute tolerances and limit are taken from
        // the defaults of `scipy.integrate.quad`
        // https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html
        let integration_out =
            integrate(lms_derivative, self.sigmas[t], self.sigmas[t + 1], 1.49e-8);
        // integrated coeff
        integration_out.integral
    }

    pub fn step(&mut self, model_output: &[f32], timestep: f32, sample: &[f32]) -> Vec<f32> {
        let step_index = self.timesteps.iter().position(|&t| t == timestep).unwrap();
        let sigma = self.sigmas[step_index];

        // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        let pred_original_sample: Vec<f32> = match self.config.prediction_type {
            PredictionType::Epsilon => {
                let mut result = vec![0.; sample.len()];
                for i in 0..sample.len() {
                    result[i] = sample[i] - sigma * model_output[i]
                }
                result
            }
            PredictionType::VPrediction => {
                let mut result = vec![0.; sample.len()];
                for i in 0..sample.len() {
                    result[i] = model_output[i] * (-sigma / (sigma.powi(2) + 1.).sqrt())
                        + (sample[i] / (sigma.powi(2) + 1.))
                }
                result
            }
            _ => unimplemented!("Prediction type must be one of `epsilon` or `v_prediction`"),
        };

        // 2. Convert to an ODE derivative
        let mut derivative: Vec<f32> = vec![0.0; sample.len()];
        for i in 0..sample.len() {
            derivative[i] = (sample[i] - pred_original_sample[i]) / sigma;
        }
        self.derivatives.push(derivative);
        if self.derivatives.len() > self.config.order {
            // remove the first element
            self.derivatives.drain(0..1);
        }

        // 3. compute linear multistep coefficients
        let order = self.config.order.min(step_index + 1);
        let lms_coeffs: Vec<_> = (0..order)
            .map(|o| self.get_lms_coefficient(order, step_index, o))
            .collect();

        // 4. compute previous sample based on the derivatives path
        // https://github.com/huggingface/diffusers/blob/769f0be8fb41daca9f3cbcffcfd0dbf01cc194b8/src/diffusers/schedulers/scheduling_lms_discrete.py#L243-L245
        let deriv_prod: Vec<Vec<f32>> = lms_coeffs
            .iter()
            .zip(self.derivatives.iter().rev())
            .map(|(coeff, derivative)| {
                let mut result: Vec<f32> = vec![0.; sample.len()];
                for i in 0..derivative.len() {
                    result[i] = derivative[i] * coeff
                }
                result
            })
            .collect();
        let mut deriv_sum: Vec<f32> = vec![0.; sample.len()];
        for i in 0..deriv_prod.len() {
            for j in 0..deriv_prod[i].len() {
                deriv_sum[j] += deriv_prod[i][j]
            }
        }

        let mut result: Vec<f32> = vec![0.; sample.len()];
        for i in 0..sample.len() {
            result[i] = deriv_sum[i] + sample[i]
        }

        result
    }

    pub fn init_noise_sigma(&self) -> f32 {
        self.init_noise_sigma
    }

    pub fn add_noise(&self, original_samples: &[f32], noise: &[f32], timestep: f32) -> Vec<f32> {
        let step_index = self.timesteps.iter().position(|&t| t == timestep).unwrap();
        let sigma = self.sigmas[step_index];

        // noisy samples
        let mut samples: Vec<f32> = vec![0.; original_samples.len()];
        for i in 0..original_samples.len() {
            samples[i] = original_samples[i] + noise[i] * sigma
        }
        samples
    }
}
