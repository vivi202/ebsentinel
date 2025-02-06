use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::prelude::*;

/// Encoder module
#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    linear: Linear<B>,
    activation: Relu,
}

impl<B: Backend> Encoder<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear.forward(input);
        self.activation.forward(x)
    }
}

/// Middle latent representation
#[derive(Module, Debug)]
pub struct Middle<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> Middle<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(input)
    }
}

/// Decoder module
#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    linear1: Linear<B>,
    activation: Relu,
    linear2: Linear<B>,
}

impl<B: Backend> Decoder<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(input);
        let x = self.activation.forward(x);
        self.linear2.forward(x) // Linear output layer
    }
}

/// Full Autoencoder Model
#[derive(Module, Debug)]
pub struct Autoencoder<B: Backend> {
    encoder: Encoder<B>,
    middle: Middle<B>,
    decoder: Decoder<B>,
}

impl<B: Backend> Autoencoder<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let encoded = self.encoder.forward(input);
        let latent = self.middle.forward(encoded);
        self.decoder.forward(latent)
    }
}

/// Autoencoder Configuration
#[derive(Config, Debug)]
pub struct AutoencoderConfig {
    input_size: usize,
    latent_size: usize,
}

impl AutoencoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Autoencoder<B> {
        Autoencoder {
            encoder: Encoder {
                linear: LinearConfig::new(self.input_size, self.input_size).init(device),
                activation: Relu::new(),
            },
            middle: Middle {
                linear: LinearConfig::new(self.input_size, self.latent_size).init(device),
            },
            decoder: Decoder {
                linear1: LinearConfig::new(self.latent_size, self.input_size).init(device),
                activation: Relu::new(),
                linear2: LinearConfig::new(self.input_size, self.input_size).init(device),
            },
        }
    }
}
