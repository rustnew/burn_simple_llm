use burn::{
    nn,
    tensor::{backend::Backend, Tensor},
};

use burn::prelude::{Config, Module};

#[derive(Config)]
pub struct MLPConfig {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    dropout: f64,
}

impl MLPConfig {
    pub fn init<B: Backend>(&self) -> MLP<B> {
        MLP {
            linear1: nn::LinearConfig::new(self.input_size, self.hidden_size).init(),
            linear2: nn::LinearConfig::new(self.hidden_size, self.output_size).init(),
            dropout: nn::DropoutConfig::new(self.dropout).init(),
            activation: nn::ReLU::new(),
        }
    }
}

#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    linear1: nn::Linear<B>,
    linear2: nn::Linear<B>,
    dropout: nn::Dropout,
    activation: nn::ReLU,
}

impl<B: Backend> MLP<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(input);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);
        self.linear2.forward(x)
    }
    
    pub fn forward_training(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward(input)
    }
}