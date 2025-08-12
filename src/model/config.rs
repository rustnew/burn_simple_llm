use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct ModelConfig {
    pub learning_rate: f64,
    pub epochs: usize,
    pub batch_size: usize,
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub dropout_rate: f64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            epochs: 10,
            batch_size: 64,
            input_size: 784,  // Pour MNIST 28x28
            hidden_size: 512,
            output_size: 10,  // 10 classes pour MNIST
            dropout_rate: 0.2,
        }
    }
}