mod data;
mod model;
mod train;
mod inference;

use burn::backend::{ndarray::NdArray, Autodiff};
use model::config::ModelConfig;

type Backend = Autodiff<NdArray>;

fn main() {
    // Configuration
    let config = ModelConfig::default();
    let device = Default::default();
    
    // Entraînement
    train::trainer::train::<Backend>(device, config);
    
    // Pour charger un modèle et faire de l'inférence:
    // let model = MLPConfig::new(...).init();
    // let output = inference::infer(&model, input);
}