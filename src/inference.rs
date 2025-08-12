use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};

use crate::model::network::MLP;

pub fn infer<B: Backend>(
    model: &MLP<B>,
    input: Tensor<B, 2>,
) -> Tensor<B, 1> {
    // Forward pass
    let output = model.forward(input);
    
    // Convertir les logits en prédictions
    output.argmax(1)
}