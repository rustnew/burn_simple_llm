use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    tensor::{backend::Backend, Data, Tensor},
};

pub struct MnistBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> MnistBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct MnistItem {
    pub image: Vec<f32>,
    pub label: usize,
}

impl<B: Backend> Batcher<MnistItem, MnistBatch<B>> for MnistBatcher<B> {
    fn batch(&self, items: Vec<MnistItem>) -> MnistBatch<B> {
        let images = items
            .iter()
            .map(|item| Data::<f32, 2>::from(item.image.clone()))
            .map(|data| Tensor::<B, 2>::from_data(data.convert(), &self.device))
            .collect();

        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1>::from_data(Data::from([item.label as i64]), &self.device))
            .collect();

        MnistBatch { images, targets }
    }
}

#[derive(Debug)]
pub struct MnistBatch<B: Backend> {
    pub images: Vec<Tensor<B, 2>>,
    pub targets: Vec<Tensor<B, 1>>,
}