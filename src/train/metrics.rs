use burn::{
    tensor::{backend::Backend, Bool, Int, Tensor},
    train::metric::{Metric, MetricEntry, MetricMetadata, Numeric},
};

#[derive(Default)]
pub struct AccuracyMetric {
    total: usize,
    correct: usize,
}

impl AccuracyMetric {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<B: Backend> Metric<B> for AccuracyMetric {
    type Input = (Tensor<B, 2>, Tensor<B, 1>); // (logits, targets)

    fn update(&mut self, (output, targets): Self::Input) -> MetricEntry {
        let targets = targets.detach().to_device(&output.device());
        let predicted = output.argmax(1).detach();
        let correct = predicted.equal(targets).int().sum().into_scalar();

        self.total += targets.dims()[0];
        self.correct += correct as usize;

        let accuracy = (self.correct as f64) / (self.total as f64);
        
        MetricEntry::new(
            MetricMetadata::new("accuracy").with_format(".4"),
            Numeric::new(accuracy),
        )
    }

    fn clear(&mut self) {
        self.total = 0;
        self.correct = 0;
    }
}