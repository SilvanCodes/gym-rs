use crate::GymClient;
use crate::State;
use ndarray::{concatenate, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, Axis};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StandardScaler {
    means: Array1<f64>,
    standard_deviations: Array1<f64>,
}

impl StandardScaler {
    pub fn for_environment(environment: &str) -> Self {
        Self::new(Self::generate_standard_scaler_samples(environment, 1000).view())
    }

    pub fn new(samples: ArrayView2<f64>) -> Self {
        let standard_deviations = samples
            .var_axis(Axis(0), 0.0)
            .mapv_into(|x| (x + f64::EPSILON).sqrt());

        let means = samples.mean_axis(Axis(0)).unwrap();

        StandardScaler {
            means,
            standard_deviations,
        }
    }

    pub fn scale_inplace(&self, mut sample: ArrayViewMut1<f64>) {
        sample -= &self.means;
        sample /= &self.standard_deviations;
    }

    pub fn scale(&self, sample: ArrayView1<f64>) -> Array1<f64> {
        (&sample - &self.means) / &self.standard_deviations
    }

    fn generate_standard_scaler_samples(environment: &str, num_samples: usize) -> Array2<f64> {
        let gym = GymClient::default();
        let env = gym.make(environment);

        // collect samples for standard scaler
        let samples = env
            .reset()
            .unwrap()
            .get_box()
            .expect("expected gym environment with box type observations");

        let mut samples = samples.insert_axis(Axis(0));

        println!("sampling for scaler");
        for _ in 0..num_samples {
            let State { observation, .. } = env.step(&env.action_space().sample()).unwrap();

            samples = concatenate![
                Axis(0),
                samples,
                observation.get_box().unwrap().insert_axis(Axis(0))
            ];
        }
        println!("done sampling");

        samples
    }
}
