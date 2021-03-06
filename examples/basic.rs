extern crate gym;

fn main() {
    let gym = gym::GymClient::default();
    let env = gym.make("CartPole-v1");

    for _ in 0..10 {
        env.reset().expect("Unable to reset");

        for _ in 0..100 {
            let action = env.action_space().sample();
            let state = env.step(&action).unwrap();
            env.render();
            if state.is_done {
                break;
            }
        }
    }

    env.close();
}
