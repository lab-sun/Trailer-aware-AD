# Trailer-aware-AD

Official implementation of Trailer-aware-AD for the ICRA 2026 paper.

Trailer-aware-AD studies autonomous driving of articulated tractor-trailer vehicles in CARLA. The implementation combines camera observations, route waypoints, and vehicle measurements, including the tractor-trailer articulation angle, with a reinforcement learning driving policy. Trailer-aware reward terms account for the trailer's lateral position and heading relative to the route.

## Project structure

| Directory | Contents |
| --- | --- |
| `agents/` | Driving agent, perception encoders, actor and critic networks, and replay memory |
| `carla_env/` | CARLA environments, tractor-trailer actor management, observations, rewards, and evaluation criteria |
| `config/experiments/` | Agent, observation, training, and testing configurations |
| `config/envs/` | Simulation environments, traffic, weather, rewards, and termination settings |
| `run/` | Training and evaluation entry points |
| `docker/` | Container definition and Python dependencies |
| `PointPillars/` | Third-party point-cloud implementation and associated assets |
| `utilities/` | Configuration, control, logging, and visualization helpers |

## Setup

### Prerequisites

- Linux with an NVIDIA GPU, a compatible driver, Docker, and NVIDIA Container Toolkit.
- A customized CARLA server providing the `qtruck` and `qtruckbt` vehicle blueprints and the tractor-trailer behavior used by the environment. These simulator assets are not included in this repository. A standard CARLA installation alone is insufficient for the supplied trailer experiments.
- A CARLA Python client matching that customized server. The local dependency notes refer to CARLA 0.9.15; use the version required by your simulator build.

The Docker image uses PyTorch 1.12.0 with CUDA 11.3. The dependency files reflect the supplied research environment; dependency compatibility and the simulator integration must be checked on the target machine.

### Installation

```bash
git clone https://github.com/lab-sun/Trailer-aware-AD.git
cd Trailer-aware-AD
docker build -f docker/dockerfile -t trailer-aware-ad:local docker
mkdir -p results memory
docker run -it --gpus all --network host \
  -v "$PWD":/root/Trailer-aware-AD \
  -v "$PWD/results":/root/results \
  -v "$PWD/memory":/root/memory \
  trailer-aware-ad:local bash
```

Inside the container, install the CARLA Python client supplied with your customized simulator, then enter the project directory:

```bash
cd /root/Trailer-aware-AD
```

The entry point uses Weights & Biases for experiment logging. Authenticate with `wandb login`, or select offline logging before running:

```bash
export WANDB_MODE=offline
```

The supplied `trailer_offset_2` configuration uses camera images, waypoints, speed, steering, and articulation angle. Its point-cloud inputs are commented out. If enabling the optional point-cloud components, follow [PointPillars/README.md](PointPillars/README.md) and build the CUDA extensions for your environment.

## Training

1. Start the customized CARLA server.
2. Check the server address, port, traffic-manager port, and device in the experiment configuration. The supplied configuration uses `127.0.0.1:2000`, traffic-manager port `8010`, and `cuda:0`.
3. From the project directory inside the container, run:

```bash
python3 -m run.main -en trailer_offset_2
```

This experiment uses `config/experiments/trailer_offset_2/` and the trailer-aware reward configured in `config/envs/endless_full_nocrash_trailer_follower.yaml`.

Add `-vm` during training to enable the monitor window when a display is available. Resume an existing experiment with:

```bash
python3 -m run.main -en trailer_offset_2 -rt
```

Logs, saved configurations, and trained policy weights are written to `/root/results/Trailer-aware-AD/trailer_offset_2/`, which is persisted through the mounted `results` directory.

## Evaluation

After training, keep the experiment's saved weights in its results directory and run:

```bash
python3 -m run.main -en trailer_offset_2 -ttc test.yaml
```

The test environment is selected in `config/experiments/trailer_offset_2/test.yaml`. The bundled perception checkpoints are separate from the trained driving-policy weights required for evaluation.

## Acknowledgments

This project builds on the [RLfOLD framework](https://github.com/DanielCoelho112/rlfold) by Daniel Coelho, Miguel Oliveira, and Vítor Santos. We thank its authors for releasing their reinforcement learning framework for autonomous driving.

We also thank the authors of [End-to-End Urban Driving by Imitating a Reinforcement Learning Coach](https://github.com/zhejz/carla-roach) for the CARLA training environment, and [PointPillars](https://github.com/zhulf0804/PointPillars) for the point-cloud implementation included in this repository.

## License

See [LICENSE](LICENSE) for the repository's existing MIT license. Imported framework and third-party components retain their respective licenses and attribution notices; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt), and [PointPillars/LICENSE](PointPillars/LICENSE). The repository's MIT license does not replace those component-specific terms.
