# Simoji robot brain

Run a real robot foundation model on a cloud GPU and drive a robot in the **Simoji** iPhone app,
with no code and no computer.

1. In Simoji, tap **Try a robot brain** (or open the Brev Launchable link).
2. Log in to [NVIDIA Brev](https://brev.nvidia.com) and tap **Deploy**. Brev bills your account by the hour.
3. When the ready page shows **Open in Simoji**, tap it, or type the 6-digit code in Simoji.

The GPU powers itself off 30 minutes after your phone disconnects, or straight away with **Stop now**.

## What runs on the GPU

| Part | What it is | Licence |
|---|---|---|
| [NVIDIA Isaac GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T) (`nvidia/GR00T-N1.7-DROID`) | the robot model (vision + language → arm actions), trained on DROID (Franka arm) | see NVIDIA's model card |
| [nvidia/Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) | GR00T N1.7's vision-language backbone. Gated upstream; downloaded from Simoji's unmodified copy so you need no Hugging Face token | NVIDIA Open Model License |
| `simoji-policy/` | the connector: pairs with the Simoji relay and streams observations / actions | MIT |
| `bridges/droid_policy.py` | maps Simoji's Franka Panda to the DROID inputs and outputs | MIT |
| `launchable/` | Brev setup, ready page, idle power-off | MIT |

**Built on NVIDIA Cosmos.** Cosmos-Reason2-2B is licensed by NVIDIA Corporation under the
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
Simoji is not affiliated with or endorsed by NVIDIA.

## Run it yourself

On any Ubuntu machine with an NVIDIA GPU (16 GB+):

```bash
curl -fsSL https://raw.githubusercontent.com/robotsimguy/simoji-brain/main/launchable/setup.sh | bash
```

Then open `http://<machine>:8080`. Settings (environment variables): `SIMOJI_MODEL`,
`SIMOJI_EMBODIMENT`, `SIMOJI_IDLE_MINUTES` (0 = never power off).

Your own model instead: see [`simoji-policy/README.md`](simoji-policy/README.md) and the protocol in
[`docs/remote-policy-protocol.md`](docs/remote-policy-protocol.md).

## Privacy

The phone sends the robot's joint positions, the robot camera images and your instruction to *your* GPU
(through the Simoji relay, which only forwards them and keeps none of it). Nothing is sent to Simoji otherwise.

Help: [simoji.ai/support](https://www.simoji.ai/support) · mustafa@simoji.ai
