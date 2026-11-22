# NukeTorch - Run Qwen 3.8 27B (4-bit) at 200.1 tok/s on Apple Silicon

200.1 tok/s: five demo prompts, 3 passes, NukeTorch 1.0.5 on one M5 Max, from the benchmark report below.

C++ and Metal Tensor Engine serves text and image for inference and training.

**[Detailed benchmark report](benchmark/qwen38_27b_four_engines_M5.pdf)**: NukeTorch, mlx-lm, llama.cpp and oMLX on one M5 Max.

### Quickstart

Install with `curl`, not a browser.

```sh
curl -fsSL https://github.com/NukeTorch/nuketorch/releases/download/1.0.5/install.sh | sh
```

Get the model and its draft model from Hugging Face ([model card](https://huggingface.co/NukeTorch/Qwen3.8-27B-4bit-DFlash2)):

The first line gives you the `hf` command (or: `brew install hf`).

```sh
python3 -m venv ~/NukeTorch/hf-env && ~/NukeTorch/hf-env/bin/pip install -U huggingface_hub
~/NukeTorch/hf-env/bin/hf download NukeTorch/Qwen3.8-27B-4bit-DFlash2 --local-dir ~/NukeTorch/models/Qwen3.8-27B-4bit-DFlash2
```

Run it:

```sh
cd ~/NukeTorch/nuketorch-1.0.5-macos-arm64
./nuketorch-serve http \
  --model ~/NukeTorch/models/Qwen3.8-27B-4bit-DFlash2/Qwen3.8-27B-4bit \
  --draft-model ~/NukeTorch/models/Qwen3.8-27B-4bit-DFlash2/Qwen3.8-27B-DFlash2 \
  --port 8080
```

The server prints `READY` when it can answer. Then:

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write a Python function that checks whether a string is a palindrome."}]}'
```

### Benchmark Summary

At one request at a time, NukeTorch decodes 164.6 to 205.0 tok/s on the four tests below.

**Table 1. Speed and memory**

| | NukeTorch |
|---|---:|
| **Speed (tok/s), one user** | |
| 16 coding prompts (NVIDIA SPEED-Bench), 2 seeds, 1,024-token cap | **205.0** |
| Five demo prompts, each as a new request, 3 passes, no token cap | **198.6** |
| Five demo prompts as one conversation, 1,024-token cap, 3 passes | **180.7** |
| 16 coding prompts of our own, 2 seeds, 1,024-token cap | **164.6** |
| **First token (median), new requests** | 0.14 to 0.28 s |
| **Memory at the default context** (the server's `READY` line) | 26.7 GB |

**How it runs.**

- **Tests.** The benchmark built into NukeTorch (`nuketorch-serve bench`).
- **Run-to-run spread.** The answers are the same every run; the speed is not. Let the Mac rest before you measure. The median of our runs is stored beside the best and shown in every report.
- **The prompt matters.** Speed follows how many drafted tokens are accepted: 205.0 tok/s on NVIDIA's coding prompts and 164.6 on ours; single answers ranged up to 247.9 tok/s.
- **Long conversations start slower.** Each turn reads the whole conversation again before it answers: in the conversation test the first token took 2.8 s (median).
- **Reasoning effort matters.** The server uses effort `medium` when a request sets none. At the model's own default, `xhigh`, an earlier build gave 155.7 tok/s on the SPEED-Bench coding prompts.

Qwen3.8-27B 4-bit on one Apple M5 Max (48 GB), one user, with its DFlash2 draft model, typical acceptance, reasoning on. Measured by the built-in benchmark: the SPEED-Bench coding row on NukeTorch 1.0.5, the other rows on an earlier release. tok/s is decode speed: per answer, tokens divided by the time from the first token to the last; a test's number is all its tokens divided by those times added up. Reasoning tokens count. The benchmark report compares NukeTorch with mlx-lm, llama.cpp and oMLX on the same Mac.

NukeTorch Enterprise runs one model across machines by pipeline parallelism. It is by invitation. Ask for an evaluation license:

[![Request a NukeTorch Enterprise evaluation license](https://img.shields.io/badge/NukeTorch%20Enterprise-Request%20an%20evaluation%20license-121F47?style=for-the-badge)](mailto:contact@nuketorch.ai?subject=NukeTorch%20Enterprise%20evaluation%20request&body=Name%3A%0AOrganization%3A%0AMacs%20%28how%20many%2C%20which%20chips%2C%20memory%29%3A%0AModel%20or%20workload%3A%0A)

### How we did it

**Unfolding.** A tensor's values are fixed; how they are stored and read is a choice, and changing that choice is unfolding. On one Apple M4 Max in our own runtime, expanding a four-bit model's weights to 16 bits used 2.4× the memory for a 1.14× speedup, while unfolding the same codes in the same bytes gave 1.27×. A control showed that most of that gain came from the form of the kernel's loop, not the order, and that the weight bytes a computation reads set a floor that no order lowers; three large models already decode at 81 to 90% of it in that runtime. Paper: [Unfolding: Theory and Principles](https://nuketorch.ai/papers/unfolding-theory-and-principles.pdf).

**Shaping.** A matrix product has three axes (rows, output width, reduction length) and an order in time, yet devices are mostly judged by one-axis indicators such as busy time. The paper measured the axes one at a time on Apple's Neural Engine, on an M4 Max and an M5 Max. Splitting a long reduction length into slices raised a long-reduction layer pair 3.40× and 3.12× there. Paper: [Understanding the Shapes of Neural Simulation](https://nuketorch.ai/papers/understanding-the-shapes-of-neural-simulation.pdf).

The papers report their own measurements. They do not describe NukeTorch's code and do not explain the speeds in the table above.

### Minimum hardware requirement

We recommend an M5 Max with 48 GB of RAM. We optimized for the Neural Accelerator of Apple silicon. This release runs on the M5 Max and the M4 Max; support for M3 series chips is coming. On an M4 Max with 128 GB it gave 53.2 tok/s on the SPEED-Bench coding prompts and 49.8 on the demo prompts. The model needs about 27 GB of free memory. It needs macOS 27.2 or later.

### More about running it

- `GET /v1/models` shows the model name your client should use.
- Several users: add `--users N`. The default is 1, which is the fastest for one person. Each extra user costs about 1.8 GB (32.3 GB with `--users 4`). On 48 GB, use up to 4 users: on an earlier build 8 users took 47 GB and slowed to a crawl. A server started for several users is slower for one user than one started with `--users 1`.
- Images: send them inline as an `image_url` part holding a `data:` address (PNG, JPEG or WebP), 1 to 8 per request. Images given by web address are refused. First token on the M5 Max with the benchmark's three drawn test images: 448×336, 0.42 s; 1024×1024, 2.2 s; 2000×1293, 6.2 s.
- Streaming and function tools (`tool_choice` `auto` or `none`) are supported. Forced tool choice, strict schemas and `response_format` are not.
- Options: `--host`, `--port`, `--max-context` (default 32768), `--max-memory GB`, `--prefix-cache-gb`, `--users`.

### Reproduce our numbers

`bench` runs the model's standard suite; `report` shows your run beside ours.

```sh
cd ~/NukeTorch/nuketorch-1.0.5-macos-arm64
./nuketorch-serve bench ~/NukeTorch/models/Qwen3.8-27B-4bit-DFlash2
./nuketorch-serve report
```

The SPEED-Bench coding prompts are NVIDIA's and are not in the package. This command downloads NVIDIA's own file from Hugging Face (`nvidia/SPEED-Bench`) at a pinned revision, checks it against hashes stored in the program, then runs that suite:

```sh
./nuketorch-serve bench --fetch speed-bench
./nuketorch-serve bench ~/NukeTorch/models/Qwen3.8-27B-4bit-DFlash2 --suite speed-bench
```

Other commands: `bench --list` (what each test does), `bench --inputs` (every stored input with its license and source), `bench MODEL --suite quick` (a check in minutes; no golden to compare), `selftest` (the model kernels, about 1 second, no model needed).

Before you measure: turn off Spotlight indexing (`sudo mdutil -a -i off`; turn it back on with `sudo mdutil -a -i on`), turn off Siri, on a MacBook Pro plug in power and choose High Power, close other apps, and let the Mac cool down: rest it 5 minutes before a run and 5 minutes between tests run back to back.

### License

NukeTorch is free for personal use, schools and universities, and businesses and other organizations (not government entities or international organizations) that with their affiliates have less than US$10 million in revenue and fewer than 250 people, on up to 250 machines, each working alone. Everyone else may evaluate it for 30 days, then needs a commercial license: contact@nuketorch.ai. You may not offer its function itself to others: access to its API, or running, serving or fine-tuning models for them. Terms are in [LICENSE.md](https://github.com/NukeTorch/nuketorch/blob/1.0.5/LICENSE.md). The model and its draft model are Apache-2.0, by their makers.

**Powered by NukeTorch** · https://nuketorch.ai · Back to [NukeTorch/nuketorch](https://github.com/NukeTorch/nuketorch)
