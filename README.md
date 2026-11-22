# NukeTorch - Run Qwen 3.8 27B (4-bit) at 200.3 tok/s on Apple Silicon

200.3 tok/s: prompt A (5 coding prompts, 3 passes), NukeTorch 1.0.5 on one M5 Max, from the benchmark report below.

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

At one request at a time, NukeTorch decodes 126.3 to 327.4 tok/s on prompts A to D. With 8 users at once NukeTorch generates 345.2 tok/s in total.

**Table 1. Speed, RAM and energy**

| | NukeTorch 1.0.5 | mlx-lm<br>0.32.2 | llama.cpp<br>1.0.0 build 12359 | oMLX<br>0.8.0rc1 Splash + DFlash2 |
|---|---:|---:|---:|---:|
| **Speed (tok/s)** | | | | |
| Prompt A (5 coding prompts) | **200.3** | 31.6 | 24.8\* | 101.8 |
| Prompt B (count to 100) | **327.4** | 32.1 | 25.1\* | 184.2 |
| Prompt C (short story) | 126.3 | 31.8 | 24.8\* | **136.3** |
| Prompt D (coding-15) | **173.4** | 31.0 | 25.4\* | 106.3 |
| 2 users at once | **237.2** | 31.0 | 37.8 | 156.3 |
| 4 users at once | **321.9** | 31.1 | 63.6 | 237.2 |
| 8 users at once | **345.2** | 31.1† | N/A‡ | N/A‡ |
| **RAM (peak increase, GB)** | | | | |
| Prompt A | 22.10 | **17.74** | 42.61 | 22.56 |
| Prompt B | 22.10 | **18.27** | 42.95 | 22.70 |
| Prompt C | 22.11 | **18.73** | 43.33 | 23.14 |
| Prompt D | 22.11 | **18.85** | 43.14 | 23.73 |
| 2 users at once | 25.80 | **16.73** | 41.54 | 22.38 |
| 4 users at once | 25.82 | **17.35** | 42.68 | 23.50 |
| 8 users at once | 25.84 | **18.16**† | N/A‡ | N/A‡ |
| **Power (tok/J, tokens per joule)** | | | | |
| Prompt A | **3.24** | 1.34 | 0.62\* | 1.79 |
| Prompt B | **3.94** | 1.36 | 0.62\* | 3.17 |
| Prompt C | 1.85 | 1.36 | 0.59\* | **2.17** |
| Prompt D | **2.65** | 1.32 | 0.57\* | 1.73 |
| 2 users at once | **3.93** | 1.32 | 0.73 | 2.66 |
| 4 users at once | **5.20** | 1.32 | 1.09 | 3.90 |
| 8 users at once | **5.34** | 1.32† | N/A‡ | N/A‡ |

\* llama.cpp at its default settings (4 slots × 225,280-token context) ran out of GPU memory during prompt A of the 3-pass run; its prompts A to D come from its 1-pass run on a freshly started server. We used the maker's path and did not troubleshoot llama.cpp ourselves; if it runs fine on your M5 Max 48 GB please let us know the speed you are getting.
† mlx-lm at 8 users at once completed 24 of 32 answers; the other 8 requests failed with ConnectionResetError (the server closed the connection).
‡ N/A: not served at the engine's default settings (it serves 4 requests at once).

**How it runs.**

- **Prompts.** A: the five NukeTorch demo coding prompts, 3 passes (15 requests). B: count to 100, 3 requests. C: short story, 3 requests. D: one everyday coding prompt, 3 requests. One request at a time.
- **Users at once.** The 16 coding prompts in rounds of 2, 4 or 8 requests, 32 requests per level, 1,024 output tokens per request.
- **Models.** The mlx-community Qwen3.8-27B 4-bit for NukeTorch, mlx-lm and oMLX; llama.cpp reads only GGUF, so it runs the ggml-org Q4_K_M (4-bit). Server default sampling for every engine.
- **Reasoning effort.** The server uses effort `medium` when a request sets none.

Qwen3.8-27B 4-bit on one Apple M5 Max (18 cores, 48 GB, macOS 27.2.1). Every engine runs as its maker ships it: latest release, maker's install, maker's start command, default settings, except NukeTorch's users rows, which used `--users 8`. tok/s is decode speed; reasoning tokens count.

The full report has power per test, the method, versions and the Mac's conditions: [benchmark/qwen38_27b_four_engines_M5.pdf](benchmark/qwen38_27b_four_engines_M5.pdf).

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
