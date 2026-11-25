# Running Neuralingo with Docker

Neuralingo translates Japanese manga pages into English: it finds the speech bubbles,
erases the Japanese text, reads it, translates it with an AI service, and draws the
English in. With Docker you don't need to install Python or anything else.

## What you need

- **Docker**: [Docker Desktop](https://www.docker.com/products/docker-desktop/) on Windows
  or macOS, or Docker Engine on Linux.
- **A translation API key**, either one works:
  - Google Gemini: free tier available, key from <https://aistudio.google.com/apikey>.
  - Groq: free tier available, key from <https://console.groq.com/keys>.
- **Optional, much faster:** an NVIDIA graphics card with a recent driver (CUDA 13 support,
  driver 580 or newer).
  - On Linux, also install the
    [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  - Docker Desktop on Windows supports NVIDIA cards through WSL 2.
  - Without one it runs on the CPU: roughly 20–40 seconds per page instead of a few.

## Getting the image

You'll either get a file from whoever shares it with you, or build it yourself from
the project folder.

**From a file** (e.g. `neuralingo.tar.gz`):

```
docker load -i neuralingo.tar.gz
```

**Building it yourself:**

```
docker build -t neuralingo .                                     # NVIDIA GPU (also runs on CPU)
docker build -t neuralingo:cpu --build-arg TORCH_VARIANT=cpu .   # no NVIDIA GPU: smaller download
```

## Sharing the image (for whoever builds it)

Save the image to a file, send it (it's large: use a file-sharing service), and the
receiver loads it as above:

```
docker save neuralingo | gzip > neuralingo.tar.gz          # GPU image (12.6 GB unpacked)
docker save neuralingo:cpu | gzip > neuralingo-cpu.tar.gz  # CPU image (6 GB unpacked)
```

Friends without an NVIDIA card only need the CPU image. The image contains the font in
`assets/font` (a commercial font): fine to hand to a few friends, not to publish.

## Translating pages

1. Make a folder `pages` and put the pages in it (`.png`, `.jpg`, `.jpeg` or `.webp`).
   They're processed in natural order (2 before 10).
2. Make an empty folder `translated` next to it.
3. From the folder containing both, run the command for your system. Replace
   `your-key` with your key, use `GROQ_API_KEY` instead for a Groq key, and change
   `neuralingo` to `neuralingo:cpu` if that's the image you have.

**Linux / macOS:**

```
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -e GEMINI_API_KEY=your-key -e NEURALINGO_TRANSLATOR=gemini \
  -v "$PWD/pages:/app/out:ro" -v "$PWD/translated:/app/output" \
  neuralingo
```

**Windows (PowerShell):**

```
docker run --rm --gpus all `
  -e GEMINI_API_KEY=your-key -e NEURALINGO_TRANSLATOR=gemini `
  -v "${PWD}/pages:/app/out:ro" -v "${PWD}/translated:/app/output" `
  neuralingo
```

Notes on the command:
- **No NVIDIA card:** leave out `--gpus all`. If Docker reports
  `could not select device driver`, that's the fix.
- **macOS:** Docker can't use the Mac's GPU, so leave out `--gpus all` there too.
- **When it's done**, the translated pages are in `translated/` under their original
  names, as PNG (`page1.jpg` becomes `page1.png`). The container stops by itself and
  prints `DONE ALL BATCH`.
- **Debug images:** `translated/` also gets some files showing what was detected and
  erased (`visualized_detection.png`, `image_mask.png`, `image_cleaned.png`). They're
  overwritten on every run.

## Settings

The image uses default settings. To change them (detection thresholds, the colour
sensitivity for multi-speaker captions, the inpainting model):
1. Copy `config.example.yaml` to `config.yaml` and edit it.
2. Add this to the `docker run` command:

```
-v "$PWD/config.yaml:/app/config.yaml:ro"
```

Keys can go in that file instead of `-e ...`. Keep it to yourself: it holds your keys.

| Environment variable | Meaning |
|---|---|
| `GEMINI_API_KEY`, `GROQ_API_KEY`, `DEEPSEEK_API_KEY` | Translation keys; any you give are used, as fallbacks to each other. |
| `NEURALINGO_TRANSLATOR` | Which one to try first: `gemini`, `groq` or `deepseek`. |

## Troubleshooting

- **`No images in out/`:** the `pages` folder is empty, or the `-v` path is wrong.
  Docker needs full paths, which is why the commands use `$PWD`. Run the command
  from the folder that contains `pages`.
- **Untranslated (Japanese) text in the output:** no key was given, or the key was
  rejected. Look for `No API keys provided` or an API error in the printed log.
- **Out of GPU memory:** the run continues with a slower fallback for the erasing step.
  Close other programs using the GPU, or use the CPU image.
- **Permission errors writing `translated/` (Linux):** keep the
  `--user "$(id -u):$(id -g)"` part.
