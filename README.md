# upscale-api

> [!WARNING]  
> This project is ​**ALMOST**​ written by LLMs, so the code structure is ​​**VERY UGLY​**​.

## Usage

Download models from https://github.com/the-database/MangaJaNai/releases/tag/1.0.0.

Put model files in folder `weights`

Example:
```
.
├── README.md
├── dist
├── main.py
├── pyproject.toml
├── src
│   ├── __init__.py
│   ├── __pycache__
│   ├── api
│   ├── core
│   ├── models
│   └── utils
├── uv.lock
└── weights
    ├── 2x_IllustrationJaNai_V1_ESRGAN_120k.pth
    ├── 2x_MangaJaNai_1200p_V1_ESRGAN_70k.pth
    ├── 2x_MangaJaNai_1300p_V1_ESRGAN_75k.pth
    ├── 2x_MangaJaNai_1400p_V1_ESRGAN_70k.pth
    ├── 2x_MangaJaNai_1500p_V1_ESRGAN_90k.pth
    ├── 2x_MangaJaNai_1600p_V1_ESRGAN_90k.pth
    ├── 2x_MangaJaNai_1920p_V1_ESRGAN_70k.pth
    ├── 2x_MangaJaNai_2048p_V1_ESRGAN_95k.pth
    ├── 4x_IllustrationJaNai_V1_ESRGAN_135k.pth
    ├── 4x_MangaJaNai_1200p_V1_ESRGAN_70k.pth
    ├── 4x_MangaJaNai_1300p_V1_ESRGAN_75k.pth
    ├── 4x_MangaJaNai_1400p_V1_ESRGAN_105k.pth
    ├── 4x_MangaJaNai_1500p_V1_ESRGAN_105k.pth
    ├── 4x_MangaJaNai_1600p_V1_ESRGAN_70k.pth
    ├── 4x_MangaJaNai_1920p_V1_ESRGAN_105k.pth
    └── 4x_MangaJaNai_2048p_V1_ESRGAN_70k.pth

9 directories, 21 files
```

Then start with `uv`

```
uv run main.py
```