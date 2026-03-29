# Gel Image Tool

Desktop tool for processing gel electrophoresis images. Load a raw photo of a gel, auto-crop to the gel region, and choose between three output modes:

- **Green Channel** — extracted and cropped
- **Green + Band Boost** — background-subtracted with amplified band signal (adjustable gain)
- **Residual (Detrended)** — column-detrended, horizontally smoothed residual for maximum band contrast

## Install & Run

```bash
cd gel-image-tool
uv sync
uv run gel-tool
```

## Reference

The processing pipeline was prototyped in `reference/gel_conversion.ipynb`.
