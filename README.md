# Installation

## Install Python

The following command will install a required Python version in the sandbox.
Python version is defined in the `.python-version` file.

```bash
uv install python
```

Note: You may install required building tools and system libraries such as `gcc` in order to successfully build
and install Python interpreter.
Required dependencies may differ depending on the operating system and platform.
In the case of Fedora Workstation 42, the following dependencies should be enough:

```aiignore
sudo dnf install \
  gcc \
  sqlite-devel \
  ncurses-devel \
  bzip2-devel \
  libffi-devel \
  tk-devel \
  libffi-devel \
  readline-devel \
  xz-devel \
  lzma-sdk-devel \
  openssl-devel \
  zlib-ng-compat-devel
```

## Install project dependencies

The training process can be speeded up using different GPU backends, including CPU as the last resort.

Mapping of the GPU type to `uv --extra`:

| GPU Type  | `uv --extra` Value | Requirements            |
|-----------|--------------------|-------------------------|
| NVIDIA    | `--extra=cu128`    | requires installed CUDA |
| AMD       | `--extra=rocm63`   | requires installed ROCm |
| Intel Arc | `--extra=xpu`      | requires installed XPU  |
| cpu       | `--extra=cpu`      | -                       |

Afterward, install project dependencies with the selected GPU backend:

```bash
uv sync --extra=cu128
```

# Running training process

## Change settings if needed

Set required environment variables or use default values.
List of available environment variables with default values can be found in the `src/settings.py` module.

For example, to set the number of epochs to 2, run the following command:

```bash
export TRAINING_N_EPOCHS=2
```

## Start training of the model

Run the following command:

```aiignore
uv run src/train.py
```

If everything went well, you should see the confusion matrix and the plot with the training loss and accuracy.
The trained model will be saved in the `models` directory with the current timestamp in the filename suffix.
