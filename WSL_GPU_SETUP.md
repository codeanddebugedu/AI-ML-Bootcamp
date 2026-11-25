# TensorFlow GPU Setup on Windows (WSL2)

TensorFlow 2.11 and later can't use the GPU on native Windows, so we run it inside **WSL2 (Ubuntu)**.
Your project files stay on Windows. Only the Python environment lives inside Ubuntu.

---

## 1. First-time setup (once per machine)

The NVIDIA driver is installed on **Windows only**. Do not install a driver inside Ubuntu.

Open **Ubuntu** from the Start menu, then run:

```bash
# Check the GPU is visible inside WSL
nvidia-smi

# Install Python tools (asks for your Ubuntu password)
sudo apt update && sudo apt install -y python3-venv python3-pip
```

---

## 2. Create an environment for a project

### Option A: Manual commands

```bash
python3 -m venv ~/tf
source ~/tf/bin/activate
pip install --upgrade pip

cd "/mnt/c/Users/codeanddebug/Desktop/AI-ML-Bootcamp/simple rnn"
pip install -r requirements.txt
pip install "tensorflow[and-cuda]==2.21.0"
```

The last command must download about 12 `nvidia-*` packages, such as `nvidia-cudnn-cu12` and `nvidia-cublas-cu12`, a few GB in total.
Use the same TensorFlow version that is in your `requirements.txt`.

### Option B: Reusable script (for any project)

Save this once as `~/setup_gpu_env.sh`. Run `nano ~/setup_gpu_env.sh`, paste the script, then press Ctrl+O, Enter, Ctrl+X.

```bash
#!/usr/bin/env bash
# Usage: bash ~/setup_gpu_env.sh <env-name> [path/to/requirements.txt]
set -e
[ -z "$1" ] && { echo "Usage: bash ~/setup_gpu_env.sh <env-name> [requirements.txt]"; exit 1; }

ENV_DIR="$HOME/envs/$1"
python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
pip install --upgrade pip

if [ -n "$2" ]; then
  pip install -r "$2"
fi

# Install CUDA libs matching the TensorFlow version (keeps a pinned version if requirements.txt had one)
TFV=$(pip show tensorflow 2>/dev/null | awk '/^Version:/{print $2}')
if [ -n "$TFV" ]; then
  pip install "tensorflow[and-cuda]==$TFV"
else
  pip install "tensorflow[and-cuda]" ipykernel
fi

python -c "import tensorflow as tf; print('TF', tf.__version__, '| GPUs:', tf.config.list_physical_devices('GPU'))"
echo "Done. In VS Code select kernel: $ENV_DIR/bin/python"
```

Use it for a new project:

```bash
cd "/mnt/c/Users/codeanddebug/Desktop/AI-ML-Bootcamp/<your project folder>"
bash ~/setup_gpu_env.sh myproject requirements.txt
```

If the project has no `requirements.txt`, leave that part off the command.

> **Tip:** If the new project also uses TensorFlow, you can reuse `~/tf` instead of creating a new environment.
> Install anything extra with `source ~/tf/bin/activate`, then `pip install <package>`.

---

## 3. Check the GPU works

```bash
source ~/tf/bin/activate      # or: source ~/envs/myproject/bin/activate
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

You should see something with `GPU:0` in it. Lines starting with `I0000` or mentioning `oneDNN` are normal.

**If it prints `[]`** and says *"Could not find cuda drivers"* or *"Error loading CUDA libraries"*, the CUDA packages are missing.
Activate the environment and run `pip install "tensorflow[and-cuda]==<your TF version>"` again.
You can check what is installed with `pip list | grep nvidia`.

---

## 4. Run a notebook in VS Code

1. In Ubuntu, go to the project folder and run:
   ```bash
   code .
   ```
2. VS Code opens connected to WSL; the bottom-left corner shows **WSL: Ubuntu**.
   Install the **Python** and **Jupyter** extensions if it asks (they install inside WSL).
3. Open the `.ipynb` file, click **Select Kernel**, then **Python Environments**, and choose
   `~/tf/bin/python` (or `~/envs/<name>/bin/python`).
4. Run the cells. Training now uses the GPU.

Each time you open a new Ubuntu terminal, run `source ~/tf/bin/activate` before using `python` or `pip`.

---

## 5. Delete everything (WSL only)

These commands only delete things inside WSL. Your project files on Windows (`/mnt/c/...`) are **not** touched.

```bash
# Delete one environment and all its packages
rm -rf ~/tf
rm -rf ~/envs/myproject

# Free the pip download cache (a few GB of CUDA wheels)
rm -rf ~/.cache/pip

# Remove pip and venv from Ubuntu
sudo apt remove --purge -y python3-pip python3-venv && sudo apt autoremove -y

# Remove VS Code's helper files from WSL
rm -rf ~/.vscode-server
```

**Delete the whole Ubuntu install.** Run this in **PowerShell**, not Ubuntu.
It permanently erases everything inside WSL Ubuntu; Windows files stay safe.

```powershell
wsl --unregister Ubuntu
```
