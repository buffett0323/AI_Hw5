# AI_Hw5
### Author: B09208038 地理四 劉正悅

### Packages required
I didn't add any new packages in requirements.txt, so it's the same as [HW5_PPT](https://docs.google.com/presentation/d/1enBQeAZwnNpqga9D7C8EezGplVjcS8T4/edit#slide=id.g2da8fe6dc22_57_12)

```bash
# First, install the needed packages for this task by running
pip install -r requirements.txt
```

### Training
```bash
# With all of the parameters using default values
# For training, please run
python pacman.py
```

### Evaluation
I have a already trained model weights in the submissions folder, named "pacman_dqn.pth". <br>
Please make sure you run in GPU since it's stored in torch.cuda.is_available()
```bash
# For evaluation, please run
python pacman.py --eval --eval_model_path submissions/pacman_dqn.pth
```