
All models are stored in `HunyuanWorld-Voyager/ckpts` by default, and the file structure is as follows
```shell
HunyuanWorld-Voyager
  ├──ckpts
  │  ├──README.md
  │  ├──Voyager
  │  │  ├──transformers
  │  │  │  ├──mp_rank_00_model_states.pt
  │  │  │  ├──mp_rank_00_model_states_context.pt
  │  ├──hunyuan-video-i2v-720p
  │  │  ├──vae
  │  ├──text_encoder_i2v
  │  ├──text_encoder_2
  ├──...
```

## Download HunyuanWorld-Voyager model
To download the HunyuanWorld-Voyager model, first install the huggingface-cli. (Detailed instructions are available [here](https://huggingface.co/docs/huggingface_hub/guides/cli).)

```shell
python -m pip install "huggingface_hub[cli]"
```

Then download the model using the following commands:

```shell
# Switch to the directory named 'HunyuanWorld-Voyager'
cd HunyuanWorld-Voyager
# Use the huggingface-cli tool to download HunyuanWorld-Voyager model in HunyuanWorld-Voyager/ckpts dir.
# The download time may vary from 10 minutes to 1 hour depending on network conditions.
huggingface-cli download tencent/HunyuanWorld-Voyager --local-dir ./ckpts
```

<details>
<summary>💡Tips for using huggingface-cli (network problem)</summary>

##### 1. Using HF-Mirror

If you encounter slow download speeds in China, you can try a mirror to speed up the download process. For example,

```shell
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download tencent/HunyuanWorld-Voyager --local-dir ./ckpts
```

##### 2. Resume Download

`huggingface-cli` supports resuming downloads. If the download is interrupted, you can just rerun the download 
command to resume the download process.

Note: If an `No such file or directory: 'ckpts/.huggingface/.gitignore.lock'` like error occurs during the download 
process, you can ignore the error and rerun the download command.

</details>

 

