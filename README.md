# singing voice conversion based on whisper & maxgan, and target to LoRA

```per
maxgan v1 == bigvgan + nsf        PlayVoice/lora-svc

maxgan v2 == bigvgan + latent f0  PlayVoice/max-svc
```

## Download maxgan_pretrain_48K_5L.pth and Test

> python svc_inference.py --config config/maxgan.yaml --model model_pretrain/maxgan_pretrain_48K_5L.pth --spk config/singers/singer0001.npy --wave test.wav

singer0001.npy~singer0056.npy can be used for test.

## Train
- 0 set work path

    > export PYTHONPATH=$PWD

- 1 download [Multi-Singer](https://github.com/Multi-Singer/Multi-Singer.github.io) data

    change sample rate of waves to `16000Hz`, and put waves to `./data_svc/waves-16k`

    > python prepare/preprocess_a.py -w ./data_raw -o ./data_svc/waves-16k -s 16000

    change sample rate of waves to `48000Hz`, and put waves to `./data_svc/waves-48k`

    > python prepare/preprocess_a.py -w ./data_raw -o ./data_svc/waves-48k -s 48000

- 2 extract pitch: 

    > python prepare/preprocess_f0.py -w data_svc/waves-16k -p data_svc/pitch

- 3 download whisper [multiple language medium model](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt), and put `medium.pt` into `whisper_pretrain/`

    > python prepare/preprocess_ppg.py -w data_svc/waves-16k -p data_svc/whisper

- 4 download speaker encoder: [Speaker-Encoder by @mueller91](https://drive.google.com/drive/folders/15oeBYf6Qn1edONkVLXe82MzdIi3O_9m3), and put `best_model.pth.tar` and `condif.json` into `speaker_pretrain/`

    > python prepare/preprocess_speaker.py ./data_svc/waves-16k ./data_svc/speaker

- 5 generate `filelist/train.txt` & `filelist/valid.txt`

    > python prepare/preprocess_train.py

- 6 debug data loader

    > python prepare/preprocess_zzz.py -c config/maxgan.yaml

- 7 start train

    > python svc_trainer.py -c config/maxgan.yaml -n svc

data tree like this

    data_svc/
    |
    └── pitch
    │     ├── spk1
    │     │   ├── 000001.pit.npy
    │     │   ├── 000002.pit.npy
    │     │   └── 000003.pit.npy
    │     └── spk2
    │         ├── 000001.pit.npy
    │         ├── 000002.pit.npy
    │         └── 000003.pit.npy
    └── speakers
    │     ├── spk1
    │     │   ├── 000001.spk.npy
    │     │   ├── 000002.spk.npy
    │     │   └── 000003.spk.npy
    │     └── spk2
    │         ├── 000001.spk.npy
    │         ├── 000002.spk.npy
    │         └── 000003.spk.npy 
    └── waves-16k
    │     ├── spk1
    │     │   ├── 000001.wav
    │     │   ├── 000002.wav
    │     │   └── 000003.wav
    │     └── spk2
    │         ├── 000001.wav
    │         ├── 000002.wav
    │         └── 000003.wav
    └── waves-48k
    │     ├── spk1
    │     │   ├── 000001.wav
    │     │   ├── 000002.wav
    │     │   └── 000003.wav
    │     └── spk2
    │         ├── 000001.wav
    │         ├── 000002.wav
    │         └── 000003.wav
    └── whisper
          ├── spk1
          │   ├── 000001.ppg.npy
          │   ├── 000002.ppg.npy
          │   └── 000003.ppg.npy
          └── spk2
              ├── 000001.ppg.npy
              ├── 000002.ppg.npy
              └── 000003.ppg.npy
![image](https://user-images.githubusercontent.com/16432329/230908037-127becb9-ed2a-41b5-8ac6-c9791ec2f7c7.png)

## Infer
- 0 set work path

    > export PYTHONPATH=$PWD

- 1 Export clean model

    > python svc_export.py --config config/maxgan.yaml --checkpoint_path chkpt/svc/***.pt

    you can download model for release page

- 2 Use whisper to extract content encoding; One-key reasoning is not used, in order to reduce the occupation of memory.

    > python whisper/inference.py -w test.wav -p test.ppg.npy

    out file is test.ppg.npy；If the ppg file is not specified in the next step, the next step will automatically generate it.

- 3 Specify parameters and inference

    > python svc_inference.py --config config/maxgan.yaml --model maxgan_g.pth --spk ./config/singers/singer0001.npy --wave test.wav

    The generated file is in the current directory svc_out.wav; at the same time, svc_out_pitch.wav is generated to visually display the pitch extraction results.

## Reference
[AdaSpeech: Adaptive Text to Speech for Custom Voice](https://arxiv.org/pdf/2103.00993.pdf)

https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/01-nsf

https://github.com/mindslab-ai/univnet [[paper]](https://arxiv.org/abs/2106.07889)

https://github.com/openai/whisper/ [[paper]](https://arxiv.org/abs/2212.04356)

https://github.com/NVIDIA/BigVGAN [[paper]](https://arxiv.org/abs/2206.04658)

https://github.com/chenwj1989/pafx

## Data-sets

KiSing        http://shijt.site/index.php/2021/05/16/kising-the-first-open-source-mandarin-singing-voice-synthesis-corpus/

PopCS         https://github.com/MoonInTheRiver/DiffSinger/blob/master/resources/apply_form.md

opencpop      https://wenet.org.cn/opencpop/download/

Multi-Singer  https://github.com/Multi-Singer/Multi-Singer.github.io

M4Singer      https://github.com/M4Singer/M4Singer/blob/master/apply_form.md

CSD           https://zenodo.org/record/4785016#.YxqrTbaOMU4

KSS           https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset

JVS MuSic     https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_music

PJS           https://sites.google.com/site/shinnosuketakamichi/research-topics/pjs_corpus

JUST Song     https://sites.google.com/site/shinnosuketakamichi/publication/jsut-song

MUSDB18       https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems

DSD100        https://sigsep.github.io/datasets/dsd100.html

Aishell-3     http://www.aishelltech.com/aishell_3

VCTK          https://datashare.ed.ac.uk/handle/10283/2651

# Notice
If you adopt the code or idea of this project, please list it in your project, which is the basic criterion for the continuation of the open source spirit.

如果你采用了本项目的代码或创意，请在你的项目中列出，这是开源精神得以延续的基本准则。

このプロジェクトのコードやアイデアを採用した場合は、オープンソースの精神が続く基本的なガイドラインであるプロジェクトにリストしてください。
