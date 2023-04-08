# Ultra Fast Speech Separation Model with Teacher Student Learning

## Introduction

Because of the ultra fast inference speed, the small speech separation Transformer model is preferred for the deployment on devices.   
In this work,  we elaborate Teacher Student learning for better training of the ultra fast speech separation model. 
The small student model is trained to reproduce the separation results  of  a  large  pretrained  teacher  model.

For a detailed description and experimental results, please refer to our paper: [Ultra Fast Speech Separation Model with Teacher Student Learning](https://www.isca-speech.org/archive/pdfs/interspeech_2021/chen21l_interspeech.pdf) (Accepted by INTERSPEECH 2021).

## Environment
python 3.6.9, torch 1.7.1

## Get Started
1. Download the overlapped speech of [LibriCSS dataset](https://github.com/chenzhuo1011/libri_css).

     ```bash
    wget "https://valle.blob.core.windows.net/share/CSS_with_TSTransformer/overlapped_speech.zip?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D" -O overlapped_speech.zip && rm -rf /tmp/cookies.txt && unzip overlapped_speech.zip && rm overlapped_speech.zip
   ```

2. Download the TSTransformer separation models.

    ```bash
    wget "https://valle.blob.core.windows.net/share/CSS_with_TSTransformer/checkpoints.zip?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D" -O checkpoints.zip && rm -rf /tmp/cookies.txt && unzip checkpoints.zip && rm checkpoints.zip
    ```

3. Run the separation.

    3.1 single channel separation 

    ```bash
    export MODEL_NAME=1ch_TSTransformer
    python3 separate.py \
        --checkpoint checkpoints/$MODEL_NAME \
        --wav_list utils/overlapped_speech_1ch.scp \
        --sep_dir separated_speech/1ch/utterances_with_${MODEL_NAME} \
        --device-id 0 \
        --num_spks 2 \
        --mvdr false 
    ```
   The separated speech can be found in the directory 'separated_speech/1ch/utterances_with_${MODEL_NAME}'

    3.2 seven channel separation 

    ```bash
    export MODEL_NAME=TSTransformer
    python3 separate.py \
        --checkpoint checkpoints/$MODEL_NAME \
        --wav_list utils/overlapped_speech_7ch.scp \
        --sep_dir separated_speech/7ch/utterances_with_${MODEL_NAME} \
        --device-id 0 \
        --num_spks 2 \
        --mvdr true 
    ```
    
    The separated speech can be found in the directory 'separated_speech/7ch/utterances_with_${MODEL_NAME}'

## Citation
If you find our work useful, please cite [our paper](https://www.isca-speech.org/archive/pdfs/interspeech_2021/chen21l_interspeech.pdf):
```bibtex
@inproceedings{CSS_with_TSTransformer,
  author={Sanyuan Chen and Yu Wu and Zhuo Chen and Jian Wu and Takuya Yoshioka and Shujie Liu and Jinyu Li and Xiangzhan Yu},
  title={{Ultra Fast Speech Separation Model with Teacher Student Learning}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={3026--3030},
  doi={10.21437/Interspeech.2021-142}
}
```
