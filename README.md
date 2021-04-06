# Ultra Fast Speech Separation Model with Teacher Student Learning

## Introduction

Because of the ultra fast inference speed, the small speech separation Transformer model is preferred for the deployment on devices.   
In this work,  we elaborate Teacher Student learning for better training of the ultra fast speech separation model. 
The small student model is trained to reproduce the separation results  of  a  large  pretrained  teacher  model.

For a detailed description and experimental results, please refer to our paper: [Ultra Fast Speech Separation Model with Teacher Student Learning]().

## Environment
python 3.6.9, torch 1.7.1

## Get Started
1. Download the overlapped speech of [LibriCSS dataset](https://github.com/chenzhuo1011/libri_css).

    ```bash
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1PdloA-V8HGxkRu9MnT35_civpc3YXJsT' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1PdloA-V8HGxkRu9MnT35_civpc3YXJsT" -O overlapped_speech.zip && rm -rf /tmp/cookies.txt && unzip overlapped_speech.zip && rm overlapped_speech.zip
   ```

2. Download the TSTransformer separation models.

    ```bash
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1yFTJb0AeyHfTE75BH_8Di4DUC2RDix3T' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1yFTJb0AeyHfTE75BH_8Di4DUC2RDix3T" -O checkpoints.zip && rm -rf /tmp/cookies.txt && unzip checkpoints.zip && rm checkpoints.zip
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
If you find our work useful, please cite [our paper]():
```bibtex
@article{CSS_with_TSTransformer,
}
```