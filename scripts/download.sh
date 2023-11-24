mkdir tmp
mkdir results

echo "Downloading dataset..."
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -o /dev/null
mkdir -p data/datasets
tar -xvf LJSpeech-1.1.tar.bz2 >> /dev/null
mv LJSpeech-1.1 data/datasets/ljspeech
rm LJSpeech-1.1.tar.bz2

echo "Downloading mels for dataset..."
gdown https://drive.google.com/u/0/uc?id=1cJKJTmYd905a-9GFoo5gKjzhKjUVj83j
tar -xvf mel.tar.gz -C data/datasets/ljspeech/mels
rm mel.tar.gz

echo "Downloading alignments for dataset..."
wget https://github.com/xcmyz/FastSpeech/raw/master/alignments.zip
unzip alignments.zip -d data/datasets/ljspeech/alignments >> /dev/null
rm alignments.zip

echo "Downloading waveglow pretrained model..."
gdown https://drive.google.com/u/0/uc?id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx
mkdir -p waveglow/pretrained_model/
mv waveglow_256channels_ljs_v2.pt waveglow/pretrained_model/waveglow_256channels.pt
