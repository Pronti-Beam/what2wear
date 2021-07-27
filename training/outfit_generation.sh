MODEL_DIR="models/model_75000.pth"
MODEL_TYPE="inception"
FEATS="features/"
IMG_SAVEPATH="generated_images/"
QUERY="query.json"
VOCAL="vocab"
python3 src/outfit_generation.py -m $MODEL_DIR -t $MODEL_TYPE -sp $FEATS --cuda -i $IMG_SAVEPATH -q $QUERY -v $VOCAB

