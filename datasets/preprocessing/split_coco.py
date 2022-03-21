import json

TRAIN=0
VAL=1
TEST=2
RESTVAL=3
split_map = {
    'train' : TRAIN,
    'val' : VAL,
    'test' : TEST,
    'restval' : RESTVAL
}
coco_root="/work/vjsalt22/dataset/coco"
output_root="/work/vjsalt22/poheng/coco"
split_json_fn="/work/vjsalt22/poheng/dataset_coco.json"


with open(split_json_fn, "r") as f:
    split_json = json.load(f)

imageID2split = {}
for image in split_json['images']:
    imageID2split[image["filename"]] = split_map[image["split"]]

orig_train_json_fn = f"{coco_root}/SpokenCOCO/SpokenCOCO_train.json"
orig_val_json_fn = f"{coco_root}/SpokenCOCO/SpokenCOCO_val.json"
orig_train =  {"data":[]}
orig_val =  {"data":[]}
orig_test =  {"data":[]}

with open(orig_train_json_fn, "r") as f:
    orig_data = json.load(f)
for image in orig_data['data']:
    split_num =  imageID2split[image["image"].split("/")[1]]
    if split_num == TRAIN:
        orig_train["data"].append(image)
    elif split_num == VAL:
        orig_val["data"].append(image)
    elif split_num == TEST:
        orig_test["data"].append(image)
    
orig_train_json_fn = f"{output_root}/SpokenCOCO/SpokenCOCO_train_karpathy.json"
orig_val_json_fn = f"{output_root}/SpokenCOCO/SpokenCOCO_val_karpathy.json"
orig_test_json_fn = f"{output_root}/SpokenCOCO/SpokenCOCO_test_karpathy.json"

with open(orig_train_json_fn, "w") as f:
    json.dump(orig_train, f)
with open(orig_val_json_fn, "w") as f:
    json.dump(orig_val, f)
with open(orig_test_json_fn, "w") as f:
    json.dump(orig_test, f)

unrolled_train_json_fn = f"{output_root}/SpokenCOCO/SpokenCOCO_train_unrolled_karpathy.json"
unrolled_val_json_fn = f"{output_root}/SpokenCOCO/SpokenCOCO_val_unrolled_karpathy.json"
unrolled_test_json_fn = f"{output_root}/SpokenCOCO/SpokenCOCO_test_unrolled_karpathy.json"

def unroll(orign_json_fn, unrolled_json_fn):
    with open(orign_json_fn) as f:
        orig_data = json.load(f)
    unrolled_data = {"data":[]}
    print(f"unroll {orign_json_fn} and store the unrolled file at {unrolled_json_fn}")
    for item in orig_data['data']:
        for caption in item['captions']:
            unrolled_data["data"].append({"image": item['image'], "caption": caption})
    length = len(unrolled_data["data"])
    print(f"get a total {length} items")
    with open(unrolled_json_fn, "w") as f:
        json.dump(unrolled_data, f)

unroll(orig_val_json_fn, unrolled_val_json_fn)
unroll(orig_test_json_fn, unrolled_test_json_fn)
unroll(orig_train_json_fn, unrolled_train_json_fn)