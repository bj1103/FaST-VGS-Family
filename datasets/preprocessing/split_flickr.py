import json

with open('/Users/kevin/Desktop/dataset_flickr8k.json', 'r') as f:
    data = json.load(f)

train = []
val = []
test = []

for image in data['images']:
    if image["split"] == 'train':
        for i in range(5):
            train.append({ 'image' : image["filename"], 'wav' :  image["filename"].split('.')[0]+'_'+str(i)+'.wav' })
    elif image["split"] == 'val':
        for i in range(5):
            val.append({ 'image' : image["filename"], 'wav' :  image["filename"].split('.')[0]+'_'+str(i)+'.wav' })
    elif image["split"] == 'test':
        for i in range(5):
            test.append({ 'image' : image["filename"], 'wav' :  image["filename"].split('.')[0]+'_'+str(i)+'.wav' })
    else:
        print('out split data')
        raise AssertionError()


with open('/Users/kevin/Desktop/flickr8k_train.json', 'w') as f:
    json.dump({ 'data': train }, f)
with open('/Users/kevin/Desktop/flickr8k_dev.json', 'w') as f:
    json.dump({ 'data': val }, f)
with open('/Users/kevin/Desktop/flickr8k_test.json', 'w') as f:
    json.dump({ 'data': test }, f)