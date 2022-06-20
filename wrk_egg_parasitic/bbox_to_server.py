import json
import argparse

# to ICIP server
parser = argparse.ArgumentParser(description='MMdet bbox JSON result to ICIP COCO annotations file.')
parser.add_argument('predictions', metavar='mmdet bbox predictions annotations', type=str,
                    help='Path to COCO annotations file.')
parser.add_argument('out', metavar='coco_result', type=str,
                    help='Path to COCO annotations file.')
parser.add_argument('thr', metavar='coco_result_thr', type=float, default=0.56,
    help='score threshold (default: 0.6)')

args = parser.parse_args()

def save_coco(file, result):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({'annotations': result }, coco, indent=2, sort_keys=True)

# map image_id -> image filename
def mapImageIds(images):
    imgToFilename = {}

    for img in images:
        imgToFilename[img['id']] = img['file_name']
    return imgToFilename

def filter_thr_score(bbox, imgToFilename, thr=0.6):
    ann_id = 0
    thr_bbox_score = []
    for bb in bbox:
        if(bb['score'] >= thr):
            imgId = bb['image_id']
            bb['file_name'] = imgToFilename[imgId]
            bb['id'] = ann_id
            thr_bbox_score.append(bb)
            ann_id += 1

    #remove extra attr
    for bb in thr_bbox_score:
        bb.pop('image_id', None)
        bb.pop('score', None)

    return thr_bbox_score

def main(args):
    #load predictions and images json (unlabled)
    predictions = open(args.predictions, 'rt', encoding='UTF-8')
    images = open('./unlabled_test_images.json', 'rt', encoding='UTF-8')

    predictions = json.load(predictions)
    without_thr = len(predictions) 

    images = json.load(images)["images"]

    # Pre-processing
    #lineararing
    #apply given thr
    imageid_to_filename = mapImageIds(images)
    result = filter_thr_score(predictions, imageid_to_filename, args.thr)
    with_thr = len(result)

    out_filename = args.out + '_' + str(args.thr)  + '.json'

    save_coco(out_filename, result)

    print("Total predictions {} ".format(without_thr))
    print("Saved {} annotations in {} score threshold >=  {}".format(with_thr, out_filename, args.thr))


if __name__ == "__main__":
    main(args)
