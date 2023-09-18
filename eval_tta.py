import argparse
import csv
import glob
import math
import time
from timm.models import create_model, load_checkpoint
from src.data import resolve_data_config, get_valid_transforms, get_tta_transforms, ODDDataset
from src.utils import MyEncoder
from predictor import *
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')

parser.add_argument('--flag', default="test", type=str, metavar='FLAG',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=96, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--num-classes', type=int, default=10,
                    help='Number classes in dataset')
parser.add_argument('--root-path', default='fusions', metavar='DIR',
                    help='path to root')
# Device options
parser.add_argument("-g", '--gpu-id', default='3', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

class_2_index = {0: 'aeroplane', 1: 'bicycle', 2: 'boat', 3: 'bus', 4: 'car', 5: 'chair', 6: 'diningtable', 7: 'motorbike', 8: 'sofa', 9: 'train'}

# noinspection PyTypeChecker
def dump_csv(predictor, true_output):
    results = []
    n_predictors = len(predictor.index2combine_name)
    path_json_dumps = ['%s/%s/result_%s_%s.csv' % (args.root_path, predictor.index2combine_name[i],
                                                   predictor.index2policy[i], args.flag) for i in
                       range(n_predictors)]
    print('Start eval predictor...')
    predictions = predictor.fusion_prediction(top=1, return_with_prob=True)

    if len(results) == 0:
        for i in range(len(predictions)):
            results.append([])
    for index, prediction in enumerate(predictions):
        results[index] = prediction
    assert len(results) == len(path_json_dumps), 'The result length is not equal with path_json_dumps\'s.'

    for result, save_path in zip(results, path_json_dumps):
        dir = os.path.dirname(save_path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        if args.flag == 'valid':
            total_pred_idx = []
            for i, res in enumerate(result):
                total_pred_idx.append(res[0][0])
            tf1 = f1_score(y_true=true_output, y_pred=total_pred_idx, average="macro")  # weighted macro binary
            tacc = accuracy_score(y_true=true_output, y_pred=total_pred_idx)
            print('Accuracy {:.4f} Total F1-score {:.4f}'.format(np.mean(tacc), np.mean(tf1)))
        else:
            result_list = []
            # with open(save_path, 'w', encoding="utf-8") as out_file:
            #     filenames = dataset.filenames()
            #     for i, res in enumerate(result):
            #         filename = filenames[i].split('/')[-1].strip()
            #         name = class_2_index[res[0][0]]
            #         result_data = {"image_name": str(filename), "category": name, "score": float(res[0][1])}
            #         result_list.append(result_data)
            #     json.dump(result_list, out_file)
            # classes = ['aeroplane', 'bicycle', 'boat', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa', 'train']
            with open(save_path, 'w', encoding="utf-8") as out_file:
                writer = csv.writer(out_file)
                csv_header = ["imgs", "pred"]
                writer.writerow(csv_header)
                
                filenames = dataset.filenames()
                for i, res in enumerate(result):
                    filename = filenames[i].split('/')[-1].strip()
                    name = class_2_index[res[0][0]]
                    writer.writerow([str(filename), name])
                    
        print('Dump %s finished.' % save_path)


def perform_predict(predictor, loader, model_weight, label_weight, weights, save_weights=True, save_path=None):
    temp_weight = {}
    total_true_output = []
    total_pred_output = []
    total_pred_idx = []
    total_true_idx = []
    right_count = 0
    n_labels = np.zeros((10,)) + 1e-5
    n_right_labels = np.zeros((10,))
    with torch.no_grad():
        for i, (input, target) in enumerate(tqdm(loader)):
            # if i != 1:continue
            test_pred_tta = []
            for j in range(2):
                output = predictor(input.cuda())
                output = output.data.cpu().numpy()
                test_pred_tta.append(output)
            
            output_data = softmax(np.mean(test_pred_tta, axis=0))

            if save_weights:
                predict_idx = np.argmax(output, axis=-1)
                target_idx = target.cpu().numpy()

                total_pred_idx.extend(predict_idx)
                total_true_idx.extend(target_idx)
                for j in range(len(target_idx)):
                    # 统计预测中预测对的数量，相当于precision
                    n_labels[predict_idx[j]] += 1
                    # 统计真实中预测对的数量，相当于recall
                    # n_labels[target_idx[j]] += 1

                    if predict_idx[j] == target_idx[j]:
                        right_count += 1
                        n_right_labels[predict_idx[j]] += 1
                    total_true_output.append(target_idx[j])
                    total_pred_output.append(output_data[j])
            else:
                total_pred_output.extend(output_data)

    model_name = model.default_cfg['model_name'].split('-')[1]
    if save_weights:
        # model_weight[predictor.default_cfg['model_name']] = np.array([float(right_count) / len(total_true_output)])
        # label_weight[predictor.default_cfg['model_name']] = n_right_labels / n_labels
        #
        # temp_weight['model_weight'] = float(right_count) / len(total_true_output)
        # temp_weight['label_weight'] = list(n_right_labels / n_labels)
        # weights[predictor.default_cfg['model_name']] = temp_weight

        # probs = np.max(softmax(np.array(total_pred_output)), axis=-1)
        tf1 = f1_score(y_true=total_true_idx, y_pred=total_pred_idx, average="macro")  # weighted macro binary

        target_names = ['aeroplane', 'bicycle', 'boat', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa', 'train']
        cr = classification_report(y_pred=total_pred_idx, 
                                    y_true=total_true_idx, 
                                    target_names=target_names,
                                    digits=4,
                                    output_dict=True)
        f1_list = [cr[name]['f1-score'] for name in target_names]
            
        tacc = accuracy_score(y_true=total_true_idx, y_pred=total_pred_idx)
        print('Accuracy {:.4f} Total F1-score {:.4f}'.format(tacc, tf1))
        
        with open(os.path.join(save_path, "fusion_weights_tta.json"), 'w', encoding="utf-8") as f:
            weights[predictor.default_cfg['model_name']] = {}
            weights[predictor.default_cfg['model_name']]["model_weight"] = tf1
            weights[predictor.default_cfg['model_name']]["label_weight"] = f1_list

            json.dump(weights, f, cls=MyEncoder, indent=2)

        model_weight[predictor.default_cfg['model_name']] = tf1
        label_weight[predictor.default_cfg['model_name']] = f1_list
    else:
        # with open(os.path.join(save_path, 'fusion_weights_tta.json'), 'r') as json_file:
        #     json_data = json.load(json_file)
        # model_weight[predictor.default_cfg['model_name']] = np.array(
        #     [json_data[predictor.default_cfg['model_name']]['model_weight']])
        # label_weight[predictor.default_cfg['model_name']] = np.array(
        #     [json_data[predictor.default_cfg['model_name']]['label_weight']])
        pass

    return total_pred_output, total_true_output


if __name__ == '__main__':
    test_pred = []

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    model_weight, label_weight, weights = {}, {}, {}
    predictions_dict = {}
    output_root = r'output'
    weight_path = r"weights"
    if not os.path.exists(weight_path): os.makedirs(weight_path)
    
    model_list = []
    checkpoint_list = [
        "20230912-105740-beitv2_large_patch16_224.in1k_ft_in1k-224",
        "20230912-132638-eva02_large_patch14_448.mim_m38m_ft_in22k_in1k-448",
        ]
    for checkpoint in checkpoint_list:
        name = '-'.join(checkpoint.split('-')[1:])
        model_list.append(name)

    for index, model_name in enumerate(checkpoint_list):
        img_size = int(model_name.split('-')[-1])

        checkpoint = glob.glob(os.path.join(output_root, checkpoint_list[index] + '/*.pth'))[0]
        model = create_model(
                checkpoint_list[index].split("-")[-2],
                pretrained=False,
                num_classes=10
                )
        print("load weights from:{}".format(checkpoint))
        load_checkpoint(model, checkpoint)
        model = model.cuda().eval()
        
        config = resolve_data_config(vars(args), model=model)
        if args.flag == 'valid':
            data_path = os.path.join("dataset/OODCV2023/train")
            save_weights = True
        else:
            data_path = os.path.join("dataset/OODCV2023/phase2-test-images")
            save_weights = False
            
        dataset = ODDDataset(root=data_path, 
                             transform=get_valid_transforms(img_size, resize_type="normal"), load_type="cv2")
                            #  transform=get_tta_transforms(img_size), load_type="cv2")
        test_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=args.workers,
                                                  pin_memory=True)
        model.default_cfg['model_name'] = model_list[index]

        prediction_output, true_output = perform_predict(model, test_loader, model_weight, label_weight, weights,
                                                         save_weights=save_weights, save_path=weight_path)
        predictions_dict.update({model.default_cfg['model_name']: prediction_output})

        print("finish prediction of %s" % checkpoint)
    if args.flag == 'valid':
        # ['A', 'B', 'C', 'D', 'E', 'P', 'M', 'MM', 'ML']
        # INTEGRATED_POLICY = ['A', 'B', 'C', 'D', 'E', 'M', 'MM', 'ML']
        INTEGRATED_POLICY = ['A', 'B', 'C']
    else:
        INTEGRATED_POLICY = ["A"]
    predictor = IntegratedPredictor(model_list, [predictions_dict, model_weight, label_weight], args,
                                    policies=INTEGRATED_POLICY, all_combine=False)
    dump_csv(predictor, true_output)