import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import config
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

#device = torch.device('cude') if torch.cuda.is_available() else torch.device('cpu')

def get_rpn_rois_old (images_list, model):
    '''This function returns feature maps of the selected ROIS
    shape output is [nb regions, 256, 7 , 7]
    input: list of image tensors, each tensor has shape [3,w,h]
    output tensors: selected_rois, boxes_on_image, labels, scores'''
    selected_rois = []
    boxes_on_image = []
    labels = []
    scores = []
    
    outputs = []
    model.eval()
    
    hook = model.backbone.register_forward_hook(
        lambda self, input, output: outputs.append(output)
        )
    res = model(images_list)
    hook.remove()
    #rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
    # box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
            #the locations indicated by the bounding boxes (output of RPN)
    for j in range(len(outputs)):
        selected_rois.append( model.roi_heads.box_roi_pool(
                outputs[j], [r['boxes'] for r in res], [i.shape[-2:] for i in images_list[j]]
                ) )
    for i in range (len(res)):
        boxes_on_image.append( res[i]['boxes'])
        labels.append( res[i]['labels'])
        scores.append (res[i]['scores'])  
    #import pdb;pdb.set_trace()  
    return selected_rois, boxes_on_image, labels, scores


def image_transform(image_path):
    to_tensor = transforms.ToTensor()
    #img = Image.open(pic) 
    im = cv2.imread(image_path) # Read image with cv2
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) # Convert to RGB
    #t_img = to_tensor(img).float() # convert to tensor
    im_shape = im.shape
    im_height = im_shape[0]
    im_width = im_shape[1]

    img = to_tensor(im).float() # convert to tensor
    im_info = {"width": im_width, "height": im_height}
    return img, im_info 

def get_rpn_rois (images_list, model):
    '''This function returns the embedding of fc6 layer of the selected ROIS
    shape output is [nb regions, 1024]
    input: list of image tensors, each tensor has shape [3,w,h]
    output tensors: selected_rois, boxes_on_image, labels, scores'''
    embedding_selected_rois = []
    boxes_on_image = []
    labels = []
    scores = []
    
    for image in images_list:

        outputs = []
        model.eval()
        with torch.no_grad():
            hook = model.backbone.register_forward_hook(
                lambda self, input, output: outputs.append(output)
                )
            res = model([image])
            hook.remove()

            #rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
            # box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
                    #the locations indicated by the bounding boxes (output of RPN)
            
        
            this_output =  model.roi_heads.box_roi_pool(
                    outputs[0], [r['boxes'] for r in res], [i.shape[-2:] for i in image]
                    )
            this_output = this_output.flatten(start_dim=1)
            this_output= model.roi_heads.box_head.fc6(this_output)
            embedding_selected_rois.append(this_output)
        
        for i in range (len(res)):
            boxes_on_image.append( res[i]['boxes'])
            labels.append( res[i]['labels'])
            scores.append (res[i]['scores'])
    

    return embedding_selected_rois, boxes_on_image, labels, scores

def get_selected_rois (embedding_selected_rois, boxes_on_image, labels, scores, best_features = 10):
    '''Input tensors: all of them are list of tensors 
    labels[0] is tensors of the labels in image 0
    Output Tensors are the same input tensors just after selecting based on the threshold'''
    for i in range(len(labels)): # number of images in the list
        #ind_roi = [] # list of indexes of RoIs with scores higher than threshold
        #high_indx = 1 # since the scores are sorted, we can get the higher idx that bigger thn threshold
            # at the worst case, we find one value
        #for j in range (scores[i].shape[0]):
        #    if scores[i][j].item() >= threshold :
        #        ind_roi.append(j)
        #        high_indx = j
        if len(embedding_selected_rois[i]) >= best_features:
            embedding_selected_rois[i] = embedding_selected_rois[i][:best_features]
            boxes_on_image[i] = boxes_on_image[i][:best_features]
            labels[i] = labels[i][:best_features]
            scores[i] = scores[i][:best_features]

    return embedding_selected_rois, boxes_on_image, labels, scores   


def process_feature_extraction(output, im_infos):
    '''
    output = {
        'embedding_rois': rois,
        'bbox': boxes_on_image,
        'labels': labels,
        'scores': scores
        }
    im_info = {"width": im_width, "height": im_height}
    '''
    batch_size = len(output["embedding_rois"])
    feat_list = []
    info_list = []

    for i in range(batch_size):
        #feat_list.append(output['embedding_rois'][i])
        feat_list.append(torch.cat((output['embedding_rois'][i],output['embedding_rois'][i]), 1))
        info_list.append(
            {
                "bbox": output['bbox'][i].cpu().numpy(),
                "num_boxes": len(output['bbox'][i]),
                "objects": output['labels'][i],
                "image_width": im_infos[i]["width"],
                "image_height": im_infos[i]["height"],
                "cls_prob": output['scores'][i].cpu().numpy(),
            }
        )

    return feat_list, info_list


def extract_features(image_paths, model):
    '''
    image_paths: is a list of input image paths
    '''
    img_tensor, im_infos = [], []
    for image_path in image_paths:
        im, im_info = image_transform(image_path)
        img_tensor.append(im)
        im_infos.append(im_info)

    rois, boxes_on_image, labels, scores =   get_rpn_rois(img_tensor, model)
    rois, boxes_on_image, labels, scores = get_selected_rois(rois, boxes_on_image, labels, scores,\
                                    best_features = 10)

    output = {
        'embedding_rois': rois,
        'bbox': boxes_on_image,
        'labels': labels,
        'scores': scores
        }

    features, infos = process_feature_extraction(
        output,
        im_infos,
    )

    return features, infos


def visualize_tensor(t_img_list ,boxes_on_image_tensor, labels_tensor, scores_tensor):
    ''' visualize all the images in the t_img_list
    t_img is one of the image tensors
    '''
    for im in range(len(t_img_list)):
        t_img = t_img_list[im]
        img = t_img.squeeze(0).permute(1, 2, 0) * 255
        img = img.numpy().astype(np.uint8)
        pred_class = []
        coco = config.coco['coco_instance_category_name']

        for i in list(labels_tensor[im].numpy()):
            pred_class.append(coco[i])
        
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(boxes_on_image_tensor[im].detach().numpy())] # Bounding boxes
        pred_score = list(scores_tensor[im].detach().numpy())
        
        #for i in range(len(pred_boxes)):
        #    cv2.rectangle(img, pred_boxes[i][0], pred_boxes[i][1],color=(255, 0, 0), thickness=1) # Draw Rectangle with the coordinates
        #    cv2.putText(img,pred_class[i], pred_boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1, cv2.LINE_AA) # Write the prediction class
        #    cv2.putText(img,str(round(pred_score[i], 2)), pred_boxes[i][1],  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),1, cv2.LINE_AA) # Write the prediction class

        print('labels: ', pred_class)
        print('scores: ', pred_score)
        plt.figure(figsize=(20,30)) # display the output image
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.show()

def _chunks(self, array, chunk_size):
        for i in range(0, len(array), chunk_size):
            yield array[i : i + chunk_size]

def _save_feature(self, file_name, feature, info):
    file_base_name = os.path.basename(file_name)
    file_base_name = file_base_name.split(".")[0]
    info["image_id"] = file_base_name
    info["features"] = feature.cpu().numpy()
    file_base_name = file_base_name + ".npy"

if __name__ == '__main__':
    ## Faster RCNN 
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # read image 

    pic = "test2.png"
    pic1 = "test.png"
    image_paths = [pic, pic1]
    features, infos = extract_features(image_paths, model)
    import pdb; pdb.set_trace()
