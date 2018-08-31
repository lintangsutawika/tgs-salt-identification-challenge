from torch.autograd import Variable
import numpy as np
EPS = 1e-12

def dice_accuracy(prob, truth, threshold=0.5,  is_average=True):
    batch_size = prob.size(0)
    p = prob.detach().view(batch_size,-1)
    t = truth.detach().view(batch_size,-1)

    p = p>threshold
    t = t>0.5
    intersection = p & t
    union        = p | t
    dice = (intersection.float().sum(1)+EPS) / (union.float().sum(1)+EPS)

    if is_average:
        dice = dice.sum()/batch_size
        return dice
    else:
        return dice

def accuracy(prob, truth, threshold=0.5,  is_average=True):
    batch_size = prob.size(0)
    p = prob.detach().view(batch_size,-1)
    t = truth.detach().view(batch_size,-1)

    p = p>threshold
    t = t>0.5
    correct = (p == t).float()
    accuracy = correct.sum(1)/p.size(1)

    if is_average:
        accuracy = accuracy.sum()/batch_size
        return accuracy
    else:
        return accuracy

def do_kaggle_metric(predict,truth, threshold=0.5):

   N = len(predict)
   predict = predict.reshape(N,-1)
   truth   = truth.reshape(N,-1)

   predict = predict>threshold
   truth   = truth>0.5
   intersection = truth & predict
   union        = truth | predict
   iou = intersection.sum(1)/(union.sum(1)+EPS)

   #-------------------------------------------
   result = []
   precision = []
   is_empty_truth   = (truth.sum(1)==0)
   is_empty_predict = (predict.sum(1)==0)

   threshold = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
   for t in threshold:
       p = iou>=t

       tp  = (~is_empty_truth)  & (~is_empty_predict) & (iou> t)
       fp  = (~is_empty_truth)  & (~is_empty_predict) & (iou<=t)
       fn  = (~is_empty_truth)  & ( is_empty_predict)
       fp_empty = ( is_empty_truth)  & (~is_empty_predict)
       tn_empty = ( is_empty_truth)  & ( is_empty_predict)

       p = (tp + tn_empty) / (tp + tn_empty + fp + fp_empty + fn)

       result.append( np.column_stack((tp,fp,fn,tn_empty,fp_empty)) )
       precision.append(p)

   result = np.array(result).transpose(1,2,0)
   precision = np.column_stack(precision)
   precision = precision.mean()

   return precision#, result, threshold

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
 
    print('\nsucess!')