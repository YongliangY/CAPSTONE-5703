import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFocalLoss(nn.Module):
    def __init__(self, class_weights=None, gamma=2.0):
        """
        class_weights: 各类别权重列表或张量，例如 [1, 2, 1, 1] 将第二类权重设置为 2.0
                       (Class weights for imbalance, e.g. [1, 2, 1, 1] sets weight 2.0 for the second class)
        gamma: Focal Loss 的 gamma 参数，控制难易样本的调焦强度
               (Gamma parameter for Focal Loss, controls focusing on hard samples)
        """
        super(WeightedFocalLoss, self).__init__()
        if class_weights is not None:
            # 将权重转换为张量 (Convert list of weights to tensor)
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        logits: 模型未归一化的输出 [batch_size, num_classes] (Model unnormalized outputs)
        targets: 真实标签 [batch_size] (Ground-truth labels)
        """
        # Compute log-softmax for log probabilities
        log_prob = F.log_softmax(logits, dim=1)       # [N, C] log-probabilities for each class
        prob = torch.exp(log_prob)                    # [N, C] normal probabilities
        # Select probability and log-prob of the true class for each sample
        batch_indices = torch.arange(targets.size(0))
        p_t = prob[batch_indices, targets]            #  p_t (predicted prob for true class)
        log_p_t = log_prob[batch_indices, targets]    #  log(p_t) (log prob for true class)
        #  (1 - p_t)^gamma (Compute modulating factor (1 - p_t)^gamma)
        focal_factor = (1 - p_t) ** self.gamma        # higher weight for hard samples where p_t is low
        # Compute alpha balancing factor for class imbalance
        if self.class_weights is not None:
            # use per-class weight for each sample
            alpha = self.class_weights[targets]       # [N], weight for the true class of each sample
        else:
            alpha = 1.0
        #  Focal Loss: -alpha * (1-p_t)^gamma * log(p_t)
        loss = - alpha * focal_factor * log_p_t
        return loss.mean()
