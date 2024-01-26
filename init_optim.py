'''PyTorch Optimizer with learning rate and weight decay schedulers.
'''

import math
import torch.optim as optim


class InitOptimWithSGDR:
    '''Initialize an Optimizer with Stochastic Gradient Descent with Restarts (SGDR a.k.a Cosine Annealing) with Linear WarmUp strategy and weight decay (L2 Regularization).
       The weight decay rate will also be calculated using the cosine annealing strategy.
    '''


    def __init__(self, 
                 encoder_network, 
                 predictor_network, 
                 cosine_upper_bound_lr, 
                 cosine_lower_bound_lr, 
                 warmup_start_lr, 
                 warmup_steps,
                 num_steps_to_restart_lr,
                 cosine_upper_bound_wd,
                 cosine_lower_bound_wd):

        self.cosine_upper_bound_lr = cosine_upper_bound_lr
        self.cosine_lower_bound_lr = cosine_lower_bound_lr
        self.warmup_start_lr = warmup_start_lr
        self.warmup_steps = warmup_steps
        self.num_steps_to_restart_lr = num_steps_to_restart_lr
        self.step = 0
        self.cosine_upper_bound_wd = cosine_upper_bound_wd
        self.cosine_lower_bound_wd = cosine_lower_bound_wd
        self.optimizer = torch.optim.AdamW([{'params': encoder_network.parameters()}, {'params': predictor_network.parameters()}])

        assert cosine_upper_bound_lr >= cosine_lower_bound_lr, "Upper bound for LR needs to be bigger or equal to the lower bound"
        assert cosine_upper_bound_wd >= cosine_lower_bound_wd, "Upper bound for weight decay needs to be bigger or equal to the lower bound"

        def cosine_annealing(self, start_value, end_value, fraction_term):
            '''To calculate the new learning rate and the weight decay rate using the cosine annealing strategy.
            '''
            res = start_value + 0.5 * (end_value - start_value) * (1. + math.cos(math.pi * fraction_term))
            return res


        def step(self):
            '''Must be executed at every iteration step (not epoch step).
            '''

            self.step += 1

            #we're gonna need to write 2 piece of logics. 1 for the warm up period. And 1 for after the warm up period.
            if self.step <= self.warmup_steps:
                fraction_term = (cosine_upper_bound_lr - warmup_start_lr)/(self.warmup_steps) #we don't need the -1 in the denominator since we're starting the steps from 1 not 0.
                new_lr = warmup_start_lr + self.step * fraction_term

            else:
                #cosine annealing after the warmup.
                fraction_term = float(self.step - self.warmup_steps) / float(max(1, self.num_steps_to_restart_lr))
                new_lr = max(self.cosine_lower_bound_lr, self.cosine_annealing(start_value=cosine_lower_bound_lr,
                                                                               end_value=cosine_upper_bound_lr,
                                                                               fraction_term=fraction_term))
            
            #calculate the weight decay rate. There is no warmup period for decay rate and we will be using the same num of steps we used for lr for the restart.
            fraction_term = self.step / self.num_steps_to_restart_lr
            new_wd = self.cosine_annealing(start_value=cosine_lower_bound_wd,
                                           end_value=cosine_upper_bound_wd,
                                           fraction_term=fraction_term)
            
            #update the optimizer with the new learning rate and the new weight decay rate.
            for group in self.optimizer.param_groups:
                group['lr'] = new_lr

                if ("WD_exclude" not in group) or not group("WD_exclude"):
                    group['weight_decay'] = new_wd


            return new_lr, new_wd






            






