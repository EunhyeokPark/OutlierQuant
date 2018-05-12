import torch
from torch.autograd import Function
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from collections import OrderedDict

from .ol_op.ol_lib import sort_inplace_cuda, ol_quant_cuda
        
class OL_Module(torch.nn.Module):
    def __init__(self):
        if '_modules' not in self.__dict__: 
            super(OL_Module,self).__init__()

    def named_ol_layers(self, memo = None, prefix = ''):
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            for name, module in self._modules.items():
                submodule_prefix = prefix + ('.' if prefix else '') + name
                if isinstance(module, OL_Module):
                    for m in module.named_ol_layers(memo, submodule_prefix):
                        yield m

    def load_from_original(self, state_dict, print_detail = False):
        try:
            self.load_state_dict(state_dict)
        except KeyError as e:
            print("some keys are not initialized...")            
            if print_detail:
                print(e)        
        for key, value in self.named_ol_layers():
            value.ol_update(0, 0)       


class OLParam(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self):
        super(OLParam, self).__init__()
        self.bit_width = 0
        self.threshold = 0
        self.th_val = 0
        self.modified = False
        self.pos_only = False
        

#outlier base function 
class OLFunction(Function):
    def __init__(self, fparam, bparam, in_place):
        super(OLFunction, self).__init__()
        self.fparam = fparam
        self.bparam = bparam
        self.in_place = in_place
        # TODO: modify parameter to use buffers

    def _ol_quant(self, input, bit_width, threshold, th_val, pos_only):
        if bit_width <= 0 or threshold <= 0:
            return input, th_val
        else:
            if pos_only:
                N_lv_pos = 2**bit_width-1
                input_abs = input.clone()
            else:
                N_lv_pos = 2**(bit_width-1)-1
                input_abs = input.abs()

            if th_val == 0:
                if threshold >= 1:  # full-linear quantization                
                    th_val = input_abs.max()
                else:
                    num_nonzero = torch.sum(input_abs.view(-1) != 0)
                    idx = int(threshold * num_nonzero) + (input_abs.numel() - num_nonzero)                    
                    if idx == input_abs.numel():
                        idx -= 1                    
                    sort_inplace_cuda(input_abs)  
                    th_val = input_abs[idx]
            
            if th_val == 0:
                return input, th_val

        # generate output depending on threshold value condition
        if self.in_place:
            ol_quant_cuda(input, input, th_val, N_lv_pos)
            return input, th_val  
        else:
            output = input.new()
            ol_quant_cuda(input, output, th_val, N_lv_pos)
            return output, th_val              
        
    def forward(self, input):
        if self.fparam.modified:
            output, self.fparam.th_val = self._ol_quant(input, self.fparam.bit_width, self.fparam.threshold, 0, self.fparam.pos_only)
            self.fparam.modified = False
        else:
            output, self.fparam.th_val = self._ol_quant(input, self.fparam.bit_width, self.fparam.threshold, self.fparam.th_val, self.fparam.pos_only)
        return output

    def backward(self, grad_output):
        self.fparam.modified = True
        if self.bparam.modified:            
            output, self.bparam.th_val = self._ol_quant(grad_output, self.bparam.bit_width, self.bparam.threshold, 0, self.bparam.pos_only)
            #self.bparam.modified = False
        else:
            output, self.bparam.th_val = self._ol_quant(grad_output, self.bparam.bit_width, self.bparam.threshold, self.bparam.th_val, self.bparam.pos_only)
        return output  # bypass gradient    


class OL_Operator(OL_Module):
    def __init__(self, pos_only, in_place):
        super(OL_Operator, self).__init__()
        self.pos_only = pos_only
        self.fparam = OLParam()    # forward params
        self.bparam = OLParam()    # backward params
        #self.register_buffer("fparam", OLParam())
        #self.register_buffer("bparam", OLParam())
        self.fparam.pos_only = pos_only
        self.in_place = in_place
        
    
    def named_ol_layers(self, memo, prefix):
        if self not in memo:
            memo.add(self)
            yield prefix, self

    def ol_update(self, bit_width, threshold):
        self.fparam.bit_width = bit_width
        self.fparam.threshold = threshold
        self.fparam.th_val = 0
        self.fparam.modified = True

    def ol_update_back(self, bit_width, threshold):
        self.bparam.bit_width = bit_width
        self.bparam.threshold = threshold
        self.bparam.th_val = 0
        self.bparam.modified = True

    def ol_apply(self, tensor):
        ol_func = OLFunction(self.fparam, self.bparam, self.in_place)
        return ol_func(tensor)

# if bit_width == 0 or threshold == 0, then the operator should act as floating point
class OL_Conv2d(OL_Operator, torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        torch.nn.Conv2d.__init__(
            self, in_channels, out_channels, 
            kernel_size, stride,padding, dilation, groups, bias)
        OL_Operator.__init__(self, False, False)
        self.weight_quant = None        

    def forward(self, input):
        if self.weight_quant is None:
            self.fparam.modified = True            
        if self.fparam.modified == True:
            self.weight_quant = self.ol_apply(self.weight)    
        return F.conv2d(input, self.weight_quant, self.bias,
                    self.stride, self.padding, self.dilation, self.groups)

class OL_Linear(OL_Operator, torch.nn.Linear):
    def __init__(self, in_features, out_features, bias = True):
        torch.nn.Linear.__init__(self, in_features, out_features, bias)
        OL_Operator.__init__(self, False, False)
        self.weight_quant = None
  
    def forward(self, input):
        if self.weight_quant is None:
            self.fparam.modified = True
        if self.fparam.modified == True:
            self.weight_quant = self.ol_apply(self.weight)
        return F.linear(input, self.weight_quant)


class OL_Active(OL_Operator):
    def __init__(self, pos_only, in_place, avg_exp = 0.99):
        super(OL_Active, self).__init__(pos_only, in_place)

        self.th_val = 0
        self.avg_exp = avg_exp
    
    def forward(self, input):
        if self.training:            
            self.fparam.modified = True
            out = self.ol_apply(input)

            if self.th_val == 0:
                self.th_val = self.fparam.th_val
            else:
                # exponential moving average of th_val
                self.th_val = self.th_val * self.avg_exp + self.fparam.th_val *(1-self.avg_exp)
        else:
            if self.fparam.modified == True or self.th_val == 0:
                self.fparam.th_val = 0
                out = self.ol_apply(input)
                self.th_val = self.fparam.th_val
            else:                
                self.fparam.th_val = self.th_val
                out = self.ol_apply(input)        
        return out


class OL_Sequential(torch.nn.Sequential, OL_Module):
    def __init__(self, *args):
        super(OL_Sequential, self).__init__()

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            OL_idx = 0
            # 기존 Sequential과의 호환성을 위해 ol layer들은 음수로 저장한다.
            for idx, module in enumerate(args):
                if isinstance(module, OL_Active):
                    self.add_module('-' + str(idx - OL_idx), module)
                    OL_idx += 1
                else:
                    self.add_module(str(idx - OL_idx), module)

    def __getitem__(self, idx):
        """NOTE: OL_Sequential[idx] is not operated as the same as Sequential[idx]"""
        return self._modules[idx]    
