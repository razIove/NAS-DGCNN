
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter


def copy_bn(target_bn, src_bn):
    feature_dim = (
        target_bn.num_channels
        if isinstance(target_bn, nn.GroupNorm)
        else target_bn.num_features
    )

    target_bn.weight.data.copy_(src_bn.weight.data[:feature_dim])
    target_bn.bias.data.copy_(src_bn.bias.data[:feature_dim])

    if type(src_bn) in [nn.BatchNorm1d, nn.BatchNorm2d]:
        target_bn.running_mean.data.copy_(src_bn.running_mean.data[:feature_dim])
        target_bn.running_var.data.copy_(src_bn.running_var.data[:feature_dim])

class DynamicLinear(nn.Module):

    def __init__(self, max_in_features, max_out_features, bias=True):
        super(DynamicLinear, self).__init__()

        self.max_in_features = max_in_features
        self.max_out_features = max_out_features
        self.bias = bias

        self.linear = nn.Linear(self.max_in_features, self.max_out_features, self.bias)

        self.active_in_features = self.max_in_features
        self.active_out_features = self.max_out_features

    def get_active_layer(self, in_features, preserve_weight=True):
        layer = nn.Linear(in_features,self.active_out_features,self.bias).to(self.parameters().__next__().device)
        if not preserve_weight:
            return layer

        layer.weight.data.copy_(
            self.get_active_weight(self.active_out_features, in_features).data
        )
        if self.bias:
            layer.bias.data.copy_(
            self.get_active_bias(self.active_out_features).data
            )
        return nn.Sequential(layer)

    def get_active_weight(self, out_features, in_features):

        if isinstance(out_features, int):
            assert out_features <= self.linear.weight.size(0)
            _weight = self.linear.weight[:out_features]
        elif isinstance(out_features, torch.LongTensor):
            assert out_features.max().item() <= self.linear.weight.size(0)
            _weight = torch.index_select(self.linear.weight, 0, out_features)
        else:
            raise TypeError(f"out_features is {type(out_features)} not in Union[int, LongTensor]")
        
        if isinstance(self.active_in_features, list):
            assert max(self.active_in_features) <= self.linear.weight.size(1)
            _weight =  _weight[:,self.active_in_features]
        else: 
            assert in_features <= self.linear.weight.size(1)
            _weight =  _weight[:,:in_features]

        
        return _weight.contiguous()
        # return self.linear.weight[:out_features, :in_features]

    def get_active_bias(self, out_features):
        if not self.bias:
            return None

        if isinstance(out_features, int):
            assert out_features <= self.linear.bias.size(0)
            _bias = self.linear.bias[:out_features]
        elif isinstance(out_features, torch.LongTensor):
            assert out_features.max().item() <= self.linear.bias.size(0)
            _bias = torch.index_select(self.linear.bias, 0, out_features)
        else:
            raise TypeError(f"out_features is {type(out_features)} not in Union[int, LongTensor]")
        
        return _bias.contiguous()
        # return self.linear.bias[:out_features] if self.bias else None


    def forward(self, x, out_features=None):
        if (isinstance(self.active_out_features, int) and self.active_out_features == self.max_out_features) and x.size(1) == self.max_in_features:
            return self.linear(x)

        if out_features is None:
            out_features = self.active_out_features

        in_features = x.size(1)
        weight = self.get_active_weight(out_features, in_features)
        bias = self.get_active_bias(out_features)
        y = F.linear(x, weight, bias)

        return y



class DynamicBatchNorm2d(nn.Module):
    SET_RUNNING_STATISTICS = False

    def __init__(self, max_feature_dim):
        super(DynamicBatchNorm2d, self).__init__()

        self.max_feature_dim = max_feature_dim
        self.bn = nn.BatchNorm2d(self.max_feature_dim)

    @staticmethod
    def bn_forward(x, bn: nn.BatchNorm2d, feature_dim):
        if bn.num_features == feature_dim or DynamicBatchNorm2d.SET_RUNNING_STATISTICS:
            return bn(x)
        else:
            exponential_average_factor = 0.0

            if bn.training and bn.track_running_stats:
                if bn.num_batches_tracked is not None:
                    bn.num_batches_tracked += 1
                    if bn.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(bn.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = bn.momentum
            return F.batch_norm(
                x,
                bn.running_mean[:feature_dim],
                bn.running_var[:feature_dim],
                bn.weight[:feature_dim],
                bn.bias[:feature_dim],
                bn.training or not bn.track_running_stats,
                exponential_average_factor,
                bn.eps,
            )

    def forward(self, x):
        feature_dim = x.size(1)
        y = self.bn_forward(x, self.bn, feature_dim)
        return y


class DynamicBatchNorm1d(nn.Module):
    SET_RUNNING_STATISTICS = False

    def __init__(self, max_feature_dim):
        super(DynamicBatchNorm1d, self).__init__()

        self.max_feature_dim = max_feature_dim
        self.bn = nn.BatchNorm1d(self.max_feature_dim)

    @staticmethod
    def bn_forward(x, bn: nn.BatchNorm1d, feature_dim):
        if bn.num_features == feature_dim or DynamicBatchNorm1d.SET_RUNNING_STATISTICS:
            return bn(x)
        else:
            exponential_average_factor = 0.0

            if bn.training and bn.track_running_stats:
                if bn.num_batches_tracked is not None:
                    bn.num_batches_tracked += 1
                    if bn.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(bn.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = bn.momentum           
            return F.batch_norm(
                x,
                bn.running_mean[:feature_dim],
                bn.running_var[:feature_dim],
                bn.weight[:feature_dim],
                bn.bias[:feature_dim],
                bn.training or not bn.track_running_stats,
                exponential_average_factor,
                bn.eps,
            )

    def forward(self, x):
        feature_dim = x.size(1)
        y = self.bn_forward(x, self.bn, feature_dim)
        return y


class DynamicConv2d(nn.Module):
    def __init__(self, max_in_channels, max_out_channels, kernel_size=1, bias=False):
        super(DynamicConv2d, self).__init__()

        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(
            self.max_in_channels,
            self.max_out_channels,
            self.kernel_size,
            bias=False,
        )

        self.active_out_channel = self.max_out_channels

    def get_active_filter(self, out_channel, in_channel):
        return self.conv.weight[:out_channel, :in_channel, :, :]

    def get_active_layer(self, in_features, preserve_weight=True):
        layer = nn.Conv2d(
            in_features, self.active_out_channel, self.kernel_size, bias=False
        ).to(self.parameters().__next__().device)
        if not preserve_weight:
            return layer

        layer.weight.data.copy_(
            self.get_active_filter(self.active_out_channel, in_features).data
        )

        return nn.Sequential(layer)

    def forward(self, x, out_channel=None):
        if out_channel is None:
            out_channel = self.active_out_channel
        in_channel = x.size(1)
        filters = self.get_active_filter(out_channel, in_channel).contiguous()
        y = F.conv2d(x, filters, None)
        return y


class DynamicConv1d(nn.Module):
    def __init__(self, max_in_channels, max_out_channels, kernel_size=1, bias=False):
        super(DynamicConv1d, self).__init__()

        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size = kernel_size
        self.set_input_dim = False
        self.conv = nn.Conv1d(
            self.max_in_channels,
            self.max_out_channels,
            self.kernel_size,
            bias=False,
        )

        self.active_out_channel = self.max_out_channels

    def get_active_filter(self, out_channel, in_channel):
        if in_channel == self.max_in_channels:
            return self.conv.weight[:out_channel, :, :]
        in_channels = []
        if self.set_input_dim == False:
            return self.conv.weight[:out_channel, :in_channel, :]

        for i, w in enumerate(self.sample_dims):
            in_channels.extend(range(self.ori_dims[i], w + self.ori_dims[i]))

        return self.conv.weight[:out_channel, in_channels, :]

    def set_dim(self, sample, ori):
        self.sample_dims = sample
        self.ori_dims = ori
        self.set_input_dim = True
        return

    def get_active_layer(self, in_features, preserve_weight=False):
        layer = nn.Conv1d(
            in_features, self.active_out_channel, self.kernel_size, bias=False
        ).to(self.parameters().__next__().device)
        if not preserve_weight:
            return layer

        layer.weight.data.copy_(
            self.get_active_filter(self.active_out_channel, in_features).data
        )

        return layer

    def forward(self, x, out_channel=None):
        if out_channel is None:
            out_channel = self.active_out_channel
        in_channel = x.size(1)

        filters = self.get_active_filter(out_channel, in_channel).contiguous()

        y = F.conv1d(x, filters, None)
        return y


if __name__ == "__main__":
    layer = DynamicConv1d(10, 30)
    inputs = torch.rand(1, 10, 5)

    print(layer(inputs).shape)
