import torch.optim.lr_scheduler
import vren


class Distortion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ws, deltas, ts, rays_a):
        loss, ws_inclusive_scan, wts_inclusive_scan = \
            vren.distortion_loss_fw(ws, deltas, ts, rays_a)
        ctx.save_for_backward(ws_inclusive_scan, wts_inclusive_scan,
                              ws, deltas, ts, rays_a)
        return loss

    @staticmethod
    def backward(ctx, dL_dloss):
        (ws_inclusive_scan, wts_inclusive_scan,
         ws, deltas, ts, rays_a) = ctx.saved_tensors
        dL_dws = vren.distortion_loss_bw(dL_dloss, ws_inclusive_scan,
                                         wts_inclusive_scan,
                                         ws, deltas, ts, rays_a)
        return dL_dws, None, None, None


def compute_plane_smoothness(t):
    n, c, w = t.shape
    first_difference = t[..., 1:] - t[..., :w - 1]
    second_difference = first_difference[..., 1:] - first_difference[..., :w - 2]
    return torch.square(second_difference).mean()

