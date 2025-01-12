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


def compute_plane_tv(t):
    batch_size, c, h, w = t.shape
    count_h = batch_size * c * (h - 1) * w
    count_w = batch_size * c * h * (w - 1)
    h_tv = torch.square(t[..., 1:, :] - t[..., :h - 1, :]).sum()
    w_tv = torch.square(t[..., :, 1:] - t[..., :, :w - 1]).sum()
    return 2 * (h_tv / count_h + w_tv / count_w)  # This is summing over batch and c instead of avg


def compute_plane_smoothness(t):
    batch_size, c, h, w = t.shape
    # Convolve with a second derivative filter, in the time dimension which is dimension 2
    first_difference = t[..., 1:, :] - t[..., :h - 1, :]  # [batch, c, h-1, w]
    second_difference = first_difference[..., 1:, :] - first_difference[..., :h - 2, :]  # [batch, c, h-2, w]
    # Take the L2 norm of the result
    return torch.square(second_difference).mean()


class NeDepth(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sigmas, deltas, ts, rays_a, vr_samples):
        ne_depth = \
            vren.ne_depth_fw(sigmas, deltas, ts, rays_a, vr_samples)
        ctx.save_for_backward(ne_depth, sigmas, deltas, ts, rays_a, vr_samples)
        return ne_depth

    @staticmethod
    def backward(ctx, dL_ddepth):
        (ne_depth, sigmas, deltas, ts, rays_a, vr_samples) = ctx.saved_tensors
        dl_dsigmas = vren.ne_depth_bw(dL_ddepth, ne_depth, sigmas, deltas, ts,
                                      rays_a, vr_samples)
        return dl_dsigmas, None, None, None, None
