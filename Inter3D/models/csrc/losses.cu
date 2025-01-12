#include "utils.h"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>


// for details of the formulae, please see https://arxiv.org/pdf/2206.05085.pdf

template <typename scalar_t>
__global__ void prefix_sums_kernel(
    const scalar_t* __restrict__ ws,
    const scalar_t* __restrict__ wts,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    scalar_t* __restrict__ ws_inclusive_scan,
    scalar_t* __restrict__ ws_exclusive_scan,
    scalar_t* __restrict__ wts_inclusive_scan,
    scalar_t* __restrict__ wts_exclusive_scan
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= rays_a.size(0)) return;

    const int start_idx = rays_a[n][1], N_samples = rays_a[n][2];

    // compute prefix sum of ws and ws*ts
    // [a0, a1, a2, a3, ...] -> [a0, a0+a1, a0+a1+a2, a0+a1+a2+a3, ...]
    thrust::inclusive_scan(thrust::device,
                           ws+start_idx,
                           ws+start_idx+N_samples,
                           ws_inclusive_scan+start_idx);
    thrust::inclusive_scan(thrust::device,
                           wts+start_idx,
                           wts+start_idx+N_samples,
                           wts_inclusive_scan+start_idx);
    // [a0, a1, a2, a3, ...] -> [0, a0, a0+a1, a0+a1+a2, ...]
    thrust::exclusive_scan(thrust::device,
                           ws+start_idx,
                           ws+start_idx+N_samples,
                           ws_exclusive_scan+start_idx);
    thrust::exclusive_scan(thrust::device,
                           wts+start_idx,
                           wts+start_idx+N_samples,
                           wts_exclusive_scan+start_idx);
}


template <typename scalar_t>
__global__ void distortion_loss_fw_kernel(
    const scalar_t* __restrict__ _loss,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> loss
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= rays_a.size(0)) return;

    const int ray_idx = rays_a[n][0], start_idx = rays_a[n][1], N_samples = rays_a[n][2];

    loss[ray_idx] = thrust::reduce(thrust::device, 
                                   _loss+start_idx,
                                   _loss+start_idx+N_samples,
                                   (scalar_t)0);
}


std::vector<torch::Tensor> distortion_loss_fw_cu(
    const torch::Tensor ws,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a
){
    const int N_rays = rays_a.size(0), N = ws.size(0);

    auto wts = ws * ts;

    auto ws_inclusive_scan = torch::zeros({N}, ws.options());
    auto ws_exclusive_scan = torch::zeros({N}, ws.options());
    auto wts_inclusive_scan = torch::zeros({N}, ws.options());
    auto wts_exclusive_scan = torch::zeros({N}, ws.options());

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(ws.type(), "distortion_loss_fw_cu_prefix_sums", 
    ([&] {
        prefix_sums_kernel<scalar_t><<<blocks, threads>>>(
            ws.data_ptr<scalar_t>(),
            wts.data_ptr<scalar_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            ws_inclusive_scan.data_ptr<scalar_t>(),
            ws_exclusive_scan.data_ptr<scalar_t>(),
            wts_inclusive_scan.data_ptr<scalar_t>(),
            wts_exclusive_scan.data_ptr<scalar_t>()
        );
    }));

    auto _loss = 2*(wts_inclusive_scan*ws_exclusive_scan-
                    ws_inclusive_scan*wts_exclusive_scan) + 1.0f/3*ws*ws*deltas;

    auto loss = torch::zeros({N_rays}, ws.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(ws.type(), "distortion_loss_fw_cu", 
    ([&] {
        distortion_loss_fw_kernel<scalar_t><<<blocks, threads>>>(
            _loss.data_ptr<scalar_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            loss.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return {loss, ws_inclusive_scan, wts_inclusive_scan};
}


template <typename scalar_t>
__global__ void distortion_loss_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dloss,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ws_inclusive_scan,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> wts_inclusive_scan,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ws,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> deltas,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ts,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dws
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= rays_a.size(0)) return;

    const int ray_idx = rays_a[n][0], start_idx = rays_a[n][1], N_samples = rays_a[n][2];
    const int end_idx = start_idx+N_samples-1;

    const scalar_t ws_sum = ws_inclusive_scan[end_idx];
    const scalar_t wts_sum = wts_inclusive_scan[end_idx];
    // fill in dL_dws from start_idx to end_idx
    for (int s=start_idx; s<=end_idx; s++){
        dL_dws[s] = dL_dloss[ray_idx] * 2 * (
            (s==start_idx?
                (scalar_t)0:
                (ts[s]*ws_inclusive_scan[s-1]-wts_inclusive_scan[s-1])
            ) + 
            (wts_sum-wts_inclusive_scan[s]-ts[s]*(ws_sum-ws_inclusive_scan[s]))
        );
        dL_dws[s] += dL_dloss[ray_idx] * (scalar_t)2/3*ws[s]*deltas[s];
    }
}


torch::Tensor distortion_loss_bw_cu(
    const torch::Tensor dL_dloss,
    const torch::Tensor ws_inclusive_scan,
    const torch::Tensor wts_inclusive_scan,
    const torch::Tensor ws,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a
){
    const int N_rays = rays_a.size(0), N = ws.size(0);

    auto dL_dws = torch::zeros({N}, dL_dloss.options());

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(ws.type(), "distortion_loss_bw_cu", 
    ([&] {
        distortion_loss_bw_kernel<scalar_t><<<blocks, threads>>>(
            dL_dloss.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            ws_inclusive_scan.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            wts_inclusive_scan.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            ws.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            deltas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            ts.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            dL_dws.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return dL_dws;
}



// -------------------------------------------------------------------------------

template <typename scalar_t>
__global__ void ne_depth_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> sigmas,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> deltas,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ts,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> vr_samples,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> depth
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= rays_a.size(0)) return;

    const int ray_idx = rays_a[n][0], start_idx = rays_a[n][1];
    const int N_samples=vr_samples[ray_idx];
    const int end_idx = start_idx+N_samples-1;
    scalar_t T = 1.0f;
    for (int s=end_idx; s>=start_idx; s--){
        const scalar_t a = 1.0f - __expf(-sigmas[s]*deltas[s]);
        const scalar_t w = a*T;
        depth[ray_idx] += w*ts[s];
        T *= 1.0f-a;
    }

}


torch::Tensor  ne_depth_fw_cu(
    const torch::Tensor sigmas,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const torch::Tensor vr_samples
){
    const int N_rays = rays_a.size(0), N = sigmas.size(0);
    const int threads = 256, blocks = (N_rays+threads-1)/threads;
    auto depth = torch::zeros({N_rays}, sigmas.options());


    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigmas.type(), "ne_depth_fw_cu",
    ([&] {
        ne_depth_fw_kernel<scalar_t><<<blocks, threads>>>(
            sigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            deltas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            ts.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            vr_samples.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
            depth.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()
        );
    }));


    return depth;
}



template <typename scalar_t>
__global__ void ne_depth_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_ddepth,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> depth,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> sigmas,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> deltas,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ts,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> vr_samples,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dsigmas
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= rays_a.size(0)) return;

    const int ray_idx = rays_a[n][0], start_idx = rays_a[n][1];
    const int N_samples=vr_samples[ray_idx];
    const int end_idx = start_idx+N_samples-1;

    scalar_t D = depth[ray_idx];
    scalar_t T = 1.0f,d = 0.0f;

    for (int s=end_idx; s>=start_idx; s--){
        const scalar_t a = 1.0f - __expf(-sigmas[s]*deltas[s]);
        d += a*T*ts[s];
        T *= 1.0f-a;
        dL_dsigmas[s] = deltas[s] * (
            dL_ddepth[ray_idx]*(ts[s]*T-(D-d))
        );
    }

}


torch::Tensor ne_depth_bw_cu(
    const torch::Tensor dL_ddepth,
    const torch::Tensor depth,
    const torch::Tensor sigmas,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const torch::Tensor vr_samples
){
    const int N_rays = rays_a.size(0), N = sigmas.size(0);
    const int threads = 256, blocks = (N_rays+threads-1)/threads;
    auto dL_dsigmas = torch::zeros({N}, sigmas.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigmas.type(), "ne_depth_bw_cu",
    ([&] {
        ne_depth_bw_kernel<scalar_t><<<blocks, threads>>>(
            dL_ddepth.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            depth.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            sigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            deltas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            ts.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            vr_samples.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
            dL_dsigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()
        );
    }));


    return dL_dsigmas;
}
