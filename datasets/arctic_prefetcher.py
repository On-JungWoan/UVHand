# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch
from util.misc import NestedTensor

def to_cuda(samples, targets, metas, device):
    try:
        samples = samples.to(device, non_blocking=True)
    except:
        for idx, v in enumerate(samples):
            samples[idx] = v.to(device, non_blocking=True)
    # targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

    # targets
    for k,v in targets.items():
        if k == 'labels':
            targets[k]=v
        else:
            targets[k] = v.to(device, non_blocking=True)

    # meta
    for k,v in metas.items():
        if k == 'imgname' or k == 'query_names':
            metas[k]=v
        else:
            metas[k] = v.to(device, non_blocking=True)     
    return samples, targets, metas

class data_prefetcher():
    def __init__(self, loader, device, prefetch=True):
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        if prefetch:
            self.stream = torch.cuda.Stream()
            self.preload()

    def preload(self):
        try:
            self.next_samples, self.next_targets, self.next_metas = next(self.loader)
        except StopIteration:
            self.next_samples = None
            self.next_targets = None
            self.next_metas = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            # if isinstance(self.next_samples, NestedTensor):
            self.next_samples, self.next_targets, self.next_metas = to_cuda(self.next_samples, self.next_targets, self.next_metas, self.device)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:

    def next(self):
        if self.prefetch:
            torch.cuda.current_stream().wait_stream(self.stream)
            samples = self.next_samples
            targets = self.next_targets
            metas = self.next_metas

            if samples is not None:
                if isinstance(samples, list):
                    for sample in samples:
                        sample.record_stream(torch.cuda.current_stream())
                else:
                    samples.record_stream(torch.cuda.current_stream())
            if targets is not None:
                for k, v in targets.items():
                    if k == 'labels':
                        continue
                    v.record_stream(torch.cuda.current_stream())
            if metas is not None:
                for k, v in metas.items():
                    if k == 'imgname' or k == 'query_names':
                        continue
                    v.record_stream(torch.cuda.current_stream())                        
            self.preload()
        else:
            try:
                samples, targets, metas = next(self.loader)
                samples, targets, metas = to_cuda(samples, targets, metas, self.device)
            except StopIteration:
                samples = None
                targets = None
                metas = None
        return samples, targets, metas
