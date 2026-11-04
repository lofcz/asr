"""Synchronous LoRA training: one whole session per GPU, NCCL gradient reduction."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import argparse, json, time, random, datetime
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from train import BASE, Cache, BoundaryOffload, load_model, loss_for, MODEL_REVISION


class Objective(torch.nn.Module):
    def __init__(self, model, offload):
        super().__init__()
        self.model = model
        self.offload = offload

    def forward(self, item):
        return loss_for(self.model, item, self.offload)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--mode', choices=['bf16', 'qlora'], default='qlora')
    p.add_argument('--bucket', type=int, default=1800)
    p.add_argument('--steps', type=int, default=2)
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--mlp-chunk', type=int, default=1024)
    p.add_argument('--offload', action='store_true')
    p.add_argument('--pageable-offload', action='store_true')
    p.add_argument('--save', action='store_true')
    args = p.parse_args()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(2)
    dist.init_process_group('nccl', device_id=torch.device('cuda', local_rank), timeout=datetime.timedelta(minutes=20))
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(20261022)
    random.seed(20261022)
    out = BASE / 'runs' / args.name
    if rank == 0:
        out.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    cache = Cache()
    rows = [r for r in cache.rows if r['split'] == 'train' and r['bucket'] == args.bucket and r.get('context_mode','none') == 'none']
    assert rows
    random.shuffle(rows)
    model = load_model(args.mode, args.rank, args.mlp_chunk)
    objective = Objective(model, BoundaryOffload(args.offload, pin_memory=not args.pageable_offload))
    ddp = DistributedDataParallel(objective, device_ids=[local_rank], broadcast_buffers=False,
                                  find_unused_parameters=False, gradient_as_bucket_view=True)
    # Initial parameters match; dropout streams should differ across replicas.
    torch.manual_seed(20261022 + rank)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=.01)
    result = {'config': vars(args), 'world_size': world, 'model_revision': MODEL_REVISION,
              'manifest_sha256': cache.metadata['manifest_sha256'],
              'duplicate_sessions_in_global_batch': len(rows) < world, 'steps': []}
    model.train()
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        row = rows[(step * world + rank) % len(rows)]
        item = cache.item(row)
        optimizer.zero_grad(set_to_none=True)
        dist.barrier()
        torch.cuda.synchronize()
        start = time.monotonic()
        loss = ddp(item)
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)
        assert torch.isfinite(norm) and norm > 0
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        record = dict(rank=rank, sample_id=row['id'], seconds=row['audio_duration'],
                      tokens=row['token_length'], loss=loss.item(), grad_norm=norm.item(),
                      wall_seconds=elapsed, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        records = [None] * world
        dist.all_gather_object(records, record)
        if rank == 0:
            event = dict(step=step+1, ranks=records,
                         wall_seconds=max(r['wall_seconds'] for r in records),
                         audio_seconds=sum(r['seconds'] for r in records))
            result['steps'].append(event)
            print(json.dumps(event), flush=True)
            with (out/'events.jsonl').open('a') as f:
                f.write(json.dumps(event)+'\n')
            if args.save:
                model.save_pretrained(out/f'checkpoint-{step+1}', safe_serialization=True)
    # Check that synchronized updates left adapter parameters equal across ranks.
    max_diff = torch.zeros((), device='cuda')
    for param in trainable:
        other = param.detach().clone()
        dist.broadcast(other, src=0)
        max_diff = torch.maximum(max_diff, (param.detach()-other).abs().max())
    dist.all_reduce(max_diff, op=dist.ReduceOp.MAX)
    assert max_diff.item() == 0, f'Adapters diverged: {max_diff.item()}'
    if rank == 0:
        measured = result['steps'][1:] or result['steps']
        speed = sum(r['audio_seconds'] for r in measured)/sum(r['wall_seconds'] for r in measured)
        result.update(status='completed', max_adapter_difference=max_diff.item(),
                      audio_realtime_factor=speed, hours_per_500h_epoch=500/speed)
        (out/'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
