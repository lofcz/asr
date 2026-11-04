import pathlib, tempfile, unittest
from contracts import atomic_json,sha
from checkpoint_archive import prune


class ArchiveTests(unittest.TestCase):
    def test_unbacked_checkpoints_survive_and_two_latest_stay_local(self):
        with tempfile.TemporaryDirectory() as root:
            run=pathlib.Path(root)/'runs/r64-full';run.mkdir(parents=True)
            dirs=[]
            for i in range(4):
                p=run/f'checkpoint-{i:06d}';p.mkdir();dirs.append(p)
                for n in ('state.pt','adapter_model.safetensors'):(p/n).write_bytes(b'checkpoint payload')
            prune(run);self.assertTrue(all(p.exists() for p in dirs))
            for p in dirs:
                atomic_json(run/'backup-receipts'/(p.name+'.json'),{'source':str(p),'destination':'SSD verified copy',
                            'files':{f.name:sha(f) for f in p.iterdir()}})
            prune(run)
            self.assertFalse(dirs[0].exists());self.assertFalse(dirs[1].exists())
            self.assertTrue(dirs[2].exists());self.assertTrue(dirs[3].exists())

    def test_never_prune_other_run_namespaces(self):
        with self.assertRaisesRegex(RuntimeError,'namespace'):prune('/tmp/previous-run')


if __name__=='__main__':unittest.main()
