import copy
import unittest
import torch
from vibevoice.modular.configuration_vibevoice import VibeVoiceAcousticTokenizerConfig
from vibevoice.modular.modular_vibevoice_tokenizer import VibeVoiceAcousticTokenizerModel
from audio_path import encode_means, receptive_field

class DifferentiableEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(19)
        cfg=VibeVoiceAcousticTokenizerConfig(encoder_n_filters=2,decoder_n_filters=2,
             encoder_depths='1-1-1-1-1-1-1',vae_dim=4,layer_scale_init_value=.1)
        cls.encoder=VibeVoiceAcousticTokenizerModel(cfg).encoder.float()

    def test_tiling_preserves_partial_final_frame_and_outputs(self):
        for samples in [3200*35+17,3200*80+83]:
            x=torch.randn(1,1,samples)
            with torch.no_grad():
                full=self.encoder(x).transpose(1,2)
                tiled=encode_means(self.encoder,x,tile_seconds=2.,checkpoint_activations=False)
            self.assertEqual(full.shape,tiled.shape)
            torch.testing.assert_close(full,tiled,rtol=2e-5,atol=2e-6)

    def test_gradients_cross_tile_boundaries_without_detach(self):
        a=copy.deepcopy(self.encoder);b=copy.deepcopy(self.encoder)
        x=torch.randn(1,1,3200*80+83)
        full=a(x).transpose(1,2);full.square().sum().backward()
        tiled=encode_means(b,x,tile_seconds=2.,checkpoint_activations=True);tiled.square().sum().backward()
        torch.testing.assert_close(full,tiled,rtol=2e-5,atol=2e-6)
        for (name,p),(other,q) in zip(a.named_parameters(),b.named_parameters()):
            self.assertEqual(name,other);self.assertIsNotNone(p.grad);self.assertIsNotNone(q.grad)
            torch.testing.assert_close(p.grad,q.grad,rtol=5e-4,atol=2e-5,msg=name)

    def test_upstream_encode_is_detached_but_new_path_is_not(self):
        x=torch.randn(1,1,3200*10)
        self.assertTrue(encode_means(self.encoder,x).requires_grad)
        span,stride=receptive_field(self.encoder)
        self.assertEqual(stride,3200);self.assertGreater(span,stride)

if __name__=='__main__':unittest.main()
