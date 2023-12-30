import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor



class SAMController:
    def __init__(self, SAM_checkpoint, device='cuda:0', verbose=True):
        """
        device: model device
        SAM_checkpoint: path of SAM checkpoint
        """
        if verbose:
            print(f"Initializing SAM to {device}")

        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.model = sam_model_registry['vit_h'](checkpoint=SAM_checkpoint)
        self.model.to(device=self.device)
        self.predictor = SamPredictor(self.model)
        self.embedded = False


    @torch.no_grad()
    def set_image(self, image: np.ndarray):
        # PIL.open(image_path) 3channel: RGB
        # image embedding: avoid encode the same image multiple times
        self.orignal_image = image
        if self.embedded:
            print('repeat embedding, please reset_image.')
            return
        self.predictor.set_image(image)
        self.embedded = True
        return
    
    @torch.no_grad()
    def reset_image(self):
        # reset image embeding
        self.predictor.reset_image()
        self.embedded = False

    @torch.no_grad()
    def predict(self, click_coords=None, click_labels=None, bbox=None, mask_input=None, multimask_output=True):
        """
        image: numpy array, h, w, 3
        click_coords: Optional[torch.Tensor],
        click_labels: Optional[torch.Tensor],
        box: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None, MUST be sam mask

        """
        assert self.embedded, 'prediction is called before set_image (feature embedding).'
        
        masks, scores, logits = self.predictor.predict(point_coords=click_coords, 
            point_labels=click_labels,
            box=bbox,
            mask_input=mask_input,
            multimask_output=multimask_output)
        

        # masks (n,1, h, w), scores (n,), logits (n, 256, 256)
        sam_masks = torch.from_numpy(masks).unsqueeze(1).cuda()
        return sam_masks, scores, logits
