from PIL import Image



class CleaningBase:
    def __init__(self, inpainter):
        self.inpainter = inpainter


    def batch_cleaning(self, image_pil: Image.Image, image_mask: Image.Image) -> Image.Image:
        return self.inpainter.inpaint(image_pil, image_mask)

