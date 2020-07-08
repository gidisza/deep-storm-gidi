from .read_stack_from_tiff import read_stack_from_tiff
class DataGenerator:
    def __init__(self,working_dir,tiff_file_name,csv_file_name,num_patches,camera_pixel_size,patch_before_upsample,upsampling_factor,max_examples,min_emitters):
        self.min_emitters = min_emitters
        self.max_examples = max_examples
        self.upsampling_factor = upsampling_factor
        self.num_patches = num_patches
        self.camera_pixel_size = camera_pixel_size
        self.patch_before_upsample = patch_before_upsample
        self.patch_size = self.upsampling_factor * self.patch_before_upsample
        self.csv_file_name = csv_file_name
        self.working_dir  = working_dir
        self.tiff_file_name = tiff_file_name
        self.frames_stack = None
        self.frames_number = None
        self.frame_width = None
        self.frame_height = None
        self.upsampled_frame_height = None
        self.upsampled_frame_width = None
        self.pixel_size_to_upsampling_factor_ratio = camera_pixel_size/upsampling_factor

    def generate(self):
        self.frames_stack = read_stack_from_tiff(tifFilename=self.tiff_file_name)
        self.frames_number = len(self.frames_stack)
        self.frame_height,self.frame_width = self.frames_stack[0].shape
        self.upsampled_frame_height = self.frame_height*self.upsampling_factor
        self.upsampled_frame_width = self.frame_width*self.upsampling_factor
