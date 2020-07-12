from .read_stack_from_tiff import read_stack_from_tiff
from .gaussian_kernel_2d import gaussian_kernel_2d
import cv2
from pandas import read_csv
import numpy as np
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
        self.working_dir = working_dir
        self.tiff_file_name = tiff_file_name
        self.frames_stack = None
        self.frames_number = None
        self.frame_width = None
        self.frame_height = None
        self.upsampled_frame_height = None
        self.upsampled_frame_width = None
        self.pixel_size_to_upsampling_factor_ratio = camera_pixel_size/upsampling_factor
        self.number_of_training_patches = None

    def generate(self):
        self.frames_stack = read_stack_from_tiff(tifFilename=self.tiff_file_name)
        self.frames_number = len(self.frames_stack)
        self.frame_height,self.frame_width = self.frames_stack[0].shape
        self.upsampled_frame_height = self.frame_height*self.upsampling_factor
        self.upsampled_frame_width = self.frame_width*self.upsampling_factor
        gaussian_kernel = gaussian_kernel_2d(7,1)
        self.number_of_training_patches = min(self.frames_number*self.num_patches,self.max_examples)

        patches = np.zeros((self.patch_size,self.patch_size,self.number_of_training_patches))
        heatmaps = np.zeros((self.patch_size,self.patch_size,self.number_of_training_patches))
        spikes = np.zeros((self.patch_size,self.patch_size,self.number_of_training_patches))
        csv_file_reader = read_csv(self.csv_file_name)


        for idx, frame in enumerate(self.frames_stack):
            frame_up_sampled = cv2.resize(frame,None,fx=self.upsampling_factor,fy = self.upsampling_factor,interpolation=cv2.INTER_NEAREST)
            filtered_data = csv_file_reader[csv_file_reader["frame"] == idx+1] # frames counting start from 1
            xs = np.array([max(min(x / self.pixel_size_to_upsampling_factor_ratio, self.upsampled_frame_width - 1), 0) for x in filtered_data["x [nm]"].tolist()]).astype(np.int)
            ys = np.array([max(min(y / self.pixel_size_to_upsampling_factor_ratio, self.upsampled_frame_height - 1), 0) for y in filtered_data["y [nm]"].tolist()]).astype(np.int)
            spikes_image = np.zeros((self.upsampled_frame_width,self.upsampled_frame_height))
            spikes_image[xs,ys] = 1
            heat_map_tmp = cv2.filter2D(spikes_image,-1,gaussian_kernel)

        # for index, row in csv_file_reader.iterrows():




