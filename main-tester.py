from data_generator import DataGenerator

gen = DataGenerator(upsampling_factor=8,
                    camera_pixel_size=100,
                    num_patches=500,
                    patch_before_upsample=26,
                    min_emitters=7,
                    max_examples=10000,
                    working_dir=r"../Deep-STORM/demo 1 - Simulated Microtubules",
                    tiff_file_name=r"../Deep-STORM/demo 1 - Simulated Microtubules\ArtificialDataset_demo1.tif",
                    csv_file_name=r"../Deep-STORM/demo 1 - Simulated Microtubules\positions_demo1.csv")
gen.generate()

