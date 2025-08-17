# Runs Cellpose-SAM on 3D-tiff files (nuclear channel)
import os
import time
import numpy as np
import argparse
import traceback
import glob
from skimage.color import label2rgb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_dir", help="Input directory for 3D Tiffs", required=True
    )
    parser.add_argument("-o", "--output", help="Output directory for 3D masks", required=True)
    parser.add_argument("-b", "--backend", help="Backend: cpu,gpu", default="gpu")
    parser.add_argument("-c", "--channels", help="Name of file for nuc channel e.g. 'nucleus.p.tif'", required=True)
    parser.add_argument("-z", "--colorize", help="colorize output", action='store_true')
    parser.add_argument("-s", "--stop", help="max volumes to process (integer)", default=100)

    args = parser.parse_args()

    volume_subdirs = []
    inferred_fofs = [os.path.basename(f).split("_inference")[0] for f in glob.glob(os.path.join(args.output, "FoF*.tif"))]
    print(inferred_fofs)
    try:
        if not os.path.isdir(args.input_dir):
            raise FileNotFoundError(f"Image base directory not found: {args.input_dir}")
        volume_subdirs = sorted(
            [
                d
                for d in os.listdir(args.input_dir)
                if os.path.isdir(os.path.join(args.input_dir, d))
                and d not in inferred_fofs
            ]
        )
        print(
            f"Found {len(volume_subdirs)} potential volume directories in {args.input_dir}."
        )
        if not volume_subdirs:
            print(
                "Warning: No subdirectories found. Check the args.input_dir path and contents."
            )

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while listing directories: {e}")
    if not volume_subdirs:  # Check if variable from Cell 1 exists and is not empty
        print(
            "No volume subdirectories found (list 'volume_subdirs' is empty or was not created in Cell 1). Exiting batch processing."
        )


    else:
    ###Cellpose initialization
        try:
            from cellpose import models
            from cellpose import io

            print("Cellpose library imported successfully.")
        except ImportError:
            print("🛑 Cellpose library not found. Please install it (pip install cellpose).")
            print("🛑 Prediction step will be skipped.")
        processed_volume_count = 0
        # Set how many volumes to process (e.g., 5 for testing, None for all)
        max_volumes_to_process = args.stop  # Set to None to process all found volumes

        cellpose_model = models.CellposeModel(gpu=args.backend=='gpu')
        print(
            f"\nStarting batch processing loop for up to {max_volumes_to_process if max_volumes_to_process is not None else len(volume_subdirs)} volumes..."
        )

        for volume_index, volume_name in enumerate(volume_subdirs):
            # Check processing limit
            if (
                max_volumes_to_process is not None
                and processed_volume_count >= max_volumes_to_process
            ):
                print(
                    f"\nReached processing limit ({max_volumes_to_process}). Stopping."
                )
                break

            print(
                f"\n--- Processing Volume {processed_volume_count + 1}/{max_volumes_to_process or len(volume_subdirs)}: {volume_name} ---"
            )

            # Reset variables for this volume
            image = None
            img_h, img_w, img_d = None, None, None  # Initialize dimensions

            ###Read 3D TIFF image
            try:
                # STEP 1: Assemble Image
                print("(Step 1) Assembling Image...")
                start_time_img = time.time()
                sample_img_dir_path = os.path.join(args.input_dir, volume_name)
                # for 2d series -> 3D files- image, img_h, img_w, img_d = assemble_3d_image(volume_img_dir_path)
                image = io.imread(
                    os.path.join(sample_img_dir_path, args.channels )
                )
                #If 2D, add an z_axis
                image2d = image.ndim ==2

                if not image2d:
                    image=image.transpose(1, 2, 0)
                if image is None:
                    print(f"  Skipping volume {volume_name}: Image assembly failed.")
                    # No need to increment processed_volume_count here, finally block handles it
                    continue  # Skip to the next volume in the finally block
                img_assembly_time = time.time() - start_time_img
                print(
                    f"  Image assembly took {img_assembly_time:.2f} seconds. Image Shape (H,W,D): {np.shape(image)})"
                )
            except Exception as e:
                print(f"Unable to read {volume_name}")
                print(e)

            start_time_cellpose = time.time()
            try:
                print("  Waiting for Cellpose inference result...")
                if image2d:
                    masks, _, _ = cellpose_model.eval(
                        image,
                        do_3D=False,
                        batch_size=64,
                    )
                else:
                    masks, _, _ = cellpose_model.eval(
                        image,
                        do_3D=True,
                        batch_size=16,
                        z_axis=2,  # cellpose.io automatically does z-last
                        channel_axis=None,
                        flow3D_smooth=1,
                    )

                print("  Cellpose inference finished.")
                if masks is None:
                    pred_error = "Cellpose inference function returned None"
            except Exception as e_pred:
                print(
                    f"!! Error during threaded Cellpose inference for {volume_name}: {e_pred}"
                )
                traceback.print_exc()
            print(f" Inference took {time.time() - start_time_cellpose:.2f} seconds.")

            if args.colorize and masks.size>0:
                masks_for_color = masks if masks.ndim == 3 else masks[np.newaxis, :, :]
                masks_colorized = label2rgb(masks_for_color, bg_label=0)
                path_save_colorized=os.path.join(args.output,volume_name+'_colorized.tif')
                io.imsave(
                    path_save_colorized,
                    (masks_colorized * 255).astype(np.uint8),
                    #imagej=True,
                )

            path_save_masks=os.path.join(args.output,volume_name+'_masks.tif')
            io.imsave(
                path_save_masks,
                masks
            )

            #print(f"{"Colorized" if args.colorize else ""}Image saved to {path_save_colorized}")
            processed_volume_count += 1


if __name__ == "__main__":
    main()
