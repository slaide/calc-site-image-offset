"""
calculate alignment values for fluorescence channels

intended for this use case:
you imaged the same plate twice, but you have taken the plate off the microscope in between runs.
this can lead to a small-ish misalignment between runs, up to a couple hundred pixels. (i.e. 0.1mm)
this code calculates an offset value in X and Y to align those runs, if there is at least one channel present in both.

this code currently uses the 405nm channel for alignment, though any other channel should also work.
the alignment score used for this is normalized cross-correlation, hence image data independent (does not require
a specific cell organelle to be visible, or the contrast to have a specific value, or anything like that)

the code performs a parallel grid search, and increases the grid size until some termination condition.

the output for the automatic offset fitting is written to a parquet file.
you can view the contents of this file in the command line, e.g. with duckdb

$ duckdb
select * from "results.parquet" limit 10 ;

"""

import cv2
import time
import glob
import pandas as pd
from dataclasses import dataclass, asdict
from pathlib import Path

from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor
from itertools import product

INTERACTIVE=False
SCORE_MEANSQUAREERROR=True
USE_GPU=False

if USE_GPU:
    if not SCORE_MEANSQUAREERROR:
        print("warning - gpu compute does not support normalized cross correlation")
        SCORE_MEANSQUAREERROR=True

    if INTERACTIVE:
        print("warning - cannot use gpu compute with interactive mode")
        USE_GPU=False

if USE_GPU:
    try:
        import mlx.core as np
    except:
        try:
            import cupy as np
        except:
            import numpy as np

    DTYPE=np.float32
else:
    import numpy as np
    DTYPE=np.uint8

def read_image(path:str)->np.array:
    ret=cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    return np.array(ret,dtype=DTYPE)

def calculate_score(img1:cv2.Mat, img2:cv2.Mat)->float:
    if not SCORE_MEANSQUAREERROR:
        # normalization cross-correlation (1 at best fit case)

        ret=np.sum(img1 * img2) / np.sqrt(np.sum(img1 ** 2) * np.sum(img2 ** 2))
        ret=float(ret)
        #print(f"{ret=}")
        return ret

    else:
        # mean square error (0 at best fit case)

        ret=np.sum((img1-img2)**2)
        return float(ret)

def get_acq_id(p:str):
    segments=p.split("_")
    n=0
    while True:
        if segments[n]=="acqid":
            n+=1
            break

        n+=1

    return int(segments[n])

# Load an arbitrary file as grayscale
plates=[
    "P101390",
    "P101389",
    "P106081",

    #P106075
    #P106070
    #P106080
    #P106074
    #P106069
]

def crop_images(img1, img2, x, y):
    """
    Crop overlapping regions of img1 and img2 based on offsets x and y.

    Args:
        img1 (np.ndarray): The first image.
        img2 (np.ndarray): The second image.
        x (int): Horizontal offset (positive x crops from the left).
        y (int): Vertical offset (positive y crops from the top).

    Returns:
        np.ndarray, np.ndarray: Cropped versions of img1 and img2 with the same overlapping region.
    """
    global USE_MX

    h, w = img1.shape

    # Calculate cropping indices
    row_start = max(0, y)
    row_end = h - max(0, -y)
    col_start = max(0, x)
    col_end = w - max(0, -x)

    if 1:
        # center crop
        # -> constant image size for (practically) all offsets

        img2_rolled=np.roll(np.roll(img2, x, axis=1), y, axis=0)
        img1_cropped = img1[750:-750,750:-750]
        img2_cropped = img2_rolled[750:-750,750:-750]

    else:
        # crop to overlap
        # -> changing image size
        img1_cropped = img1[row_start:row_end, col_start:col_end]
        img2_cropped = img2[row_start:row_end, col_start:col_end]

    return img1_cropped, img2_cropped

def automatic_find_offsets(out_file_path:str="results.parquet"):
    @dataclass
    class Result:
        plate:str
        well:str
        site:int
        score:float
        dx:int
        dy:int
        time_s:float

    results:list[Result]=[]
    try:
        for i in tqdm(range(len(plates))):
            for well in tqdm("D10 A05 O22 A22 D22 O05".split(" ")):
                start_time=time.time()

                img_paths=glob.glob(f"/Users/pathe605/Downloads/img_{plates[i]}_acqid_*_{well}_site_1_merged.png")
                if len(img_paths)!=2:
                    print(f"skipping, because: len({img_paths})!=2")
                    continue
                        
                img_paths=sorted(img_paths,key=lambda p:get_acq_id(p))

                print(img_paths)
                img1 = read_image(img_paths[0])
                img2 = read_image(img_paths[1])

                def calculate_offset2d(img1,img2):
                    IMG_SIZE_X=img1.shape[0]
                    IMG_SIZE_Y=img1.shape[1]

                    def estimate_offset2d(
                        img1,img2,
                        bounds_x:tuple[int,int]|list[int],
                        bounds_y:tuple[int,int]|list[int],

                        solutions:dict[tuple[int,int],float]|None=None,
                    ):
                        """
                        params:
                            solutions: store scores in here to avoid re-calculating the score for a combination of offset parameters.
                        """

                        solutions=solutions or dict()

                        def calculate_shifted_score(x,y):
                            new_score=solutions.get((x,y))                        
                                                
                            if new_score is None:
                                new_score=calculate_score(*crop_images(img1, img2, x, y))

                                solutions[(x,y)]=new_score

                        min_score=1e9
                        min_x,min_y=0,0

                        # Create the grid of all x, y combinations
                        if type(bounds_x)==type((2,)):
                            bounds_x=list(range(bounds_x[0],bounds_x[-1]+1))
                        if type(bounds_y)==type((2,)):
                            bounds_y=list(range(bounds_y[0],bounds_y[-1]+1))

                        grid = list(product(bounds_x, bounds_y))

                        # calculate results in parallel, using tqdm(executor.map()) for a dynamic progress bar
                        # and list(tqdm()) to force tqdm to iterate over the results
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            list(tqdm(executor.map(lambda args: calculate_shifted_score(*args), grid),total=len(grid)))

                        for x in bounds_x:
                            for y in bounds_y:
                                new_score=solutions.get((x,y))
                                assert new_score is not None, f"score for {(x,y)=} is None"

                                if new_score<min_score:
                                    min_score=new_score
                                    min_x,min_y=x,y

                        return min_score,min_x,min_y

                    dx=0
                    dy=0
                    best_score=1e9 # some arbitrary large number
                    last_score=best_score-1

                    solutions:dict[tuple[int,int],float]=dict()

                    bounds_x=[]
                    bounds_y=[]

                    last_min_on_boundary=False
                    num_iter=0

                    PADDING=12

                    while True:
                        num_iter+=1

                        # on odd iterations, scan a new coarse grid
                        # (less coarse on later iterations)
                        if num_iter%2==1:
                            bounds_x=list(range(-300,300,(PADDING//num_iter) or 1))
                            bounds_y=list(range(-300,300,(PADDING//num_iter) or 1))

                        # on even iterations, refine the grid
                        # (larger refine range on later iterations)
                        else:
                            bounds_x=list(range(dx-int(PADDING*(num_iter-1)*1.5),dx+int(PADDING*(num_iter-1))))
                            bounds_y=list(range(dy-int(PADDING*(num_iter-1)*1.5),dy+int(PADDING*(num_iter-1))))

                        new_score,dx,dy=estimate_offset2d(img1,img2,bounds_x=bounds_x,bounds_y=bounds_y,solutions=solutions)

                        print(f"local solution: score={new_score}, {dx=} {dy=}")

                        # if we have just refined the local solution
                        if num_iter%2==0:
                            # if the new solution is better than the last local best, save it
                            if new_score<best_score:
                                best_score=new_score

                                # stop after one refinement
                                break

                            # if we have not found an optimal solution after several iterations, give up
                            if num_iter>5:
                                break

                    return best_score,dx,dy

                best_score,dx,dy=calculate_offset2d(img1,img2)

                print(f"min score: {best_score}, {dx=} {dy=} (plate {plates[i]}, {well=})")

                results.append(Result(plate=plates[i],well=well,site=1,score=best_score,dx=dx,dy=dy,time_s=time.time()-start_time))

    finally:
        res_df=pd.DataFrame.from_records([asdict(r) for r in results])
        print(res_df.to_string())

        if Path(out_file_path).exists():
            out_file_path=out_file_path[:-len(".parquet")]
            ending_num=2
            last_num=None

            while Path(out_file_path+f".{ending_num}.parquet").exists():
                if len(out_file_path.split("."))>=2:
                    try:
                        existing_ending_num=int(list(out_file_path.split("."))[-2])
                        if ending_num<existing_ending_num:
                            ending_num=existing_ending_num
                    except:
                        pass

                ending_num+=1

            out_file_path+=f".{ending_num}.parquet"

        print(f"writing to {out_file_path}")
        res_df.to_parquet(out_file_path)

def interactive_offset2d(
    plate_index:int=1,
    well_name:str="A05",

    OFFSET_X_MIN = -300,
    OFFSET_Y_MIN = -300,
    OFFSET_X_MAX = 300,
    OFFSET_Y_MAX = 300,
    WINDOW_SIZE_X = 1400,
    WINDOW_SIZE_Y = 800,
    WINDOW_HANDLE = "Align (press esc|q to quit)",
):
    OFFSET_X_RANGE = OFFSET_X_MAX - OFFSET_X_MIN
    OFFSET_Y_RANGE = OFFSET_Y_MAX - OFFSET_Y_MIN

    i=plate_index
    well=well_name

    print(f"checking {plates[i]} : {well=}")
    img_paths=glob.glob(f"/Users/pathe605/Downloads/img_{plates[i]}_acqid_*_{well}_site_1_merged.png")
    if len(img_paths)!=2:
        print(f"skipping, because: len({img_paths})!=2")
        raise RuntimeError()
            
    img_paths=sorted(img_paths,key=lambda p:get_acq_id(p))

    print(img_paths)
    img1 = read_image(img_paths[0])
    img2 = read_image(img_paths[1])

    IMG_SIZE_X=img1.shape[0]
    IMG_SIZE_Y=img1.shape[1]

    dx=0
    dy=0

    def add_outline(image:cv2.Mat, thickness:int=5)->cv2.Mat:
        """Add a white outline to the image."""
        h, w = image.shape
        outlined = np.zeros((h + 2 * thickness, w + 2 * thickness), dtype=np.uint8)
        outlined[thickness:-thickness, thickness:-thickness] = image
        return cv2.copyMakeBorder(image, thickness, thickness, thickness, thickness, cv2.BORDER_CONSTANT, value=255)

    def overlay_images(img1:cv2.Mat, img2:cv2.Mat)->cv2.Mat:
        """Overlay two images with transparency."""
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)  # Ensure both are RGB
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

    def calculate_monochrome_overlay(img1:cv2.Mat, img2:cv2.Mat)->cv2.Mat:
        """Calculate the difference image as a monochrome image."""
        diff = img1.astype(np.int16) - img2.astype(np.int16)
        # Normalize the difference to fit within the 0-255 range
        diff_normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        return diff_normalized.astype(np.uint8)

    def resize_to_match(image:cv2.Mat, target_shape:tuple[int,int])->cv2.Mat:
        """Resize or pad the image to match the target shape."""
        target_h, target_w = target_shape
        h, w = image.shape[:2]

        if h > target_h or w > target_w:
            # Crop to target shape
            return image[:target_h, :target_w]
        else:
            # Pad to target shape
            pad_h = target_h - h
            pad_w = target_w - w
            return cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

    # Initial zoom region
    zoom_rect = [0, 0, None, None]  # [x_start, y_start, x_end, y_end]
    dragging = False
    reset_zoom = True  # Used to reset zoom when R is pressed

    def get_zoom_rect():
        nonlocal zoom_rect
        if zoom_rect[2] is not None and zoom_rect[3] is not None:
            ret=[a for a in zoom_rect]
            ret[0], ret[1] = min(zoom_rect[0], zoom_rect[2]), min(zoom_rect[1], zoom_rect[3])
            ret[2], ret[3] = max(zoom_rect[0], zoom_rect[2]), max(zoom_rect[1], zoom_rect[3])
            return ret
        else:
            return zoom_rect

    # Mouse event handler
    def mouse_callback(event, x, y, flags, param):
        nonlocal zoom_rect, dragging

        panel_width = IMG_SIZE_X
        panel_height = IMG_SIZE_Y

        # Map mouse coordinates to panel coordinates
        panel_x = x % panel_width
        panel_y = y % panel_height

        if event == cv2.EVENT_LBUTTONDOWN and not dragging:
            zoom_rect[0], zoom_rect[1] = panel_x, panel_y  # Start point
            dragging = True
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            zoom_rect[2], zoom_rect[3] = panel_x, panel_y  # Update end point
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
            # Ensure valid zoom rectangle
            zoom_rect=get_zoom_rect()

        draw()

    def crop_to_zoom(image, rect):
        """Crop the image to the zoom rectangle."""
        x_start, y_start, x_end, y_end = rect
        if x_end is None or y_end is None:  # No zoom applied
            return image

        # Ensure cropping coordinates are within bounds
        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end = min(image.shape[1], x_end)
        y_end = min(image.shape[0], y_end)

        return image[y_start:y_end, x_start:x_end]

    raw_display=None
    font_scale=1
    score=1e9

    def on_trackbar_overlay(val):
        nonlocal dx, dy

        dx = cv2.getTrackbarPos('X Offset', WINDOW_HANDLE) + OFFSET_X_MIN
        dy = cv2.getTrackbarPos('Y Offset', WINDOW_HANDLE) + OFFSET_Y_MIN

        calc_display()
        draw()

    img1_rgb,aligned_rgb,colored_overlay,overlay_panel=None,None,None,None

    def get_raw_display():
        nonlocal font_scale,dragging,zoom_rect,img1_rgb,aligned_rgb,colored_overlay,overlay_panel

        if len([i for i in [img1_rgb,aligned_rgb,colored_overlay,overlay_panel] if i is None]):
            return None
        
        # Apply zoom crop to all panels
        font_scale=4
        if not dragging:
            img1_zoomed = crop_to_zoom(img1_rgb, zoom_rect)
            aligned_zoomed = crop_to_zoom(aligned_rgb, zoom_rect)
            colored_zoomed = crop_to_zoom(colored_overlay, zoom_rect)
            overlay_zoomed = crop_to_zoom(overlay_panel, zoom_rect)

            rect=get_zoom_rect()
            if rect[2] is not None:
                font_scale/=IMG_SIZE_X/(rect[2]-rect[0])
        else:
            img1_zoomed = (img1_rgb)
            aligned_zoomed = (aligned_rgb)
            colored_zoomed = (colored_overlay)
            overlay_zoomed = (overlay_panel)

        # Stack the panels in a 2x2 grid
        top_row = np.hstack((img1_zoomed, aligned_zoomed))
        bottom_row = np.hstack((colored_zoomed, overlay_zoomed))
        raw_display = np.vstack((top_row, bottom_row))

        return raw_display

    def calc_display():
        nonlocal raw_display, font_scale, score, img1, img2, img1_rgb,aligned_rgb,colored_overlay,overlay_panel

        img1_aligned,img2_aligned=crop_images(img1, img2, dx, dy)
        score=calculate_score(img1_aligned,img2_aligned)
        aligned_outlined = add_outline(img2_aligned)

        # Convert outlined images to RGB for consistent stacking
        img1_rgb = cv2.cvtColor(img1_outlined, cv2.COLOR_GRAY2BGR)
        aligned_rgb = cv2.cvtColor(aligned_outlined, cv2.COLOR_GRAY2BGR)

        # Create the monochrome difference overlay and apply a colormap
        diff_overlay = calculate_monochrome_overlay(img1_aligned,img2_aligned)
        colored_overlay = cv2.applyColorMap(diff_overlay, cv2.COLORMAP_JET)

        # Create the direct overlay
        overlay_panel = overlay_images(img1_aligned,img2_aligned)

        # Resize all images to match the largest dimensions
        target_shape = img1_rgb.shape[:2]
        aligned_rgb = resize_to_match(aligned_rgb, target_shape)
        colored_overlay = resize_to_match(colored_overlay, target_shape)
        overlay_panel = resize_to_match(overlay_panel, target_shape)

    def draw(
        ZOOM_RECT_COLOR=(0,0,0xff),
        TEXT_COLOR=(0xfe,0xfe,0xfe),
    ):
        """ colors in BGR format! """
        nonlocal dx,dy, font_scale, score

        display=get_raw_display()
        if display is None:
            return

        display=display.copy()

        # Draw the rectangle indicator
        if dragging and zoom_rect[2] is not None and zoom_rect[3] is not None:
            rect=get_zoom_rect()
            for x in range(2):
                for y in range(2):
                    cv2.rectangle(display, (rect[0]+IMG_SIZE_X*x, rect[1]+IMG_SIZE_Y*y), (rect[2]+IMG_SIZE_X*x, rect[3]+IMG_SIZE_Y*y), ZOOM_RECT_COLOR, 2)

        cv2.putText(display, f"Score: {score:.4f} - {dx=} {dy=}", (10, int(30*font_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, TEXT_COLOR, 2)
        cv2.imshow(WINDOW_HANDLE, display)

    # Add outlines
    img1_outlined = add_outline(img1)
    img2_outlined = add_outline(img2)

    print(f"{(dx,dy,OFFSET_X_RANGE,OFFSET_Y_RANGE,dy + OFFSET_Y_RANGE // 2)=}")

    # Setup GUI
    cv2.namedWindow(WINDOW_HANDLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_HANDLE, WINDOW_SIZE_X, WINDOW_SIZE_Y)  # Adjust the window size
    cv2.setMouseCallback(WINDOW_HANDLE, mouse_callback)
    cv2.createTrackbar('X Offset', WINDOW_HANDLE, dx + OFFSET_X_RANGE // 2, OFFSET_X_RANGE, on_trackbar_overlay)
    #cv2.createTrackbar('Y Offset', WINDOW_HANDLE, dy + OFFSET_Y_RANGE // 2, OFFSET_Y_RANGE, on_trackbar_overlay)
    cv2.createTrackbar('Y Offset', WINDOW_HANDLE,300,600, on_trackbar_overlay)

    try:
        # Event loop
        frame_num = 0
        while True:
            frame_num += 1
            if frame_num<5:
                cv2.resizeWindow(WINDOW_HANDLE, 100,100)
                cv2.resizeWindow(WINDOW_HANDLE, WINDOW_SIZE_X, WINDOW_SIZE_Y)
            
                draw()

            # Handle keypresses
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"): # ESC key to exit
                break

            elif key == ord('r'):  # Reset zoom
                # Reset zoom

                dragging=False # force disable dragging
                zoom_rect[:] = [0, 0, None, None]

                draw()

            elif key == ord('d'):  # force draw
                cv2.resizeWindow(WINDOW_HANDLE, 100, 100)

                draw()

            if cv2.getWindowProperty(WINDOW_HANDLE, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()
    finally:
        print(f"{plates[i]} : {score=} at {dx=} {dy=}")


if __name__=="__main__":
    if not INTERACTIVE:
        automatic_find_offsets()
    else:
        interactive_offset2d(0,"D10")