import numpy as np


def overlay_transparent(background, overlay, pos_overlay):
    background_width = background.shape[1]
    background_height = background.shape[0]

    x, y = tuple(pos_overlay)

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype)
                * 255,
            ],
            axis=2,
        )

    crop_top = max(y, 0)
    crop_left = max(x, 0)
    crop_bottom = min(y + h, background_height)
    crop_right = min(x + w, background_width)

    crop_overlay_top = crop_top - y
    crop_overlay_bottom = crop_overlay_top + crop_top + crop_bottom
    crop_overlay_left = crop_left - x
    crop_overlay_right = crop_overlay_left + crop_left + crop_right

    overlay_image = overlay[
        crop_overlay_top:crop_overlay_bottom, crop_overlay_left:crop_overlay_right, :3
    ]
    mask = (
        overlay[
            crop_overlay_top:crop_overlay_bottom,
            crop_overlay_left:crop_overlay_right,
            3:,
        ]
        / 255.0
    )

    try:
        background[
            crop_top : crop_top + overlay_image.shape[0],
            crop_left : crop_left + overlay_image.shape[1],
        ] = (
            (1.0 - mask)
            * background[
                crop_top : crop_top + overlay_image.shape[0],
                crop_left : crop_left + overlay_image.shape[1],
            ]
            + mask * overlay_image
        )
    except Exception as e:
        print(str(e))

    return background
