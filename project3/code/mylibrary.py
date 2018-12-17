


def count_pixels(img_gray, include_border=False):
    """
    Purpose:
        Count the number of pixels
    Input:
        img_gray: matrix with value type real
        include_border: boolean
    """
    def count_aux(border_width):
        stat = {}
        for i in range(border_width, img_gray.shape[0] - border_width):
            for j in range(border_width, img_gray.shape[1] - border_width):
                if img_gray[i,j] not in stat:
                    stat[img_gray[i,j]] = 0
                stat[img_gray[i,j]] += 1
        return stat
    
    if include_border:
        return count_aux(0)
    else:
        return count_aux(1)